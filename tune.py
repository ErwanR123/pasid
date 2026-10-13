"""Hyperparameter tuning with Optuna : one TPE study per category, model dimension and
arm, the same number of trials for every study, on the validation HR@10 at the end of
the training budget (no pruning : every trial trains for the whole budget).

Runs on the cluster, so the queue does not depend on the laptop or the VPN :
    python submit.py tune.py
An arm is an experiment config (configs/experiment/) : its encoder and SID candidate are
kept, Optuna sets its lr, weight decay and dropout in the search space of its family
(id, or the SID arms), starting from the values of the config. Trial i runs as
task_name=tune_<category>_d<d>_<arm>_<i>.
The categories run one after the other, in the order of the config : a category gets the
free GPUs once every study of the previous one has started all its trials. Inside it,
the study with the fewest trials starts the next one. A trial that fails (lost node,
HDFS) runs again once. An interrupted tuning starts again from scratch.
"""

import json
import os
import time
from datetime import datetime

import fsspec
import hydra
import optuna
from omegaconf import OmegaConf
from ray.job_submission import JobStatus, JobSubmissionClient

from src.test import last_checkpoint, list_runs
from src.utils.tools import patch_fsspec
from submit import submit


def space(config, arm, d_model):
    """Search space of an arm at a model dimension : the ID baselines have their own, the
    SID arms share one."""
    return config.space[d_model]["id" if arm in ["id", "id_vanilla"] else "sid"]


def suggest(trial, config, arm, d_model):
    """Optuna trial -> overrides of src/train.py for its hyperparameters."""
    ranges = space(config, arm, d_model)
    lr = trial.suggest_float("lr", *ranges.lr, log=True)
    weight_decay = trial.suggest_float("weight_decay", *ranges.weight_decay, log=True)
    dropout = trial.suggest_float("dropout", *ranges.dropout)
    return [
        f"model.optimizer.lr={lr}",
        f"model.optimizer.weight_decay={weight_decay}",
        f"model.net.dropout={dropout}",
    ]


def arm_defaults(arm):
    """Hyperparameters of the experiment config of an arm : the first trial of its
    studies."""
    path = os.path.join(
        os.path.dirname(__file__), "configs", "experiment", f"{arm}.yaml"
    )
    model = OmegaConf.load(path).model
    return {
        "lr": model.optimizer.lr,
        "weight_decay": model.optimizer.weight_decay,
        "dropout": model.net.dropout,
    }


def last_validation(fs, folder, task_name, metric, since):
    """Last metric reported by the run of a task started after since, None before its
    validation. The runs of an earlier tuning, under the same task name, are ignored :
    run names end with their start date, as since."""
    runs = sorted(r for r in list_runs(fs, folder, f"{task_name}_") if r[-19:] >= since)
    try:
        _, _, metrics = last_checkpoint(fs, runs[-1])
    # Not reported yet, or a failed HDFS read (the file being rewritten) : next poll
    except (IndexError, OSError, json.JSONDecodeError):
        return None
    return metrics[metric]


@hydra.main(config_path="configs", config_name="tune", version_base="1.3")
def main(config):
    patch_fsspec()
    fs = fsspec.filesystem(config.paths.protocol)
    # Inside a job, RAY_ADDRESS is the head's GCS address : its dashboard is on 8265
    head = os.environ["RAY_ADDRESS"].split(":")[0]
    client = JobSubmissionClient(f"http://{head}:8265")

    studies = {}  # (category, d_model, arm) -> study, in the order they run
    for category in config.categories:
        for d_model in config.d_models:
            for arm in config.arms:
                studies[category, d_model, arm] = optuna.create_study(
                    study_name=f"{category}_d{d_model}_{arm}",
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(
                        seed=config.seed,
                        multivariate=True,  # lr and weight decay interact
                        constant_liar=True,  # parallel trials of a study spread out
                    ),
                )
    started = dict.fromkeys(studies, 0)
    numbers = dict.fromkeys(studies, 0)  # of the next task : names stay unique
    running = {}  # job id -> study key, trial, task, submission date
    for (category, d_model, arm), study in studies.items():
        study.enqueue_trial(arm_defaults(arm))
    while running or min(started.values()) < config.trials:
        for job_id, job in list(running.items()):
            try:
                status = client.get_job_status(job_id)
            except Exception as error:  # the dashboard can drop a request : next poll
                print(f"{job['task']} status unknown : {error}", flush=True)
                continue
            if not status.is_terminal():
                continue
            value = last_validation(
                fs,
                config.paths.model_folder_tplt,
                job["task"],
                f"valid/{job['key'][0]}/HR@10",
                job["since"],
            )
            if status == JobStatus.SUCCEEDED and value is None:
                continue  # its validation could not be read : next poll
            del running[job_id]
            study, trial = studies[job["key"]], job["trial"]
            if status == JobStatus.SUCCEEDED:
                study.tell(trial, value)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                # A node or HDFS failure must not cost the arm a trial : the same
                # point runs again, once
                if not trial.user_attrs.get("retry"):
                    study.enqueue_trial(trial.params, user_attrs={"retry": True})
                    started[job["key"]] -= 1
            print(f"{job['task']} {status} {value}", flush=True)

        # The first category with trials to start, then its study with the fewest
        while len(running) < config.max_parallel:
            waiting = [k for k in studies if started[k] < config.trials]
            if not waiting:
                break
            category = waiting[0][0]  # studies are in the order they run
            key = min(
                (k for k in waiting if k[0] == category), key=lambda k: started[k]
            )
            category, d_model, arm = key
            trial = studies[key].ask()
            task = f"tune_{category}_d{d_model}_{arm}_{numbers[key]:02d}"
            numbers[key] += 1
            overrides = [
                f"experiment={arm}",
                f"dataset.category={category}",
                f"model.net.d_model={d_model}",
                f"task_name={task}",
                *suggest(trial, config, arm, d_model),
            ]
            # Same format and clock as the run names of src/train.py
            since = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            job_id = submit("src/train.py", overrides, client)
            running[job_id] = {"key": key, "trial": trial, "task": task, "since": since}
            started[key] += 1
        time.sleep(60)

    for (category, d_model, arm), study in studies.items():
        print(
            f"{category} d{d_model} {arm} : best validation HR@10 "
            f"{study.best_value:.4f}, {study.best_params}"
        )


if __name__ == "__main__":
    main()
