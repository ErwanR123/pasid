"""Final runs : every category, model dimension, arm and seed of configs/campaign.yaml,
each with the hyperparameters tuned on its category and dimension (configs/tuned.yaml),
max_parallel at a time.

Runs on the cluster, so the queue does not depend on the laptop or the VPN :
    python submit.py campaign.py
    python submit.py campaign.py "d_models=[128]" seeds=[0,1,2]
    python submit.py campaign.py arms=[pasid] "sid_candidates=[<other PASID codes>]"
Every run is submitted, even if an earlier campaign already ran it. Then evaluate on
test with
    python submit.py src/test.py prefix=<task_name>
"""

import os
import time

import hydra
from omegaconf import OmegaConf
from ray.job_submission import JobSubmissionClient

from submit import submit


@hydra.main(config_path="configs", config_name="campaign", version_base="1.3")
def main(config):
    # Inside a job, RAY_ADDRESS is the head's GCS address : its dashboard is on 8265
    head = os.environ["RAY_ADDRESS"].split(":")[0]
    client = JobSubmissionClient(f"http://{head}:8265")
    tuned = OmegaConf.load(
        os.path.join(os.path.dirname(__file__), "configs", "tuned.yaml")
    )

    running = set()
    # Seed by seed : the first seeds of every arm show early if a config is unstable
    queue = [
        (category, d_model, arm, candidate, seed)
        for seed in config.seeds
        for category in config.categories
        for d_model in config.d_models
        for arm in config.arms
        # The candidates of the config, else the one of the arm's experiment config
        for candidate in config.sid_candidates
        or [tuned[category][f"d{d_model}"][arm].get("sid_candidate")]
    ]
    print(f"{len(queue)} runs to submit", flush=True)

    while queue or running:
        for job_id in list(running):
            try:
                status = client.get_job_status(job_id)
            except Exception as error:  # the dashboard can drop a request : next poll
                print(f"{job_id} status unknown : {error}", flush=True)
                continue
            if status.is_terminal():
                running.discard(job_id)
                print(f"{job_id} {status}", flush=True)
        while queue and len(running) < config.max_parallel:
            category, d_model, arm, candidate, seed = queue.pop(0)
            hparams = tuned[category][f"d{d_model}"][arm]
            overrides = [
                f"experiment={arm}",
                f"dataset.category={category}",
                f"model.net.d_model={d_model}",
                f"task_name={config.task_name}_{category}",
                f"seed={seed}",
                f"model.optimizer.lr={hparams.lr}",
                f"model.optimizer.weight_decay={hparams.weight_decay}",
                f"model.net.dropout={hparams.dropout}",
                *([f"sid_candidate={candidate}"] if candidate else []),
                *config.overrides,
            ]
            running.add(submit("src/train.py", overrides, client))
        time.sleep(60)


if __name__ == "__main__":
    main()
