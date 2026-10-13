"""Submit one script of this repo as a Ray job on the cluster.

Usage, from this folder :
    RAY_ADDRESS=http://<dashboard>:8265 python submit.py <script> [hydra overrides...]
    e.g. python submit.py data_scripts/2_train_rqkmeans.py codebook_sizes=[256,256,256]

The whole folder is uploaded as the job working_dir. Scripts start their own
Ray tasks (CPU / GPU), so the entrypoint itself only takes one CPU on a worker.
"""

import os
import shlex
import sys

from ray.job_submission import JobSubmissionClient

# Packages missing from the cluster image, per script (versions of pyproject.toml)
PIP = {
    "data_scripts/0_raw_to_parquet.py": ["huggingface_hub==0.36.2"],
    "data_scripts/1_make_embeddings.py": [
        "sentence-transformers==2.7.0",
        "transformers==4.42.4",
    ],
    "src/train.py": ["wandb==0.19.7"],
    "tune.py": ["optuna==5.0.0"],
}


def submit(script, overrides, client=None):
    """Submit `python <script> <overrides>` as a Ray job, return its id.
    client : a JobSubmissionClient, by default on the RAY_ADDRESS dashboard."""
    runtime_env = {
        "working_dir": ".",
        # Ray refuses uploads above 100 MiB : the local venv alone is bigger
        "excludes": [".venv", "__pycache__", "/data"],
        "env_vars": {
            "PYTHONPATH": ".",  # `from src...` imports
            "PYTHONUNBUFFERED": "1",  # prints reach the job log live
            "RAY_TRAIN_V2_ENABLED": "1",
        },
    }
    # Read from the local environment : never written in the repo
    for name in ["PASID_ROOT", "WANDB_API_KEY"]:
        if name in os.environ:
            runtime_env["env_vars"][name] = os.environ[name]
    if script in PIP:
        runtime_env["pip"] = PIP[script]

    # The cluster pods read and write on HDFS, the default protocol is the local disk
    if not any(o.startswith("paths.protocol=") for o in overrides):
        overrides = ["paths.protocol=viewfs", *overrides]
    entrypoint = shlex.join(["python", script, *overrides])
    client = client or JobSubmissionClient()  # address read from RAY_ADDRESS
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        # The head node has no CPU to lend : one CPU sends the entrypoint to a worker,
        # instead of piling every job's driver on the small head (out of memory at 7 jobs)
        entrypoint_num_cpus=1,
    )
    print(f"Submitted {job_id} : {entrypoint}")
    print(f"Logs : ray job logs -f {job_id}")
    return job_id


if __name__ == "__main__":
    submit(sys.argv[1], sys.argv[2:])
