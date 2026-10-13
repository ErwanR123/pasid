# 1. Setup

From zero to a first job on the cluster.

## Install

Python 3.11, from the repo root:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Nothing trains on the laptop: it only submits jobs. A CPU torch is enough.

## Environment variables

Three variables, read from your shell. `submit.py` forwards them to the jobs, so they
are never written in the repo:

```sh
export RAY_ADDRESS=http://<your Ray cluster dashboard>:8265   # `ray-cli clusters` gives it
export PASID_ROOT=/user/<you>/ray/pasid   # where data and models go, on HDFS
export WANDB_API_KEY=<key>                # https://wandb.ai/authorize
```

On the Criteo clusters, the jobs can only write under `/user/<you>/ray`, hence
`PASID_ROOT`. Training curves go to the wandb project `pasid`.

## How a script runs

Every script runs as a Ray job, submitted from the repo root:

```sh
python submit.py <script> [overrides...]
```

`submit.py` uploads the repo folder, installs the few packages missing from the cluster
image, and starts `python <script> [overrides...]` on the cluster. It prints the job id.

Each script reads its settings from a yaml in `configs/` (Hydra). Any setting can be
changed on the command line, e.g. `dataset.category=Sports_and_Outdoors` or
`model.net.d_model=256`. The settings shared by all scripts, the category and the paths,
are in `configs/common.yaml`.

## Follow a job

```sh
ray job status <id>
ray job logs -f <id>     # its output, live
ray job stop <id>
```

The Ray dashboard (`RAY_ADDRESS` in a browser) shows the same, and the GPUs in use.

## First command

Download the Toys data (CPU only):

```sh
python submit.py data_scripts/0_raw_to_parquet.py dataset.category=Toys_and_Games
```

When it succeeds, the timelines are under `$PASID_ROOT/data/timelines/`. Next:
[2_reproduce.md](2_reproduce.md).

## Without a Ray cluster

On one GPU machine, a script can run directly: Ray then starts on the machine, and the
data and models go to the local disk. The queues (`tune.py`, `campaign.py`) need a cluster.

```sh
export PYTHONPATH=. RAY_TRAIN_V2_ENABLED=1 PASID_ROOT=$PWD/data WANDB_MODE=offline
python data_scripts/0_raw_to_parquet.py
python src/train.py experiment=pasid
```

## Tests

```sh
pytest                         # a few seconds, on CPU
ruff check . && ruff format .
```
