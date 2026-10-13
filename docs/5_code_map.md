# 5. Code map

## The path of a run

```
data_scripts/0_raw_to_parquet.py    Amazon 2023 -> timelines (train / valid / test) + item metadata
data_scripts/1_make_embeddings.py   text of each item -> Sentence-T5-XL embedding
data_scripts/2_train_rqkmeans.py    embeddings -> Semantic IDs (RQ-KMeans)
data_scripts/3_add_pasid_token.py   Semantic IDs + popularity -> PASID (code p appended)
        |
src/train.py                        config -> one training run, on a Ray GPU worker
  src/data/ray_data.py              timelines -> item ids ; valid / test examples
  src/data/datamodule.py            training batches, drawn on the GPU
  src/models/item_encoders.py       item id -> vector (ID table, or codes + MLP)
  src/models/sasrec.py              SASRec++ : Transformer over the history, loss, scores
  src/lightning_module.py           training step, metrics (Recall, NDCG, per band)
  src/utils/callbacks.py            checkpoint at the end of training, to $PASID_ROOT/models
        |
src/test.py                         checkpoints -> test metrics, next to each checkpoint
src/collect_results.py              metrics of all the runs -> table, mean ± std over seeds
```

## Around it

| File | Role |
|---|---|
| `submit.py` | sends a script to the cluster as a Ray job |
| `campaign.py` | queue of training runs with the tuned hyperparameters (a job itself) |
| `tune.py` | queue of Optuna trials (a job itself) |
| `src/data/sasrec.py` | valid / test example: the last 50 items of a timeline, the last one is the target |
| `src/utils/scheduler.py` | learning rate: linear warmup, then cosine decay |
| `src/utils/tools.py` | registers HDFS (`viewfs://`) in fsspec |
| `src/models/__init__.py` | id of the padding item (0) |

## Configs

| File | Role |
|---|---|
| `configs/common.yaml` | category, embedding name, paths: shared by every script |
| `configs/train.yaml` | training: budget, trainer, optimizer, data, Ray resources |
| `configs/experiment/*.yaml` | one per model: its encoder, its SIDs, default hyperparameters |
| `configs/experiment/sasrec.yaml` | SASRec++ itself: layers, dimension, loss. Every model but `id_vanilla` builds on it |
| `configs/item_encoder/*.yaml` | the two encoders: `id`, `sid` |
| `configs/tuned.yaml` | best hyperparameters per category, dimension and model: read by `campaign.py` |
| `configs/tune.yaml`, `configs/campaign.yaml` | settings of the two queues |
| `configs/data/*.yaml` | settings of the data steps |
| `configs/test.yaml`, `configs/collect_results.yaml` | `prefix` of the runs to evaluate |

A config is a stack: `train.yaml` takes `common.yaml`, an item encoder and an
experiment, which itself takes `sasrec.yaml`. To see the final config of a run, look at
the start of its log, or at `config.yaml` next to its checkpoint.

## Where things are written

Everything goes under `$PASID_ROOT` (`paths` in `configs/common.yaml`):

```
data/timelines/<category>.{train,valid,test}.parquet, <category>_items.pkl
data/meta/<category>.parquet
data/embeddings/<emb_method>/<category>/embeddings.parquet
data/embeddings/<emb_method>/<category>/<SIDs>.parquet
models/<run name>/                 Ray checkpoint: model, config.yaml, metrics.pkl after src/test.py
```

## Tests

| File | What it checks |
|---|---|
| `tests/test_pasid_token.py` | the code p: unique SIDs, head, groups closed at tau, cold items |
| `tests/test_item_encoders.py` | the encoders: padding, levels, shapes |
| `tests/test_timeline_batches.py` | the training batches: windows, crops, shuffled ties |
| `tests/test_twins.py` | twins: negatives, ranks at evaluation, ties, padding out of the loss |
| `tests/test_stratified_metrics.py` | the PASID bands and their metrics |
