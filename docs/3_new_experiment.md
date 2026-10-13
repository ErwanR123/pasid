# 3. Run your own experiments

Each recipe assumes the data of the category is ready ([2_reproduce.md](2_reproduce.md),
step 1). Give every experiment its own `task_name`: it starts the run names, and
evaluation finds the runs by it:

```sh
python submit.py src/test.py prefix=<task_name>
python submit.py src/collect_results.py prefix=<task_name> split=test
```

## Two ways to train

**`campaign.py`: with the tuned hyperparameters.** Choose the categories, dimensions,
models and seeds:

```sh
python submit.py campaign.py "categories=[Toys_and_Games]" "d_models=[128]" "arms=[sid,pasid]" "seeds=[0]" task_name=try
```

**`src/train.py`: one run, your own hyperparameters.** Every setting of
`configs/train.yaml` and of the model config can be changed:

```sh
python submit.py src/train.py experiment=pasid dataset.category=Toys_and_Games model.net.d_model=256 model.optimizer.lr=0.003 seed=1 task_name=try
```

Careful: `src/train.py` does not read `configs/tuned.yaml`. Without the overrides, a run
takes the default hyperparameters of its model config, not the tuned ones.

## The models

`experiment=<model>` for `src/train.py`, `arms=[...]` for `campaign.py` and `tune.py`:

| Model | Item representation | Config |
|---|---|---|
| `id` | one learned embedding per item | `configs/experiment/id.yaml` |
| `sid` | 4 content codes | `configs/experiment/sid.yaml` |
| `sid5` | 5 content codes: same length as PASID, without popularity | `configs/experiment/sid5.yaml` |
| `pasid` | 4 content codes + p | `configs/experiment/pasid.yaml` |
| `id_vanilla` | the original SASRec on ids: BCE with one negative, dot product, no crops | `configs/experiment/id_vanilla.yaml` |

## Another category

Any category of Amazon Reviews 2023, e.g. `Office_Products`: prepare its data (step 1
of [2_reproduce.md](2_reproduce.md)), then train with `dataset.category=...` and
`src/train.py`, or tune it first (below): `configs/tuned.yaml` only has the three
categories of the paper.

## Degraded content

How much does each model rely on the text of the items? Step 1 can cut the text of each
item to its first characters (`max_chars`) or empty it for a random fraction of the items
(`empty_fraction`). The embeddings get a new name, and the next steps must use the same:

```sh
C=dataset.category=Toys_and_Games
E=dataset.emb_method=sentence-t5-xl_chars10
python submit.py data_scripts/1_make_embeddings.py $C $E max_chars=10   # or empty_fraction=0.5
python submit.py data_scripts/2_train_rqkmeans.py $C $E
python submit.py data_scripts/3_add_pasid_token.py $C $E
python submit.py campaign.py "categories=[Toys_and_Games]" "d_models=[128]" "arms=[sid,pasid]" task_name=chars10 "overrides=['$E']"
```

## Other Semantic IDs

Step 2 takes the number of codes and their sizes, e.g. 3 codes of 1024 values:

```sh
python submit.py data_scripts/2_train_rqkmeans.py $C "codebook_sizes=[1024,1024,1024]"
python submit.py data_scripts/3_add_pasid_token.py $C source=rqkmeans_1024-1024-1024
python submit.py src/train.py experiment=pasid $C sid_candidate=rqkmeans_1024-1024-1024_pasid task_name=k1024
```

`sid_candidate` is the SID file a run reads. `campaign.py` takes several at once:
`"sid_candidates=[...]"`, each run with the hyperparameters of the model.

## Tuning

One Optuna study per category, dimension and model, on the learning rate, weight decay
and dropout. Search spaces and number of trials: `configs/tune.yaml`.

```sh
python submit.py tune.py "categories=[Office_Products]" "arms=[id,pasid]"
```

It runs on the cluster, 32 trials at a time, and prints the best values of each study at
the end of its log (`ray job logs <id>`). Copy them into `configs/tuned.yaml` for
`campaign.py`. A failed trial runs again once. If the job itself stops, tuning starts
again from scratch.

## Metrics

`src/test.py` computes Recall@k (= HR@k, one target) and NDCG@k for k = 1, 5, 10, 20, over
the whole catalogue, the items of the history excluded:

- overall;
- per place of the target in PASID (`band/`): in the head, in a group of 2-9, 10-49,
  50-199 or 200+ items sharing p, or never seen in training (`cold`). The bands come from
  the file `rqkmeans_256-256-256-256_pasid` of the run's embeddings, whatever the model.

Items with the same SID (twins) get the same score: the target is ranked at random among
them, so a twin ranked first is not a hit.
