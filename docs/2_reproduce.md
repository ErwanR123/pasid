# 2. Reproduce the paper

Three categories of Amazon Reviews 2023 (`Toys_and_Games`, `Sports_and_Outdoors`,
`Beauty_and_Personal_Care`), five models, three model dimensions (128, 256, 512),
three seeds. Results to expect: [results.md](results.md).

## Step 1: data, for each category

Each command is one job. Wait until a job has succeeded (`ray job status <id>`) before
submitting the next one: each step reads the output of the previous one.

```sh
C=dataset.category=Toys_and_Games
python submit.py data_scripts/0_raw_to_parquet.py $C
python submit.py data_scripts/1_make_embeddings.py $C
python submit.py data_scripts/2_train_rqkmeans.py $C
python submit.py data_scripts/2_train_rqkmeans.py $C "codebook_sizes=[256,256,256,256,256]"
python submit.py data_scripts/3_add_pasid_token.py $C
```

| Step | What it does | Output, under `$PASID_ROOT/data/` |
|---|---|---|
| 0 | downloads the timelines (official 5-core, last-out split) and the item metadata | `timelines/`, `meta/` |
| 1 | embeds the text of each item with Sentence-T5-XL (1 GPU) | `embeddings/sentence-t5-xl/<category>/embeddings.parquet` |
| 2 | Semantic IDs with RQ-KMeans, 4 codes of 256 values. Prints the share of items sharing their SID (twins) | `.../rqkmeans_256-256-256-256.parquet` |
| 2 bis | the same with 5 codes, for the `sid5` model | `.../rqkmeans_256-256-256-256-256.parquet` |
| 3 | adds the popularity code p to the 4 codes | `.../rqkmeans_256-256-256-256_pasid.parquet` |

## Step 2: training

```sh
python submit.py campaign.py
```

One job that submits all the runs: 3 categories x 3 dimensions x 4 models (`id`,
`sid`, `sid5`, `pasid`) x 3 seeds = 108 runs, 32 at a time, each with its
hyperparameters from `configs/tuned.yaml`. Add the vanilla SASRec with:

```sh
python submit.py campaign.py "arms=[id_vanilla]" task_name=vanilla
```

A run trains for a fixed budget (50 epochs over the users) and is validated once, at the
end. Its checkpoint goes to `$PASID_ROOT/models/<run name>`, its curves to wandb. Run
names: `<task_name>_<category>_<SIDs or id>_seed<seed>_<date>`.

If `campaign.py` stops, run it again: it submits every run again.

## Step 3: evaluation

```sh
python submit.py src/test.py prefix=final
python submit.py src/collect_results.py prefix=final split=test
```

`src/test.py` evaluates on test every run whose name starts with `prefix`.
`src/collect_results.py` then prints one line per run, and the table: mean ± std over
the seeds, per category, dimension and model, in %.

Without `split=test`, `collect_results.py` prints the validation metrics logged at the
end of training: no need to run `src/test.py` first.
