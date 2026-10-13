# PASID: Popularity-Aware Semantic IDs

Sequential recommendation where each item is represented by its **Semantic ID** (4 content
codes from RQ-KMeans on its text embedding) plus one code **p** built from its training
popularity:

1. rank the items by popularity n (number of training events);
2. head: the 1 % most popular items get one value of p each. tau = popularity of the
   last head item;
3. groups: below the head, the items are taken in order and put into groups. A group is
   closed as soon as its items add up to tau events. All the items of a group share one
   value of p, and rarer items make larger groups;
4. twins: two items with the same content codes never share p. Items never seen in
   training join the last group.

The model is SASRec++. Only its item representation changes between the compared models.

## Guides

| | |
|---|---|
| [1. Setup](docs/1_setup.md) | install, cluster, first command |
| [2. Reproduce the paper](docs/2_reproduce.md) | data, training, evaluation, step by step |
| [3. Run your own experiments](docs/3_new_experiment.md) | other categories, hyperparameters, degraded content, tuning, metrics |
| [4. Add a model](docs/4_add_a_model.md) | new codes, or a new encoder |
| [5. Code map](docs/5_code_map.md) | what each file does, where outputs go |
| [Design notes](docs/design.md) | choices of the implementation |
| [Results](docs/results.md) | all the numbers, known issues, next steps |

## In short

```sh
pip install -e ".[dev]"
export RAY_ADDRESS=... PASID_ROOT=... WANDB_API_KEY=...         # docs/1_setup.md

python submit.py data_scripts/0_raw_to_parquet.py dataset.category=Toys_and_Games   # + steps 1-3
python submit.py campaign.py                                     # train every model
python submit.py src/test.py prefix=final                        # evaluate on test
python submit.py src/collect_results.py prefix=final split=test  # table
```

## Origin

This repository is a fork of [COSETTE & MARIUS](https://github.com/Simon-Lepage/cosette_and_marius)
(Lepage, Mary, Picard, Criteo AI Lab & ENPC). The Amazon Reviews 2023 pipeline, SASRec++,
training and evaluation come from it. See [NOTICE](NOTICE) for the changes, and please
cite their work:

```bibtex
@article{lepage2025closing,
  title={Closing the Performance Gap in Generative Recommenders with Collaborative Tokenization and Efficient Modeling},
  author={Lepage, Simon and Mary, J{\'e}r{\'e}mie and Picard, David},
  journal={arXiv:2508.14910},
  year={2025}
}
```
