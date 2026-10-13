# Results

Amazon Reviews 2023, full catalogue (156k-207k items). Recall@10 = HR@10 (one target).

## Main table

Test Recall@10 (%), mean over 3 seeds (std <= 0.05).

| | Beauty 128 | 256 | 512 | Sports 128 | 256 | 512 | Toys 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|
| `id_vanilla` | 1.93 | 2.08 | 2.27 | 1.82 | 2.01 | 2.27 | 2.28 | 2.90 | 3.21 |
| `id` | 3.72 | 3.78 | 3.70 | 3.21 | 3.32 | 3.32 | 5.01 | 5.04 | 4.96 |
| `sid` | 3.55 | 3.91 | 3.98 | 3.40 | 3.61 | 3.67 | 4.53 | 5.04 | 5.04 |
| `sid5` | 3.76 | 4.07 | 4.12 | 3.54 | 3.71 | 3.78 | 4.77 | 5.23 | 5.08 |
| `pasid` | 4.15 | 4.17 | 4.22 | 3.80 | 3.80 | 3.82 | 5.17 | 5.25 | 5.29 |

- `id_vanilla`: Sports d128 over 2 seeds, Toys d512 over 1.
- COSETTE & MARIUS report on the same data (their Table 11): SASRec++ 3.84 / 3.48 and
  MARIUS 4.04 / 3.72 on Beauty / Sports. Their Table 5 is on Amazon 2014 (about 12k
  items): not comparable.

## Where PASID helps

Test, d = 128. Test cases split by the place of their target in PASID (`band/`).
Cases: share of the test cases (%); `sid`, `pasid`: Recall@10 (%) on these cases.

| Target item | Beauty cases | `sid` | `pasid` | Sports cases | `sid` | `pasid` | Toys cases | `sid` | `pasid` |
|---|---|---|---|---|---|---|---|---|---|
| in the head | 23.4 | 11.22 | 11.74 | 16.3 | 13.73 | 13.55 | 14.9 | 12.71 | 12.74 |
| in a group of 2-9 | 32.3 | 2.47 | 3.42 | 33.0 | 2.99 | 3.94 | 32.3 | 5.75 | 6.64 |
| in a group of 10-49 | 24.7 | 0.48 | 1.11 | 29.6 | 0.51 | 0.89 | 30.2 | 2.15 | 3.24 |
| in a group of 50-199 | 13.7 | 0.09 | 0.21 | 17.1 | 0.10 | 0.15 | 18.8 | 0.66 | 0.77 |
| in a group of 200+ | 5.8 | 0.03 | 0.03 | 3.6 | 0.04 | 0.02 | 3.5 | 0.23 | 0.13 |
| never seen in training | 0.1 | 0.00 | 0.00 | 0.3 | 0.00 | 0.00 | 0.3 | 0.14 | 0.00 |

The gain is not on the head but in the groups just below it.

## Degraded content

The text of the items is cut or emptied, then embeddings, SIDs and PASID are rebuilt
(hyperparameters not tuned again). Validation Recall@10 (%), d = 128, 3 seeds,
`sid` / `pasid`.

| Text of the items | Beauty | Sports | Toys |
|---|---|---|---|
| full | 4.04 / 4.75 | 4.04 / 4.59 | 5.57 / 6.53 |
| cut to 20 characters | 2.38 / 4.67 | 2.41 / 4.18 | 2.75 / 6.25 |
| cut to 10 characters | 0.56 / 4.57 | 0.64 / 4.08 | 0.59 / 5.99 |
| empty for 50 % of items | 0.95 / 3.92 | 0.79 / 3.57 | 1.15 / 5.59 |

## Read before reusing these numbers

- **Version of p.** All the `pasid` numbers above, and its tuning, used an earlier
  version of p: a group closed at `tau * (tau / n) ** 0.3` events instead of `tau`
  (larger groups for the rarer items). On validation, the two versions are within 0.07
  point.
- **Ties.** These runs were evaluated before the last fix of the ranking: other items
  with exactly the target's score now tie with it. Before, a collapsed model counted
  every target as a hit. These runs did not collapse (std <= 0.05).
- **Code simplified after the runs.** `tune.py` and `campaign.py` passed a dry run with a
  fake Ray client, `src/test.py` the unit tests, but none ran on the cluster since.

## Known issues and next steps

- A vanilla SASRec on SIDs / PASID is not in the repo: its tuning picked collapsed trials
  before the fix of the ties. To redo with `normalize: false` and `BinaryCrossEntropy`,
  as `configs/experiment/id_vanilla.yaml`.
- With 90 % of the texts empty, PASID on Toys fell to 0.49: not investigated.
- Not run: SID + a learned popularity bias, and SID + the head identities only, to split
  the gain of PASID between its two parts.
- Not run: a comparison with SIDs learned with collaborative signal (LETTER, COSETTE).
