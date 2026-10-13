# 4. Add a model to compare

A model here is one way to turn an item into a vector: the rest (SASRec++, training,
evaluation) stays the same for all, so the comparison isolates the item representation.
There are two cases.

## Case 1: new codes, same encoder

The item is still represented by codes, and only the codes change, e.g. a variant of p,
or SIDs from another tokenizer. Nothing to change in `src/`:

1. Write the codes in a parquet next to the other SIDs:
   `$PASID_ROOT/data/embeddings/<emb_method>/<category>/<name>.parquet`, with one row per
   item: `product_id`, then one integer column per code, `L0`, `L1`, ... The easiest way
   is a copy of `data_scripts/3_add_pasid_token.py` that changes how the last column is
   built.
2. Create `configs/experiment/<model>.yaml`, a copy of `configs/experiment/pasid.yaml`
   with `sid_candidate: <name>`.
3. Run it as the others: `experiment=<model>` or `arms=[<model>]`.

The encoder embeds each code (one table for all the codes of all the levels), then an
MLP mixes the embeddings of an item into its vector (`SIDMLPItemEncoder` in
`src/models/item_encoders.py`).

## Case 2: a new encoder

The item vector is built another way, e.g. ID + SID.

1. In `src/models/item_encoders.py`, write a function that builds the encoder from the
   data, as `id_encoder` and `sid_encoder` do. The encoder is a `torch.nn.Module` that
   takes item ids `(*)` and returns vectors `(*, d_model)`, with:
   - `num_embeddings`: the number of item ids, padding included;
   - id 0 (padding) mapped to a zero vector;
   - optionally `offset_codes` `(V, Q)`: items with the same codes then count as twins
     (the same vector), in the loss and in the evaluation.
2. Create `configs/item_encoder/<encoder>.yaml`, as `configs/item_encoder/sid.yaml`:
   `_target_` points to the function, the other keys are its arguments.
3. Create `configs/experiment/<model>.yaml` with `override /item_encoder: <encoder>`, as
   `configs/experiment/sid.yaml`.
4. Add a test in `tests/test_item_encoders.py`: the padding gives a zero vector, the
   output has the right shape.

## Then, in both cases

- **Tuning**: `tune.py "arms=[<model>]"`. It takes the search space of the SID models,
  unless the name is `id` or `id_vanilla` (`space()` in `tune.py`).
- **Final runs**: add its best values to `configs/tuned.yaml`, under each category and
  dimension, then `campaign.py "arms=[<model>]"`.
- **Table**: `src/collect_results.py` groups the runs by the `sid_candidate` they read,
  or `id`. A new encoder on the same codes as an existing model needs its own name there
  (`run_row` in `src/collect_results.py`).
