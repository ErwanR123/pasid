# Design notes

Choices of the implementation, for whoever extends it. The README covers how to run it.

- **SASRec++, not vanilla SASRec.** Sampled softmax over 30k shared negatives, cosine
  scores (L2-normalized), pre-norm, random crops of the timelines. Only the item embedding
  changes between arms (`src/models/item_encoders.py`), so their comparison isolates it.
- **Training batches on the GPU.** The train timelines are read once and stay on the GPU,
  where each batch draws its users, crops and shuffles (`src/data/datamodule.py`), as C&M's
  Ray Data pipeline did on CPU one example at a time : 60k timelines/s instead of 21k.
- **The loss ignores the padding.** Timelines are left-padded to 50 items, and most are
  short : about 89 % of the positions of a batch are padding. As in the original SASRec,
  only the positions with a real target enter the loss, and negatives are real items.
  The code of C&M computed the loss on every position.
- **SID encoder.** The embeddings of the codes are concatenated and reduced by an MLP
  without biases : unlike a sum, it keeps track of the levels and can mix them, and a
  padding item stays a zero vector. The same encoder embeds the history and the candidates.
- **Twins.** Items with the same SID get the same vector. They are never sampled as
  negatives of each other. At evaluation the target is the item itself : it is ranked
  uniformly at random among its twins, and HR and NDCG are their expectation over this
  draw. A twin ranked first is not a hit.
- **Ties.** Any other item with exactly the target's score ties with it, at random.
- **Popularity** is the number of occurrences of the item in the train timelines only.
- **Evaluation.** Full ranking over the catalogue, items of the history excluded, scores
  in fp32 (bf16 would tie many items).
- **Two runs per GPU.** A step leaves the GPU idle most of the time : two runs share each
  GPU (+33 % throughput), and their CPUs (driver, trainer, valid preprocessing) fit a GPU
  worker.
