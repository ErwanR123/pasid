import numpy as np
import ray

from src.models import SpecialTokens


class SASRecPrePro:
    """Validation or test timeline -> query and target : its last crop_length items, the
    last one being the target, left padded. The training examples are drawn on the GPU
    (src.data.datamodule.TimelineBatches)."""

    def __init__(self, *, items_ref, crop_length):
        self.crop_length = crop_length
        self.item_to_id = ray.get(items_ref)["item_to_id"]

    def __call__(self, row):
        tl = [self.item_to_id[item] for item in row["timeline"][-self.crop_length :]]
        padding = [SpecialTokens.PAD.value] * (self.crop_length - len(tl) + 1)
        return {
            "query": np.array(padding + tl[:-1]),
            "target": np.array(padding + tl[1:]),
        }
