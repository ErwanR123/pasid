"""Item encoders of SASRec, built from the data : item ids (*) -> embeddings (*, D).

V = vocabulary size (special tokens + items), Q SID levels, D = d_model. Both encoders expose num_embeddings = V and map padding to a zero vector.
They initialize their own weights : SASRec does not re-initialize them.
"""

import fsspec
import torch

from src.data.ray_data import get_items_map, get_quantized
from src.models import SpecialTokens
from src.utils.tools import patch_fsspec


def init_embedding_table(weight):
    """In place N(0, 0.02) init of a (rows, D) table, row 0 (padding) kept at zero."""
    torch.nn.init.normal_(weight, std=0.02)
    with torch.no_grad():
        weight[0].zero_()


def build_mlp(input_dim, hidden_dims, output_dim):
    """Linear + GELU per hidden dim, then a Linear to output_dim.
    No biases : a zero input (padding) gives a zero output."""
    layers = []
    for hidden_dim in hidden_dims:
        layers += [torch.nn.Linear(input_dim, hidden_dim, bias=False), torch.nn.GELU()]
        input_dim = hidden_dim
    return torch.nn.Sequential(
        *layers, torch.nn.Linear(input_dim, output_dim, bias=False)
    )


class SIDMLPItemEncoder(torch.nn.Module):
    """Concatenate the Q SID code embeddings and reduce them to D with an MLP,
    which can learn interactions between levels."""

    def __init__(self, item_codes, d_model, mlp_hidden_dims):
        """item_codes (V, Q) : codes 1..K_q at level q, 0 = padding (special tokens).
        MLP Q * D -> mlp_hidden_dims -> D."""
        super().__init__()
        self.num_embeddings, num_levels = item_codes.shape
        # Single table for all levels : level q starts after the K_1 + ... + K_{q-1} codes
        # of the previous levels (K_q differ : the PASID code has far more values),
        # padding code 0 stays at row 0.
        level_sizes = item_codes.max(dim=0).values  # (Q,) K_q
        level_offsets = torch.cumsum(level_sizes, dim=0) - level_sizes  # (Q,)
        offset_codes = torch.where(
            item_codes == 0, 0, item_codes + level_offsets
        )  # (V, Q)
        self.register_buffer("offset_codes", offset_codes, persistent=False)

        self.code_embeddings = torch.nn.Embedding(
            int(level_sizes.sum()) + 1, d_model, padding_idx=0
        )
        init_embedding_table(self.code_embeddings.weight)
        self.mlp = build_mlp(num_levels * d_model, mlp_hidden_dims, d_model)

    def forward(self, item_ids):
        """item_ids (*) -> embeddings (*, D), zero for padding."""
        code_embeddings = self.code_embeddings(self.offset_codes[item_ids])  # (*, Q, D)
        return self.mlp(code_embeddings.flatten(start_dim=-2))  # (*, Q * D) -> (*, D)


def _items_map(paths, category):
    patch_fsspec()  # Built on the Ray workers, which need the viewfs protocol too
    fs = fsspec.filesystem(paths.protocol)
    return fs, get_items_map(fs, paths.unique_items_tplt.format(category=category))


def id_encoder(paths, category, d_model):
    """One learned embedding per item id : the SASRec baseline."""
    _, items_map = _items_map(paths, category)
    vocab_size = len(items_map["item_to_id"]) + len(SpecialTokens)
    encoder = torch.nn.Embedding(
        vocab_size, d_model, padding_idx=SpecialTokens.PAD.value
    )
    init_embedding_table(encoder.weight)
    return encoder


def sid_encoder(paths, category, emb_method, quant_method, d_model, mlp_hidden_widths):
    """SIDMLPItemEncoder on the codes of the quant_method candidate, its hidden dims
    mlp_hidden_widths times d_model : the MLP keeps its shape at every d_model."""
    fs, items_map = _items_map(paths, category)
    quantized = get_quantized(
        fs,
        paths.semantic_ids_tplt.format(
            emb_method=emb_method, category=category, quant_method=quant_method
        ),
    )  # (N, Q) indexed by product_id

    # item_to_id numbers the items in its own order : read the codes in that order,
    # after the all-padding rows of the special tokens
    codes = quantized.loc[list(items_map["item_to_id"])].to_numpy()  # (N, Q)
    codes = torch.from_numpy(codes).long() + 1  # 0 = padding
    padding = torch.zeros(len(SpecialTokens), codes.shape[1], dtype=torch.long)
    item_codes = torch.cat([padding, codes])  # (V, Q)
    mlp_hidden_dims = [width * d_model for width in mlp_hidden_widths]
    return SIDMLPItemEncoder(item_codes, d_model, mlp_hidden_dims)
