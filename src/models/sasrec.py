import torch

from src.models import SpecialTokens


class SampledSoftmax(torch.nn.Module):
    def __init__(self, n_items, temperature=1.0):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.n_items = n_items
        self.temperature = temperature

    def __call__(self, embs, target, model):
        B, L, D = embs.shape
        # Flatten everything, and keep the positions with a real target : the left
        # padding has nothing to predict
        embs = embs.view(B * L, D)
        target = target.view(B * L)
        real = target != SpecialTokens.PAD.value
        embs, target = embs[real], target[real]

        # Get target scores
        target_scores = (embs * model.get_embs(target)).sum(
            dim=-1
        )  # (N,) real positions
        # Sample shared noise among the real items & get scores
        samples = torch.randint(
            low=len(SpecialTokens),
            high=model.vocab_size,
            size=(self.n_items,),
            device=embs.device,
        )
        noise_scores = embs @ model.get_embs(samples).T

        # Reject samples with the target's SID : same vector, indistinguishable from it
        groups = model.item_groups
        reject_samples = groups[target][:, None] == groups[samples][None, :]
        noise_scores = noise_scores.masked_fill(reject_samples, -1e6)

        logits = (
            torch.cat([target_scores[:, None], noise_scores], dim=1) / self.temperature
        )
        loss = self.cross_entropy(logits, torch.zeros_like(target))

        return loss, logits


class BinaryCrossEntropy(torch.nn.Module):
    """Loss of the original SASRec : each position scores its target against one item
    drawn uniformly from the catalogue, with a binary cross-entropy on both."""

    def __call__(self, embs, target, model):
        B, L, D = embs.shape
        embs = embs.view(B * L, D)
        target = target.view(B * L)
        real = target != SpecialTokens.PAD.value
        embs, target = embs[real], target[real]  # (N, D), (N,) real positions

        negatives = torch.randint(
            low=len(SpecialTokens),
            high=model.vocab_size,
            size=target.shape,
            device=embs.device,
        )  # (N,)
        target_scores = (embs * model.get_embs(target)).sum(dim=-1)  # (N,)
        noise_scores = (embs * model.get_embs(negatives)).sum(dim=-1)  # (N,)
        logits = torch.stack([target_scores, noise_scores], dim=1)  # (N, 2)
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return loss, logits


class L2Norm(torch.nn.Module):
    """L2 normalization as a module : the final norm of the Transformer encoder."""

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=-1, eps=1e-6)


class SASRec(torch.nn.Module):
    """Causal Transformer over the item embeddings of a timeline, L2-normalized : the
    score of an item is the cosine between its embedding and the output. Without
    normalize, as the original SASRec : a final LayerNorm and a dot product."""

    def __init__(
        self,
        n_layers,
        d_head,
        d_model,
        item_encoder,  # src.models.item_encoders : item ids (*) -> (*, D)
        dropout,
        seq_len,
        criterion,
        normalize=True,
    ):
        super().__init__()

        self.vocab_size = item_encoder.num_embeddings
        self.normalize = normalize
        L = seq_len

        self.position_embeddings = torch.nn.Embedding(L, d_model)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones((L, L), dtype=torch.bool), diagonal=1).view(L, L),
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=d_model // d_head,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=n_layers,
            enable_nested_tensor=False,
            # Final norm at the output of the Transformer
            norm=L2Norm() if normalize else torch.nn.LayerNorm(d_model),
        )

        self.edrop = torch.nn.Dropout(dropout)
        self.inp_norm = torch.nn.LayerNorm(d_model)

        self.criterion = criterion

        self.apply(self._init_weights)
        # Registered after the init above : the item encoder initializes its own weights
        self.item_encoder = item_encoder
        # Items sharing their SID share their vector : one group per distinct SID,
        # one group per item for the ID encoder
        codes = getattr(item_encoder, "offset_codes", None)
        groups = (
            torch.arange(self.vocab_size)
            if codes is None
            else torch.unique(codes, dim=0, return_inverse=True)[1]
        )
        self.register_buffer("item_groups", groups, persistent=False)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_param_groups(self):
        # No weight decay on the embedding tables (item ids, SID codes, positions) : the
        # rest, the MLP of the SID encoder included, is decayed
        no_decay = [
            p
            for m in self.modules()
            if isinstance(m, torch.nn.Embedding)
            for p in m.parameters()
        ]
        decay = [p for p in self.parameters() if all(p is not q for q in no_decay)]

        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def forward(self, x):
        # Each distinct item of the batch encoded once : most positions are padding.
        # Spread back with an embedding lookup, whose backward sums the gradients of
        # repeated items with a sort (indexing would add them one by one)
        items, position_item = torch.unique(x, return_inverse=True)
        x_emb = torch.nn.functional.embedding(position_item, self.item_encoder(items))
        x_emb += self.position_embeddings.weight[None, : x.shape[1], :]
        x_emb = self.edrop(self.inp_norm(x_emb))

        L = x_emb.shape[1]
        # (B, L, D), L2-normalized by the final norm of the encoder if normalize
        return self.encoder(
            x_emb,
            mask=self.causal_mask[:L, :L],
            src_key_padding_mask=(x == SpecialTokens.PAD.value),
        )

    def get_loss(self, batch):
        query, target = batch["query"], batch["target"]

        embs = self(query)  # B x L x D
        loss, logits = self.criterion(embs, target, model=self)

        return loss, logits

    def get_embs(self, idx=None):
        if idx is None:
            idx = torch.arange(
                self.vocab_size, device=self.position_embeddings.weight.device
            )
        embs = self.item_encoder(idx)
        if self.normalize:
            embs = torch.nn.functional.normalize(embs, dim=-1, eps=1e-6)
        return embs

    def scores(self, batch):
        """Batch -> (B, V) score of every item id as the next item, -inf for the items
        of the history : they are not recommended again."""
        assert self.training is False, "Not in evaluation mode (dropout)."

        query = batch["query"]
        embs = self(query)[:, -1, :]  # Get last embedding
        voc = self.get_embs()  # Get everything
        # In fp32 : bf16 rounds the scores of ~1e5 items into many ties
        with torch.autocast(device_type=embs.device.type, enabled=False):
            logits = embs.float() @ voc.float().T  # / temperature keeps the order

        logits[torch.arange(len(query), device=query.device)[:, None], query] = float(
            "-inf"
        )
        return logits
