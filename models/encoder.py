"""
Token encoder based on pre-trained transformers (e.g., BERT or XLNet)
Similar to [self-attentive-parser](https://github.com/nikitakit/self-attentive-parser)
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from omegaconf import DictConfig
from .utils import (
    BatchIndices,
    FeatureDropout,
    MultiHeadAttention,
    PartitionedPositionwiseFeedForward,
)
from typing import Dict, Any


class OneHotEmbedding(nn.Module):
    "One-hot embeddings for tokens or POS tags"

    def __init__(self, num_words: int, d_emb: int) -> None:
        super().__init__()  # type: ignore
        self.embedding = nn.Embedding(num_words, d_emb)

    def forward(
        self, tokens_idx: torch.Tensor, valid_tokens_mask: torch.Tensor
    ) -> torch.Tensor:
        words_emb: torch.Tensor = self.embedding(tokens_idx[valid_tokens_mask])
        return words_emb


class TransformerEmbedding(nn.Module):
    "Transformers followed by a linear projection layer"

    def __init__(self, encoder: str, d_emb: int) -> None:
        super().__init__()  # type: ignore
        self.contextual_embedding = AutoModel.from_pretrained(encoder)
        self.linear = nn.Linear(1024 if "-large" in encoder else 768, d_emb, bias=False)

    def forward(
        self,
        tokens_idx: torch.Tensor,
        valid_tokens_mask: torch.Tensor,
        word_end_mask: torch.Tensor,
    ) -> torch.Tensor:
        bert_output = self.contextual_embedding(
            tokens_idx, attention_mask=valid_tokens_mask.to(dtype=torch.float32)
        )[0]
        words_emb: torch.Tensor = self.linear(bert_output[word_end_mask])
        return words_emb


class Encoder(nn.Module):
    use_tags: bool
    use_words: bool
    # word_embedding
    # word_dropout
    # ..

    def __init__(
        self, position_table: torch.Tensor, vocabs: Dict[str, Any], cfg: DictConfig
    ) -> None:
        super().__init__()  # type: ignore

        # whether to use POS tags
        self.use_tags = cfg.use_tags
        # whether to use one-hot encodings of words
        self.use_words = cfg.use_words
        d_model = cfg.d_model
        d_content = d_model // 2
        d_positional = d_model - d_content

        self.word_embedding = TransformerEmbedding(cfg.encoder, d_content)
        self.word_dropout = FeatureDropout(cfg.word_emb_dropout)

        if self.use_tags:
            self.tag_embedding = OneHotEmbedding(len(vocabs["tag"]), d_content)
            self.tag_dropout = FeatureDropout(cfg.tag_emb_dropout)
        if self.use_words:
            self.word_onehot_embedding = OneHotEmbedding(
                len(vocabs["token"]), d_content
            )

        # share the position embedding matrix with the GCNs
        self.position_table = position_table

        self.layer_norm = nn.LayerNorm([d_model])

        # self-attention layers
        self.attn_layers = []
        for i in range(cfg.num_attn_layers):
            attn_sublayer = MultiHeadAttention(
                cfg.num_attn_heads,
                d_model,
                cfg.d_kqv,
                cfg.d_kqv,
                cfg.residual_dropout,
                cfg.attention_dropout,
                d_positional,
            )
            feedforward_sublayer = PartitionedPositionwiseFeedForward(
                d_model, cfg.d_ff, d_positional, cfg.relu_dropout, cfg.residual_dropout
            )
            self.add_module("attn_%d" % i, attn_sublayer)
            self.add_module("feedforward_%d" % i, feedforward_sublayer)
            self.attn_layers.append((attn_sublayer, feedforward_sublayer))

    def forward(
        self,
        tokens_idx: torch.Tensor,
        tags_idx: torch.Tensor,
        valid_tokens_mask: torch.Tensor,
        word_end_mask: torch.Tensor,
    ) -> torch.Tensor:
        # transformer features
        words_emb = self.word_embedding(tokens_idx, valid_tokens_mask, word_end_mask)

        # features from one-hot encodings
        if self.use_words:
            words_emb += self.word_onehot_embedding(tokens_idx, valid_tokens_mask)

        lens = word_end_mask.detach().sum(dim=-1).tolist()
        max_len = max(lens)
        batch_idxs = BatchIndices.from_lens(lens)
        # num_tokens x d_content
        words_emb = self.word_dropout(words_emb, batch_idxs)

        if self.use_tags:
            valid_tags_mask = valid_tokens_mask.new_zeros(
                (len(lens), max_len), dtype=torch.bool
            )
            for i, l in enumerate(lens):
                valid_tags_mask[i, :l] = True
            tags_idx = tags_idx[:, :max_len]
            tags_emb = self.tag_dropout(
                self.tag_embedding(tags_idx, valid_tags_mask), batch_idxs
            )
            words_emb += tags_emb

        # concat with position encodings
        timing_signal = torch.cat(
            [self.position_table[:l] for l in lens], dim=0
        )  # num_tokens x d_positional
        # num_tokens x d_model
        tokens_emb = torch.cat([words_emb, timing_signal], dim=1)
        tokens_emb = self.layer_norm(tokens_emb)

        for attn, feedforward in self.attn_layers:
            tokens_emb, _ = attn(tokens_emb, batch_idxs)
            tokens_emb = feedforward(tokens_emb, batch_idxs)

        # batch_size x max_len x d_model
        tokens_emb_padded = batch_idxs.inflate(tokens_emb)
        return tokens_emb_padded
