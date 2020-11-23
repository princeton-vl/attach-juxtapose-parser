"""
Top-level model for the parser
"""
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import GraphDecoder, SequenceDecoder
from omegaconf import DictConfig
from env import State
from typing import Union, Dict, Any, Tuple, Optional


class Parser(nn.Module):
    encoder: Encoder
    decoder: Union[GraphDecoder, SequenceDecoder]

    def __init__(self, vocabs: Dict[str, Any], cfg: DictConfig) -> None:
        super().__init__()  # type: ignore

        # The position embedding used by both the self-attention encoder and the GCN
        d_positional = cfg.d_model - cfg.d_model // 2
        self.position_table = nn.Parameter(
            torch.empty(cfg.max_sentence_len, d_positional)
        )
        nn.init.normal_(self.position_table)

        self.encoder = Encoder(self.position_table, vocabs, cfg)

        if cfg.decoder == "graph":
            self.decoder = GraphDecoder(self.position_table, vocabs, cfg)
        else:
            assert cfg.decoder == "sequence"
            self.decoder = SequenceDecoder(vocabs, cfg)

    def forward(self, state: State, topk: Optional[int] = None) -> Any:
        return (
            self.decoder(state, topk)
            if isinstance(self.decoder, GraphDecoder)
            else self.decoder(state)
        )
