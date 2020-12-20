"""
Beam search for evaluation
"""
from transition_systems import AttachJuxtapose
import hydra
from omegaconf.dictconfig import DictConfig
from models.parser import Parser
import torch
from time import time
from omegaconf import DictConfig
from evaluation_metric import FScore, evalb
from env import State
from progressbar import ProgressBar
from tree import InternalParseNode, Tree
from utils import get_device
from typing import List
import logging

log = logging.getLogger(__name__)


class Beam:
    batch_size: int
    beam_size: int
    tokens_word: List[List[str]]
    tags: List[List[str]]
    state: State
    log_probs: torch.Tensor  # batch_size x beam_size
    n_step: int
    finished: torch.Tensor  # batch_size
    pred_trees: List[List[Tree]]
    model: Parser

    def __init__(
        self,
        tokens_word: List[List[str]],
        tags: List[List[str]],
        tokens_emb: torch.Tensor,
        model: Parser,
        cfg: DictConfig,
    ) -> None:
        self.batch_size = len(tokens_word)
        self.beam_size = cfg.beam_size
        self.tokens_word = tokens_word
        self.tokens_emb = tokens_emb
        self.tags = tags
        self.model = model
        device = tokens_emb.device

        init_state = State(
            [None for _ in range(self.batch_size)],
            tokens_word,
            tokens_emb,
            next_token_pos=torch.zeros(
                self.batch_size, dtype=torch.int64, device=device
            ),
            n_step=0,
            batch_idx=list(range(self.batch_size)),
        )

        actions, log_probs = self.model(init_state, topk=self.beam_size)
        self.log_probs = log_probs

        partial_trees = []
        tokens_word_expanded = []
        tokens_emb_expanded = []
        batch_idx = []
        self.finished = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        self.pred_trees = [
            [None for _ in range(self.beam_size)] for _ in range(self.batch_size)
        ]

        for i in range(self.batch_size):
            for j in range(self.beam_size):
                tag = self.tags[i][0]
                word = self.tokens_word[i][0]
                tree = AttachJuxtapose.execute(
                    None, actions[i][j], 0, tag, word, immutable=False
                )
                assert isinstance(tree, InternalParseNode)
                if len(self.tokens_word[i]) > 1:
                    partial_trees.append(tree)
                    tokens_word_expanded.append(tokens_word[i])
                    tokens_emb_expanded.append(tokens_emb[i])
                    batch_idx.append(i)
                else:
                    self.finished[i] = True
                    self.pred_trees[i][j] = tree

        tokens_emb_expanded_t = torch.stack(tokens_emb_expanded)

        self.state = State(
            partial_trees,  # type: ignore
            tokens_word_expanded,
            tokens_emb_expanded_t,
            next_token_pos=torch.ones(
                len(partial_trees), dtype=torch.int64, device=device
            ),
            n_step=1,
            batch_idx=batch_idx,
        )
        self.n_step = 1

    def done(self) -> bool:
        return self.finished.all().item()  # type: ignore

    def grow(self) -> bool:
        actions, log_probs = self.model(self.state, topk=self.beam_size)

        x = log_probs.view(
            -1, self.beam_size, self.beam_size
        )  # batch_size x beam_size x beam_size
        x = self.log_probs[~self.finished].unsqueeze(-1) + x
        y = x.view(
            -1, self.beam_size * self.beam_size
        )  # batch_size x (beam_size * beam_size)
        values, indices = y.topk(self.beam_size, dim=-1)
        self.log_probs[~self.finished] = values

        partial_trees = []
        tokens_word = []
        tokens_emb = []
        batch_idx = []
        cnt = 0

        for i in range(self.batch_size):
            if self.finished[i]:
                continue

            for j in range(self.beam_size):
                idx = indices[cnt, j].item()
                m = cnt * self.beam_size + idx // self.beam_size
                n = idx % self.beam_size
                current_tree = self.state.partial_trees[m]
                action = actions[m][n]
                tag = self.tags[i][self.n_step]
                word = self.tokens_word[i][self.n_step]
                tree = AttachJuxtapose.execute(
                    current_tree, action, self.n_step, tag, word, immutable=True,
                )
                assert isinstance(tree, InternalParseNode)
                if self.n_step >= len(self.tokens_word[i]) - 1:
                    self.finished[i] = True
                    self.pred_trees[i][j] = tree
                else:
                    partial_trees.append(tree)
                    tokens_word.append(self.tokens_word[i])
                    tokens_emb.append(self.tokens_emb[i])
                    batch_idx.append(self.state.batch_idx[m])

            cnt += 1

        if tokens_emb == []:
            assert self.done()
            return True

        tokens_emb_t = torch.stack(tokens_emb)

        self.n_step += 1
        self.state = State(
            partial_trees,  # type: ignore
            tokens_word,
            tokens_emb_t,
            self.state.next_token_pos.new_full(
                (len(partial_trees),), fill_value=self.n_step
            ),
            n_step=self.n_step,
            batch_idx=batch_idx,
        )
        return False

    def best_trees(self) -> List[InternalParseNode]:
        return [
            self.pred_trees[i][j]
            for i, j in enumerate(self.log_probs.argmax(dim=-1).tolist())
        ]


def beam_search(
    loader: torch.utils.data.DataLoader, model: Parser, cfg: DictConfig  # type: ignore
) -> FScore:
    "Run validation/testing with beam search"

    model.eval()
    device, _ = get_device()
    gt_trees = []
    pred_trees = []
    bar = ProgressBar(max_value=len(loader))
    time_start = time()

    with torch.no_grad():  # type: ignore

        for i, data_batch in enumerate(loader):
            # calculate token embeddings
            tokens_emb = model.encoder(
                data_batch["tokens_idx"].to(device=device, non_blocking=True),
                data_batch["tags_idx"].to(device=device, non_blocking=True),
                data_batch["valid_tokens_mask"].to(device=device, non_blocking=True),
                data_batch["word_end_mask"].to(device=device, non_blocking=True),
            )
            # initialize the beam
            beam = Beam(
                data_batch["tokens_word"], data_batch["tags"], tokens_emb, model, cfg,
            )
            # keep executing actions and updating the beam until the entire batch is finished
            while not beam.grow():
                pass

            gt_trees.extend(data_batch["trees"])
            pred_trees.extend(beam.best_trees())

            bar.update(i)

    f1_score = evalb(hydra.utils.to_absolute_path("./EVALB"), gt_trees, pred_trees)
    log.info("Time elapsed: %f" % (time() - time_start))
    return f1_score
