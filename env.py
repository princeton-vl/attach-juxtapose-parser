"""
Environment for training, validation, and testing

A state consists of a batch of partially parsed sentences
The environment takes a bactch of actions to update the state
It also pre-computes the token embeddings and splits a data batch in to multiple subbatches
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tree import InternalParseNode, Tree
from transition_systems import Action, AttachJuxtapose
from collections import defaultdict
from typing import Dict, Iterator, List, Any, Sequence, Tuple, Optional
from utils import get_device


class EpochEnd(Exception):
    """
    Exception raised by Environment.reset() when an epoch ends
    """

    pass


class State:
    """
    Batched parser state
    """

    # a batch of partial trees
    partial_trees: Sequence[Tree]
    # a list where each element is a list of tokens in a sentence
    tokens_word: List[List[str]]
    # batch_size x max_len x d_model
    tokens_emb: torch.Tensor
    # batch_size, the next token's position in the sentence (starting from 0)
    next_token_pos: torch.Tensor
    # the number of actions executed on the current batch
    n_step: int
    # batch_size, the index in the current batch
    batch_idx: List[int]

    def __init__(
        self,
        partial_trees: List[Tree],
        tokens_word: List[List[str]],
        tokens_emb: torch.Tensor,
        next_token_pos: torch.Tensor,
        n_step: int,
        batch_idx: List[int],
    ) -> None:
        assert all(next_token_pos[i] <= len(sent) for i, sent in enumerate(tokens_word))
        assert (
            len(partial_trees)
            == len(tokens_word)
            == tokens_emb.size(0)
            == next_token_pos.size(0)
        )
        self.partial_trees = partial_trees
        self.tokens_word = tokens_word
        self.tokens_emb = tokens_emb
        self.next_token_pos = next_token_pos
        self.n_step = n_step
        self.batch_idx = batch_idx

    @property
    def batch_size(self) -> int:
        "Here 'batch' actually means subbatch"
        return len(self.partial_trees)


def is_completing(state: State, i: int) -> bool:
    "See if the ith partial tree in the state is about to complete"
    pos = state.next_token_pos[i].item()
    return pos >= len(state.tokens_word[i]) - 1


class Environment:
    "Environment for executing actions to update the state"

    # data loader
    loader: Iterator[Any]
    # token encoder
    encoder: nn.Module
    # CPU or GPU
    device: torch.device
    # the maximum number of tokens in a subbatch
    subbatch_max_tokens: int
    state: State
    # the entire batch
    cached_data: Dict[str, Any]
    # the current data subbatch
    data_batch: Dict[str, Any]
    # the current subbatch is cached_data[_start_idx : _end_idx] in the entire batch
    _start_idx: int
    _end_idx: int
    # predicted trees in the current subbatch
    pred_trees: List[Optional[InternalParseNode]]
    # ground truth trees in the current subbatch
    gt_trees: List[InternalParseNode]
    # tensors in a batch
    _tensors_to_load = ["tokens_idx", "word_end_mask", "valid_tokens_mask", "tags_idx"]

    def __init__(
        self,
        loader: DataLoader,  # type: ignore
        encoder: torch.nn.Module,
        subbatch_max_tokens: int,
    ) -> None:
        self.loader = iter(loader)
        self.encoder = encoder
        self.device, _ = get_device()
        self.subbatch_max_tokens = subbatch_max_tokens

        self.cached_data = {}
        self.data_batch = {}
        self.pred_trees = []
        self.gt_trees = []

    def _load_data(self) -> None:
        """
        Load a data subbatch to self.data_batch
        The loaded data examples range from self._start_idx to self._end_idx in self.cached_data
        """

        need_data = (
            self.cached_data == {} or self._end_idx >= self.cached_data["num_examples"]
        )
        if need_data:  # need to load another batch
            try:
                self.cached_data = next(self.loader)
            except StopIteration:
                raise EpochEnd()
            self.cached_data["num_examples"] = len(self.cached_data["tokens_word"])
            self._end_idx = 0

        self._start_idx = self._end_idx
        self._end_idx += 1

        # increase self._end_idx until reaching subbatch_max_tokens
        lens = self.cached_data["valid_tokens_mask"].detach().sum(dim=-1)
        max_len = lens[self._start_idx].item()
        while self._end_idx < self.cached_data["num_examples"]:
            max_len = max(max_len, lens[self._end_idx].item())
            # including dummy tokens due to padding
            total_num_tokens = max_len * (self._end_idx - self._start_idx + 1)
            if total_num_tokens > self.subbatch_max_tokens:
                max_len = lens[self._start_idx : self._end_idx].max().item()
                break
            self._end_idx += 1

        for k, v in self.cached_data.items():
            # no data left in self.data_batch
            assert k not in self.data_batch or self.data_batch[k] == []
            if k == "num_examples":
                pass
            elif k in Environment._tensors_to_load:
                self.data_batch[k] = v[self._start_idx : self._end_idx, :max_len].to(
                    device=self.device, non_blocking=True
                )
            else:
                self.data_batch[k] = v[self._start_idx : self._end_idx]

    def reset(self, force: bool = False) -> State:
        """
        Reset a completed state

        1. Load a new data subbatch into self.data_batch
        2. Run the token encoder on self.data_batch
        force: reset even the current subbatch hasn't completed
        """

        if force:
            self.data_batch = {}
        self._load_data()  # load some data examples
        batch_size = self.data_batch["tokens_idx"].size(0)

        # run the token encoder
        is_train = self.encoder.training
        with torch.set_grad_enabled(is_train):
            tokens_emb = self.encoder(
                self.data_batch["tokens_idx"],
                self.data_batch["tags_idx"],
                self.data_batch["valid_tokens_mask"],
                self.data_batch["word_end_mask"],
            )

        # initialize the state
        self.state = State(
            [None for _ in range(batch_size)],
            self.data_batch["tokens_word"],
            tokens_emb,
            next_token_pos=torch.zeros(
                batch_size, dtype=torch.int64, device=self.device
            ),
            n_step=0,
            batch_idx=list(range(batch_size)),
        )
        self.pred_trees = [None for _ in range(batch_size)]
        self.gt_trees = self.data_batch["trees"] if "trees" in self.data_batch else None

        return self.state

    def step(self, actions: List[Action]) -> Tuple[State, bool]:
        """
        Execute a subbatch of actions to update the state
        """

        batch_size = len(actions)
        assert batch_size == len(self.state.partial_trees)
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        new_partial_trees = []
        new_tokens_word = []
        new_next_token_pos = []
        new_batch_idx = []
        data_batch = defaultdict(list)

        for i, action in enumerate(actions):
            # execute the ith action
            pos = self.state.next_token_pos[i].item()
            tag = self.data_batch["tags"][i][pos]
            word = self.data_batch["tokens_word"][i][pos]
            tree = AttachJuxtapose.execute(
                self.state.partial_trees[i], action, pos, tag, word, immutable=False,  # type: ignore
            )

            if is_completing(self.state, i):
                # the tree is completed
                done[i] = True
                assert tree is not None
                batch_idx = self.state.batch_idx[i]
                assert self.pred_trees[batch_idx] is None
                self.pred_trees[batch_idx] = tree
            else:
                # the tree hasn't completed
                new_partial_trees.append(tree)
                new_tokens_word.append(self.state.tokens_word[i])
                for k, v in self.data_batch.items():
                    data_batch[k].append(v[i])
                new_next_token_pos.append(pos + 1)
                new_batch_idx.append(self.state.batch_idx[i])

        self.data_batch = dict(data_batch)
        self.state.partial_trees = new_partial_trees
        self.state.tokens_word = new_tokens_word
        self.state.tokens_emb = self.state.tokens_emb[~done]
        self.state.next_token_pos = self.state.next_token_pos.new_tensor(
            new_next_token_pos
        )
        self.state.n_step += 1
        self.state.batch_idx = new_batch_idx
        all_done = done.all().item()
        assert isinstance(all_done, bool)

        return self.state, all_done

    def gt_actions(self) -> List[Action]:
        """
        Get the ground truth actions at the current step
        """
        return [
            action_seq[self.state.n_step].normalize()
            for action_seq in self.data_batch["action_seq"]
        ]

    def gt_action_seqs(self) -> List[List[Action]]:
        """
        Get all ground truth actions for the current subbatch
        """
        return [
            [action.normalize() for action in action_seq]
            for action_seq in self.data_batch["action_seq"]
        ]
