"""
Dataloaders for PTB and CTB
"""
from collections import defaultdict
import functools
import hydra
import numpy as np
from omegaconf import DictConfig
from progressbar import ProgressBar
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tree import (
    TreeBatch,
    Label,
    DUMMY_LABEL,
    ParseNode,
    InternalParseNode,
    LeafParseNode,
)
from transition_systems import AttachJuxtapose
from typing import Dict, Any, Optional, List, Tuple, Set
import logging

log = logging.getLogger(__name__)

# Some simple transformations to normalize tokens before encoding
# Also used in prior work: see https://github.com/nikitakit/self-attentive-parser/blob/master/src/parse_nk.py
TOKEN_MAPPING: Dict[str, str] = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‹": "'",
    "›": "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}


class TreeBank(Dataset):  # type: ignore
    """
    A treebank dataset such as PTB and CTB
    """

    # vocabularies of tokens, POS tags, and consituency labels
    vocabs: Dict[str, Any]
    # a mapping from POS tags to their indice in the vocabulary
    tag_idx_map: Dict[str, int]
    # a list of parse trees
    trees: TreeBatch
    # tokenizer used by transformers
    tokenizer: Any

    def __init__(
        self, datapath: str, split: str, encoder: str, vocabs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__()
        assert split in ["train", "val", "test"]

        # read constituency trees
        log.info("Loading constituency trees from " + datapath)
        self.trees = TreeBatch.from_file(datapath)

        if vocabs is not None:  # the vocabs are given
            self.vocabs = vocabs

        else:  # create new vocabs from data
            tag_vocab = set()
            label_vocab: Set[Label] = {DUMMY_LABEL}
            token_freq: Dict[str, int] = defaultdict(int)

            def collect_labels(node: ParseNode) -> None:
                if isinstance(node, LeafParseNode):
                    tag_vocab.add(node.tag)
                    token_freq[node.word.lower()] += 1
                else:
                    assert isinstance(node, InternalParseNode)
                    label_vocab.add(node.label)

            self.trees.traverse_preorder(collect_labels)
            self.vocabs = {
                "label": sorted(list(label_vocab)),
                "tag": sorted(list(tag_vocab)),
                "token": sorted(list(token_freq.keys())),
            }

        self.tag_idx_map = {t: i for i, t in enumerate(self.vocabs["tag"])}

        # tokenizer for pre-trained transformers
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder, do_lower_case=("-cased" not in encoder)
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sentence and its parse tree
        """
        tree = self.trees[idx]
        assert tree is not None

        tokens_word = []
        tags = []
        tags_idx = []
        for node in tree.iter_leaves():
            tokens_word.append(node.word)
            tags.append(node.tag)
            tags_idx.append(self.tag_idx_map[node.tag])

        # oracle action sequence
        action_seq = AttachJuxtapose.oracle_actions(tree, immutable=True)

        example = {
            "tokens_word": tokens_word,  # a list of strings
            "tags": tags,  # a list of strings
            "tags_idx": tags_idx,  # a list of integers
            "tree": tree,  # the parse tree
            "action_seq": action_seq,  # a list of actions
        }

        cleaned_words = self._preprocess(tokens_word)
        subtokens = [self.tokenizer.cls_token]
        word_end_mask = [False]

        for w in cleaned_words:
            subtokens_w = self.tokenizer.tokenize(w)
            word_end_mask.extend([False] * (len(subtokens_w) - 1) + [True])
            subtokens.extend(subtokens_w)

        subtokens.append(self.tokenizer.sep_token)
        word_end_mask.append(False)
        tokens_idx = self.tokenizer.convert_tokens_to_ids(subtokens)

        example["tokens_idx"] = tokens_idx  # a list of integers
        example["word_end_mask"] = word_end_mask  # a list of booleans
        # Some tokens in PTB/CTB correspond to multiple (sub-)tokens in BERT/XLNet
        # word_end_mask is true for the ending sub-token for each token

        return example

    def __len__(self) -> int:
        return len(self.trees)

    def _preprocess(self, words: List[str]) -> List[str]:
        """
        Preprocess the tokens before encoding using transformers
        """
        cleaned_words: List[str] = []
        for w in words:
            w = TOKEN_MAPPING.get(w, w)
            if w == "n't" and cleaned_words != []:  # e.g., wasn't -> wasn 't
                cleaned_words[-1] = cleaned_words[-1] + "n"
                w = "'t"
            cleaned_words.append(w)
        return cleaned_words


def form_batch(encoder: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Put sentences of different lengths into one batch
    Pad with zeros
    """

    batch_size = len(examples)
    max_num_tokens: int = np.max([len(x["tokens_idx"]) for x in examples])
    tokens_idx = torch.zeros(batch_size, max_num_tokens, dtype=torch.int64)
    valid_tokens_mask = torch.zeros_like(tokens_idx, dtype=torch.bool)
    word_end_mask = torch.zeros_like(tokens_idx, dtype=torch.bool)

    max_num_tags = np.max([len(x["tags_idx"]) for x in examples])
    tags_idx = torch.zeros(batch_size, max_num_tags, dtype=torch.int64)

    tokens_word = []
    tags = []
    trees = []
    action_seq = []

    for i, x in enumerate(examples):
        l = len(x["tokens_idx"])
        tokens_idx[i, :l] = tokens_idx.new_tensor(x["tokens_idx"])
        valid_tokens_mask[i, :l] = True
        word_end_mask[i, :l] = word_end_mask.new_tensor(x["word_end_mask"])
        tokens_word.append(x["tokens_word"])
        tags.append(x["tags"])
        tags_idx[i, : len(x["tags_idx"])] = tags_idx.new_tensor(x["tags_idx"])
        trees.append(x["tree"])
        action_seq.append(x["action_seq"])

    data_batch = {
        "tokens_word": tokens_word,  # List[List[str]]
        "tokens_idx": tokens_idx,  # 2-D tensor
        "valid_tokens_mask": valid_tokens_mask,  # 2d tensor
        "tags": tags,  # List[List[str]]
        "tags_idx": tags_idx,  # 2-D tensor
        "trees": trees,  # a list of parse trees
        "action_seq": action_seq,  # a list of lists of actions
        "word_end_mask": word_end_mask,  # 2-D tensor
    }

    return data_batch


def create_dataloader(
    datapath: str,
    split: str,
    encoder: str,
    vocabs: Optional[Dict[str, Any]],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader[Any], Optional[Dict[str, Any]]]:
    is_train = "train" in split
    ds = TreeBank(datapath, split, encoder, vocabs)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=functools.partial(form_batch, encoder),
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
    )
    return (loader, ds.vocabs) if is_train else (loader, None)


@hydra.main(config_path="conf/train.yaml")
def main(cfg: DictConfig) -> None:
    log.info("\n" + cfg.pretty())

    # creat data loaders
    loader_train, vocabs = create_dataloader(
        hydra.utils.to_absolute_path(cfg.path_train),
        "train",
        cfg.encoder,
        None,
        cfg.batch_size,
        cfg.num_workers,
    )
    loader_val, _ = create_dataloader(
        hydra.utils.to_absolute_path(cfg.path_val),
        "val",
        cfg.encoder,
        vocabs,
        cfg.batch_size,
        cfg.num_workers,
    )
    loader_test, _ = create_dataloader(
        hydra.utils.to_absolute_path(cfg.path_test),
        "test",
        cfg.encoder,
        vocabs,
        cfg.batch_size,
        cfg.num_workers,
    )

    # Loading the data and perform some sanity checks
    for loader in [loader_train, loader_val, loader_test]:
        bar = ProgressBar(max_value=len(loader))
        for i, data_batch in enumerate(loader):
            for j in range(len(data_batch["trees"])):
                # check if the tree is well-formed
                tree = data_batch["trees"][j]
                assert tree.is_well_formed()
                # convert the tree to actions and convert back
                words = [t.word for t in tree.iter_leaves()]
                tags = [t.tag for t in tree.iter_leaves()]
                actions = data_batch["action_seq"][j]
                reconstructed_tree = AttachJuxtapose.actions2tree(words, tags, actions)
                assert reconstructed_tree.is_well_formed()
                s1 = tree.linearize()
                s2 = reconstructed_tree.linearize()
                assert s1 == s2

            bar.update(i)

    log.info("Testing completed. The dataloader seems to work fine.")


if __name__ == "__main__":
    main()
