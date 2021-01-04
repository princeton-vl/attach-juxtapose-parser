"""
The script for parsing user-provided texts
"""
import hydra
import torch
import numpy as np
import sys
from time import time
from torch.utils.data import DataLoader, Dataset
from dataloader import TOKEN_MAPPING
from omegaconf import DictConfig, OmegaConf
from models.parser import Parser
from env import Environment, EpochEnd
from transformers import AutoTokenizer
from test import restore_hyperparams
from utils import get_device, load_model
import spacy
from progressbar import ProgressBar
from typing import List, Dict, Any
import logging

log = logging.getLogger(__name__)


class UserProvidedTexts(Dataset):  # type: ignore
    words: List[List[str]]
    tags: List[List[str]]
    vocabs: Dict[str, Any]
    tag_idx_map: Dict[str, int]

    def __init__(
        self, filename: str, language: str, vocabs: Dict[str, Any], encoder: str
    ) -> None:
        self.words = []
        self.tags = []
        self.vocabs = vocabs
        self.tag_idx_map = {t: i for i, t in enumerate(self.vocabs["tag"])}

        spacy_model = spacy.load(
            "en_core_web_sm" if language == "english" else "zh_core_web_sm"
        )
        bar = ProgressBar()
        log.info("Loading input sentences and performing POS tagging..")

        for i, line in enumerate(open(filename)):
            sentence = line.strip()
            assert sentence != ""
            words_sent = []
            tags_sent = []
            for t in spacy_model(sentence):
                words_sent.append(t.text)
                tags_sent.append(t.tag_)
            self.words.append(words_sent)
            self.tags.append(tags_sent)
            bar.update(i)

        log.info("%d input sentences loaded from %s" % (len(self.words), filename))

        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder, do_lower_case=("-cased" not in encoder)
        )

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cleaned_words = self._preprocess(self.words[idx])
        subtokens = [self.tokenizer.cls_token]
        word_end_mask = [False]

        for w in cleaned_words:
            subtokens_w = self.tokenizer.tokenize(w)
            word_end_mask.extend([False] * (len(subtokens_w) - 1) + [True])
            subtokens.extend(subtokens_w)

        subtokens.append(self.tokenizer.sep_token)
        word_end_mask.append(False)
        tokens_idx = self.tokenizer.convert_tokens_to_ids(subtokens)

        tags_idx = [self.tag_idx_map[t] for t in self.tags[idx]]

        return {
            "tokens_word": self.words[idx],  # a list of strings
            "tags": self.tags[idx],  # a list of strings
            "tags_idx": tags_idx,  # a list of integers
            "tokens_idx": tokens_idx,  # a list of integers
            "word_end_mask": word_end_mask,  # a list of booleans
        }

    def __len__(self) -> int:
        return len(self.words)


def form_batch(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(examples)
    max_num_tokens: int = np.max([len(x["tokens_idx"]) for x in examples])
    tokens_idx = torch.zeros(batch_size, max_num_tokens, dtype=torch.int64)
    valid_tokens_mask = torch.zeros_like(tokens_idx, dtype=torch.bool)
    word_end_mask = torch.zeros_like(tokens_idx, dtype=torch.bool)

    max_num_tags = np.max([len(x["tags_idx"]) for x in examples])
    tags_idx = torch.zeros(batch_size, max_num_tags, dtype=torch.int64)
    tokens_word = []
    tags = []

    for i, x in enumerate(examples):
        l = len(x["tokens_idx"])
        tokens_idx[i, :l] = tokens_idx.new_tensor(x["tokens_idx"])
        valid_tokens_mask[i, :l] = True
        word_end_mask[i, :l] = word_end_mask.new_tensor(x["word_end_mask"])
        tokens_word.append(x["tokens_word"])
        tags.append(x["tags"])
        tags_idx[i, : len(x["tags_idx"])] = tags_idx.new_tensor(x["tags_idx"])

    data_batch = {
        "batch_idx": list(range(batch_size)),  # List[int]
        "tokens_word": tokens_word,  # List[List[str]]
        "tokens_idx": tokens_idx,  # 2-D tensor
        "valid_tokens_mask": valid_tokens_mask,  # 2d tensor
        "tags": tags,  # List[List[str]]
        "tags_idx": tags_idx,  # 2-D tensor
        "word_end_mask": word_end_mask,  # 2-D tensor
    }

    return data_batch


@hydra.main(config_path="conf", config_name="parse.yaml")
def main(cfg: DictConfig) -> None:
    "The entry point for parsing user-provided texts"

    assert cfg.model_path is not None, "Need to specify model_path for testing."
    assert cfg.input is not None
    assert cfg.language in ("english", "chinese")
    log.info("\n" + OmegaConf.to_yaml(cfg))

    # load the model checkpoint
    model_path = hydra.utils.to_absolute_path(cfg.model_path)
    log.info("Loading the model from %s" % model_path)
    checkpoint = load_model(model_path)
    restore_hyperparams(checkpoint["cfg"], cfg)
    vocabs = checkpoint["vocabs"]

    model = Parser(vocabs, cfg)
    model.load_state_dict(checkpoint["model_state"])
    device, _ = get_device()
    model.to(device)
    log.info("\n" + str(model))
    log.info("#parameters = %d" % sum([p.numel() for p in model.parameters()]))

    input_file = hydra.utils.to_absolute_path(cfg.input)
    ds = UserProvidedTexts(input_file, cfg.language, vocabs, cfg.encoder)
    loader = DataLoader(
        ds,
        batch_size=cfg.eval_batch_size,
        collate_fn=form_batch,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    env = Environment(loader, model.encoder, subbatch_max_tokens=9999999)
    state = env.reset()
    oup = (
        sys.stdout
        if cfg.output is None
        else open(hydra.utils.to_absolute_path(cfg.output), "wt")
    )
    time_start = time()

    with torch.no_grad():  # type: ignore
        while True:
            with torch.cuda.amp.autocast(cfg.amp):  # type: ignore
                actions, _ = model(state)
            state, done = env.step(actions)
            if done:
                for tree in env.pred_trees:
                    assert tree is not None
                    print(tree.linearize(), file=oup)
                # pred_trees.extend(env.pred_trees)
                # load the next batch
                try:
                    with torch.cuda.amp.autocast(cfg.amp):  # type: ignore
                        state = env.reset()
                except EpochEnd:
                    # no next batch available (complete)
                    log.info("Time elapsed: %f" % (time() - time_start))
                    break

    if cfg.output is not None:
        log.info("Parse trees saved to %s" % cfg.output)


if __name__ == "__main__":
    main()
