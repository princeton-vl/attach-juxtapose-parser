"""
The script for validation and testing
"""
from transition_systems import AttachJuxtapose
import hydra
from omegaconf.dictconfig import DictConfig
from dataloader import create_dataloader
from models.parser import Parser
import itertools
import torch
import numpy as np
from time import time
from omegaconf import DictConfig
from evaluation_metric import FScore, evalb
from env import Environment, EpochEnd, State
from progressbar import ProgressBar
from tree import InternalParseNode
from utils import get_device
from beam_search import beam_search
from typing import Dict, Any
import logging

log = logging.getLogger(__name__)


def validate_beam_search(loader: torch.utils.data.DataLoader, model: Parser, cfg: DictConfig) -> FScore:  # type: ignore
    "Run validation/testing with beam search"

    model.eval()
    gt_trees = []
    pred_trees = []
    bar = ProgressBar(max_value=len(loader))
    time_start = time()

    with torch.no_grad():  # type: ignore
        # with torch.cuda.amp.autocast(cfg.amp):
        device, _ = get_device()

        for i, data_batch in enumerate(loader):
            tokens_emb = model.encoder(
                data_batch["tokens_idx"].to(device=device, non_blocking=True),
                data_batch["tags_idx"].to(device=device, non_blocking=True),
                data_batch["valid_tokens_mask"].to(device=device, non_blocking=True),
                data_batch["word_end_mask"].to(device=device, non_blocking=True),
            )
            next_token_pos = torch.zeros(1, dtype=torch.int64, device=device)
            assert len(data_batch["tokens_word"]) == 1
            beam = [
                State(
                    [None],
                    data_batch["tokens_word"],
                    tokens_emb,
                    next_token_pos,
                    n_step=0,
                )
            ]
            beam_log_prob = [0.0]

            for n_step, (word, tag) in enumerate(
                zip(data_batch["tokens_word"][0], data_batch["tags"][0])
            ):
                next_token_pos = torch.full_like(next_token_pos, fill_value=n_step + 1)
                new_beam = []
                new_beam_log_prob = []

                # try to expand each element in the beam
                for state, logp in zip(beam, beam_log_prob):
                    actions, log_probs = model(state, topk=cfg.beam_size)
                    new_beam_log_prob.extend((logp + log_probs).tolist())
                    for act in actions:
                        tree = AttachJuxtapose.execute(
                            state.partial_trees[0],
                            act,
                            n_step,
                            tag,
                            word,
                            immutable=True,
                        )
                        new_beam.append(
                            State(
                                [tree],
                                data_batch["tokens_word"],
                                tokens_emb,
                                next_token_pos,
                                n_step,
                            )
                        )

                idxs_to_keep = np.argsort(new_beam_log_prob)[::-1][: cfg.beam_size]
                beam = np.array(new_beam)[idxs_to_keep]
                beam_log_prob = np.array(new_beam_log_prob)[idxs_to_keep]

            # take the tree with maximum log probability
            assert beam_log_prob[0] == max(beam_log_prob)
            max_tree = beam[0].partial_trees[0]
            assert isinstance(max_tree, InternalParseNode)
            pred_trees.append(max_tree)
            gt_trees.append(data_batch["trees"][0])

            bar.update(i)

        f1_score = evalb(hydra.utils.to_absolute_path("./EVALB"), gt_trees, pred_trees)
        log.info("Time elapsed: %f" % (time() - time_start))
        return f1_score


def validate(loader: torch.utils.data.DataLoader, model: Parser, cfg: DictConfig) -> FScore:  # type: ignore
    "Run validation/testing without beam search"

    model.eval()
    # testing requires far less GPU memory than training
    # so there is no need to split a batch into multiple subbatches
    env = Environment(loader, model.encoder, subbatch_max_tokens=9999999)
    state = env.reset()

    pred_trees = []
    gt_trees = []
    time_start = time()

    with torch.no_grad():  # type: ignore
        while True:
            with torch.cuda.amp.autocast(cfg.amp):  # type: ignore
                actions, _ = model(state)

            if cfg.decoder == "graph":
                # actions for a single step
                state, done = env.step(actions)
                if not done:
                    continue
            else:
                assert cfg.decoder == "sequence"
                # actions for all steps
                for n_step in itertools.count():
                    a_t = [
                        action_seq[n_step]
                        for action_seq in actions
                        if len(action_seq) > n_step
                    ]
                    _, done = env.step(a_t)
                    if done:
                        break

            pred_trees.extend(env.pred_trees)
            gt_trees.extend(env.gt_trees)

            # load the next batch
            try:
                with torch.cuda.amp.autocast(cfg.amp):  # type: ignore
                    state = env.reset()
            except EpochEnd:
                # no next batch available (complete)
                f1_score = evalb(
                    hydra.utils.to_absolute_path("./EVALB"), gt_trees, pred_trees
                )
                log.info("Time elapsed: %f" % (time() - time_start))
                return f1_score


def restore_hyperparams(saved_cfg: Dict[str, Any], cfg: DictConfig) -> DictConfig:
    """
    Restore the hyperparameters in a checkpoint
    """
    log.info("Restoring hyperparameters from the saved model checkpoint..")
    for name in cfg.model_spec:
        if name not in saved_cfg:
            log.warning("Missing: %s" % name)
            continue
        value = saved_cfg[name]
        if name in cfg and getattr(cfg, name) != value:
            log.warning("Overriding %s -> %s" % (str(getattr(cfg, name)), str(value)))
        setattr(cfg, name, value)
        log.info("%s: %s" % (name, str(value)))
    return cfg


@hydra.main(config_path="conf/test.yaml", strict=False)
def main(cfg: DictConfig) -> None:
    "The entry point for testing"

    assert cfg.model_path is not None, "Need to specify model_path for testing."
    log.info("\n" + cfg.pretty())

    # restore the hyperparameters used for training
    model_path = hydra.utils.to_absolute_path(cfg.model_path)
    log.info("Loading the model from %s" % model_path)
    checkpoint = torch.load(model_path)  # type: ignore
    restore_hyperparams(checkpoint["cfg"], cfg)

    # create dataloaders for validation and testing
    vocabs = checkpoint["vocabs"]
    loader_val, _ = create_dataloader(
        hydra.utils.to_absolute_path(cfg.path_val),
        "val",
        cfg.encoder,
        vocabs,
        cfg.eval_batch_size,
        cfg.num_workers,
    )
    loader_test, _ = create_dataloader(
        hydra.utils.to_absolute_path(cfg.path_test),
        "test",
        cfg.encoder,
        vocabs,
        cfg.eval_batch_size,
        cfg.num_workers,
    )

    # restore the trained model checkpoint
    model = Parser(vocabs, cfg)
    model.load_state_dict(checkpoint["model_state"])
    device, _ = get_device()
    model.to(device)
    log.info("\n" + str(model))
    log.info("#parameters = %d" % sum([p.numel() for p in model.parameters()]))

    # validation
    log.info("Validating..")
    f1_score = validate(loader_val, model, cfg)
    log.info(
        "Validation F1 score: %.03f, Exact match: %.03f, Precision: %.03f, Recall: %.03f"
        % (
            f1_score.fscore,
            f1_score.complete_match,
            f1_score.precision,
            f1_score.recall,
        )
    )

    # testing
    log.info("Testing..")
    if cfg.beam_size > 1:
        log.info("Performing beam search..")
        f1_score = beam_search(loader_test, model, cfg)
    else:
        log.info("Running without beam search..")
        f1_score = validate(loader_test, model, cfg)
    log.info(
        "Testing F1 score: %.03f, Exact match: %.03f, Precision: %.03f, Recall: %.03f"
        % (
            f1_score.fscore,
            f1_score.complete_match,
            f1_score.precision,
            f1_score.recall,
        )
    )


if __name__ == "__main__":
    main()
