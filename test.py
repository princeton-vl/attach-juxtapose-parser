"""
The script for validation and testing
"""
from transition_systems import AttachJuxtapose
import hydra
from dataloader import create_dataloader
from models.parser import Parser
import itertools
import torch
import numpy as np
from time import time
from omegaconf import DictConfig, OmegaConf
from evaluation_metric import FScore, evalb
from env import Environment, EpochEnd, State
from progressbar import ProgressBar
from tree import InternalParseNode
from utils import get_device, load_model
from beam_search import beam_search
from typing import Dict, Any
import logging

log = logging.getLogger(__name__)


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
                    hydra.utils.to_absolute_path("./EVALB"), gt_trees, pred_trees  # type: ignore
                )
                log.info("Time elapsed: %f" % (time() - time_start))
                return f1_score


def restore_hyperparams(saved_cfg: Dict[str, Any], cfg: DictConfig) -> DictConfig:
    """
    Restore the hyperparameters in a checkpoint
    """
    OmegaConf.set_struct(cfg, False)
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


@hydra.main(config_path="conf", config_name="test.yaml")
def main(cfg: DictConfig) -> None:
    "The entry point for testing"

    assert cfg.model_path is not None, "Need to specify model_path for testing."
    log.info("\n" + OmegaConf.to_yaml(cfg))

    # restore the hyperparameters used for training
    model_path = hydra.utils.to_absolute_path(cfg.model_path)
    log.info("Loading the model from %s" % model_path)
    checkpoint = load_model(model_path)
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
