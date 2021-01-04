"""
The script for training
"""
import subprocess
from omegaconf import DictConfig, OmegaConf
import torch
import os
import shutil
import numpy as np
from time import time
import random
import hydra
from dataloader import create_dataloader
from env import Environment, EpochEnd
from models.parser import Parser
from tree import Label
from utils import get_device, load_model, count_params, count_actions, conf2dict
from loss import action_loss, action_seqs_loss
from test import validate
import gc
from typing import List, Tuple, Dict, Any
import logging

log = logging.getLogger(__name__)


def adjust_lr(
    num_iters: int, optimizer: torch.optim.Optimizer, cfg: DictConfig
) -> None:
    "Increase the learning rate linearly from 0 to cfg.learning_rate in cfg.learning_rate steps."

    warmup_coeff = cfg.learning_rate / cfg.learning_rate_warmup_steps
    new_lr = num_iters * warmup_coeff
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    log.info("Learning rate adjusted to %.08f" % new_lr)


def train(
    num_iters: int,
    loader: torch.utils.data.DataLoader,  # type: ignore
    model: Parser,
    optimizer: torch.optim.Optimizer,
    label_vocab: List[Label],
    cfg: DictConfig,
) -> Tuple[int, float, float]:
    "Train the model for one epoch"

    model.train()
    env = Environment(loader, model.encoder, cfg.subbatch_max_tokens)
    optimizer.zero_grad()
    state = env.reset()
    device, _ = get_device()
    loss = torch.tensor(0.0, device=device)

    # stats
    losses = [0.0]
    num_examples = 0
    num_correct_actions = 0
    num_total_actions = 0

    time_start = time()

    # Each batch is divided into multiple subbatches (for saving GPU memory).
    # Accumulate gradients calculated from subbatches and perform a single optimization step for a batch (not subbatch)
    while True:
        actions, logits = model(state)  # action generation from partial trees

        if cfg.decoder == "graph":
            # for graph-based decoder, actons: List[Action] are actions at the current step for a subbatch
            gt_actions = env.gt_actions()
            loss += action_loss(logits, gt_actions, label_vocab, cfg.batch_size)

            correct, total = count_actions(actions, gt_actions)
            num_correct_actions += correct
            num_total_actions += total

            state, done = env.step(gt_actions)  # teacher forcing
            if done:  # a subbatch is finished
                num_examples += len(env.pred_trees)
            else:
                continue

        else:
            # for sequence-based decoder, actons: List[List[Action]] are action sequences for all steps
            assert cfg.decoder == "sequence"
            all_gt_actions = env.gt_action_seqs()
            loss = action_seqs_loss(logits, all_gt_actions, label_vocab, cfg.batch_size)

            correct, total = count_actions(actions, all_gt_actions)
            num_correct_actions += correct
            num_total_actions += total

            num_examples += len(all_gt_actions)

        # a subbatch is finished
        losses[-1] += loss.item()
        loss.backward()  # type: ignore
        loss = 0  # type: ignore

        if num_examples % cfg.batch_size == 0:  # a full batch is finished
            if num_iters <= cfg.learning_rate_warmup_steps:
                adjust_lr(num_iters, optimizer, cfg)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            losses.append(0)
            num_iters += 1

        try:
            state = env.reset(force=True)  # load a new batch
        except EpochEnd:
            accuracy = 100 * num_correct_actions / num_total_actions
            return num_iters, accuracy, np.mean(losses)

        # log training stats
        if (num_examples / cfg.batch_size) % cfg.log_freq == 0:
            recent_loss = np.mean(losses[-cfg.log_freq :])
            running_accuracy = 100 * num_correct_actions / num_total_actions
            log.info(
                "[%d] Loss: %.03f, Running accuracy: %.03f, Time: %.02f"
                % (num_examples, recent_loss, running_accuracy, time() - time_start,)
            )
            time_start = time()


def save_checkpoint(
    filename: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    f1_score: float,
    vocabs: Dict[str, Any],
    cfg: DictConfig,
) -> None:
    model.cpu()
    path = os.path.join("checkpoints", filename)
    torch.save(
        {
            "cfg": conf2dict(cfg),
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "f1_score": f1_score,
            "vocabs": vocabs,
        },
        path,
    )
    log.info("Checkpoint saved to %s" % path)
    device, _ = get_device()
    model.to(device)


def launch_test(dataset: str) -> None:
    model_path = os.path.join(os.getcwd(), "checkpoints/model_latest.pth")
    os.chdir(hydra.utils.get_original_cwd())
    cmd = "python test.py model_path=%s dataset=%s" % (
        os.path.relpath(model_path),
        dataset,
    )
    log.info(cmd)
    subprocess.run(cmd, shell=True)


def sanity_check(cfg: DictConfig) -> None:
    "Some initialization and sanity checks based on hyperparameters"

    if cfg.random_seed is not None:  # use user-sepecifed random seed
        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        random.seed(cfg.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # sub-batches have to fit the longest sentence
    assert cfg.subbatch_max_tokens >= cfg.max_sentence_len

    if not torch.cuda.is_available():
        log.warning("Using CPU")

    # create the directory for saving model checkpoints
    if os.path.exists("checkpoints"):
        log.warning(
            "The exp_id %s already exists. Previous results will be overridden."
            % cfg.exp_id
        )
        shutil.rmtree("checkpoints")
    os.mkdir("checkpoints")


def train_val(cfg: DictConfig) -> None:

    # create dataloaders for training and validation
    loader_train, vocabs = create_dataloader(
        hydra.utils.to_absolute_path(cfg.path_train),
        "train",
        cfg.encoder,
        None,
        cfg.batch_size,
        cfg.num_workers,
    )
    assert vocabs is not None
    loader_val, _ = create_dataloader(
        hydra.utils.to_absolute_path(cfg.path_val),
        "val",
        cfg.encoder,
        vocabs,
        cfg.eval_batch_size,
        cfg.num_workers,
    )

    # create the model
    model = Parser(vocabs, cfg)
    device, _ = get_device()
    model.to(device)
    log.info("\n" + str(model))
    log.info("#parameters = %d" % count_params(model))

    # create the optimizer
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    start_epoch = 0
    if cfg.resume is not None:  # resume training from a checkpoint
        checkpoint = load_model(cfg.resume)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        del checkpoint
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=cfg.learning_rate_patience,
        cooldown=cfg.learning_rate_cooldown,
        verbose=True,
    )

    # start training and validation
    best_f1_score = -1.0
    num_iters = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        log.info("Epoch #%d" % epoch)

        if not cfg.skip_training:
            log.info("Training..")
            num_iters, accuracy_train, loss_train = train(
                num_iters, loader_train, model, optimizer, vocabs["label"], cfg,
            )
            log.info(
                "Action accuracy: %.03f, Loss: %.03f" % (accuracy_train, loss_train)
            )

        log.info("Validating..")
        f1_score_val = validate(loader_val, model, cfg)

        log.info(
            "Validation F1 score: %.03f, Exact match: %.03f, Precision: %.03f, Recall: %.03f"
            % (
                f1_score_val.fscore,
                f1_score_val.complete_match,
                f1_score_val.precision,
                f1_score_val.recall,
            )
        )

        if f1_score_val.fscore > best_f1_score:
            log.info("F1 score has improved")
            best_f1_score = f1_score_val.fscore

        scheduler.step(best_f1_score)

        save_checkpoint(
            "model_latest.pth",
            epoch,
            model,
            optimizer,
            f1_score_val.fscore,
            vocabs,
            cfg,
        )


@hydra.main(config_path="conf", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    "The entry point for training"

    sanity_check(cfg)
    log.info("\n" + OmegaConf.to_yaml(cfg))

    train_val(cfg)

    log.info("Training completed. Launching the testing script..")
    gc.collect()
    torch.cuda.empty_cache()
    launch_test("ctb" if "ctb" in cfg.path_train else "ptb")


if __name__ == "__main__":
    main()
