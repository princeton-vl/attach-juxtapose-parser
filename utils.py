"""
Some utility functions
"""
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from typing import Tuple, Any, List, Union, Dict


def get_device() -> Tuple[torch.device, bool]:
    "Get GPU if available"

    use_gpu = torch.cuda.is_available()
    return torch.device("cuda" if use_gpu else "cpu"), use_gpu


def load_model(model_path: str) -> Dict[str, Any]:
    "Load a model checkpoint"

    if torch.cuda.is_available():
        return torch.load(model_path)  # type: ignore
    else:
        return torch.load(model_path, map_location=torch.device("cpu"))  # type: ignore


def count_params(model: nn.Module) -> int:
    "The number of parameters in a PyTorch model"

    return sum([p.numel() for p in model.parameters()])


def count_actions(
    pred: Union[List[Any], List[List[Any]]], gt: Union[List[Any], List[List[Any]]]
) -> Tuple[int, int]:
    "Count the number of correct actions and the number of total actions"

    if isinstance(pred[0], list):
        num_correct = np.sum(
            np.sum(x == y for x, y in zip(pred_seq, gt_seq))
            for pred_seq, gt_seq in zip(pred, gt)
        )
        num_total = np.sum(len(pred_seq) for pred_seq in pred)

    else:
        num_correct = np.sum([x == y for x, y in zip(pred, gt)])
        num_total = len(pred)

    return num_correct, num_total


def conf2list(cfg: ListConfig) -> List[Any]:
    cfg_list: List[Any] = []
    for v in cfg:
        if isinstance(v, ListConfig):
            cfg_list.append(conf2list(v))
        elif isinstance(v, DictConfig):
            cfg_list.append(conf2dict(v))
        else:
            assert v is None or isinstance(v, (str, int, float, bool))
            cfg_list.append(v)
    return cfg_list


def conf2dict(cfg: DictConfig) -> Dict[str, Any]:
    cfg_dict: Dict[str, Any] = {}
    for k, v in cfg.items():
        assert isinstance(k, str)
        if isinstance(v, ListConfig):
            cfg_dict[k] = conf2list(v)
        elif isinstance(v, DictConfig):
            cfg_dict[k] = conf2dict(v)
        else:
            assert v is None or isinstance(v, (str, int, float, bool))
            cfg_dict[k] = v
    return cfg_dict
