"""
Loss functions for training
"""
import torch
import torch.nn.functional as F
from transition_systems import Action, Action
from tree import Label
from typing import List, Dict


def action_loss(
    logits: Dict[str, torch.Tensor],  # logits for target_node, parent_label, new_label
    gt_actions: List[Action],
    label_vocab: List[Label],
    batch_size: int,
) -> torch.Tensor:
    "Loss function for the attach-juxtapose transition system"

    # convert ground truth actions to tensors
    gt_target_nodes_list = []
    gt_parent_labels_list = []
    gt_new_labels_list = []

    for action in gt_actions:
        gt_target_nodes_list.append(action.target_node)
        gt_parent_labels_list.append(label_vocab.index(action.parent_label))
        gt_new_labels_list.append(label_vocab.index(action.new_label))

    gt_target_nodes = logits["target_node"].new_tensor(
        gt_target_nodes_list, dtype=torch.int64
    )
    gt_parent_labels = logits["parent_label"].new_tensor(
        gt_parent_labels_list, dtype=torch.int64
    )
    gt_new_labels = logits["new_label"].new_tensor(
        gt_new_labels_list, dtype=torch.int64
    )

    # calculate the loss
    node_loss = F.cross_entropy(logits["target_node"], gt_target_nodes, reduction="sum")
    parent_label_loss = F.cross_entropy(
        logits["parent_label"], gt_parent_labels, reduction="sum"
    )
    new_label_loss = F.cross_entropy(
        logits["new_label"], gt_new_labels, reduction="sum"
    )
    loss = (node_loss + parent_label_loss + new_label_loss) / batch_size

    return loss


def action_seqs_loss(
    logits: Dict[str, torch.Tensor],
    gt_actions: List[List[Action]],
    label_vocab: List[Label],
    batch_size: int,
) -> torch.Tensor:
    "Loss function for all actions (used by the sequence-based baseline)"

    # different examples may have different number of actions
    subbatch_size, max_len, _ = logits["target_node"].size()
    valid_action_mask = logits["target_node"].new_zeros(
        (subbatch_size, max_len), dtype=torch.bool
    )
    for i, action_seq in enumerate(gt_actions):
        valid_action_mask[i, : len(action_seq)] = True

    node_logits = logits["target_node"][valid_action_mask]
    parent_label_logits = logits["parent_label"][valid_action_mask]
    new_label_logits = logits["new_label"][valid_action_mask]

    # convert ground truth actions to tensors
    gt_target_nodes_list = []
    gt_parent_labels_list = []
    gt_new_labels_list = []

    for action_seq in gt_actions:
        for action in action_seq:
            gt_target_nodes_list.append(action.target_node)
            gt_parent_labels_list.append(label_vocab.index(action.parent_label))
            gt_new_labels_list.append(label_vocab.index(action.new_label))

    gt_target_nodes = node_logits.new_tensor(gt_target_nodes_list, dtype=torch.int64)
    gt_parent_labels = parent_label_logits.new_tensor(
        gt_parent_labels_list, dtype=torch.int64
    )
    gt_new_labels = new_label_logits.new_tensor(gt_new_labels_list, dtype=torch.int64)

    # calculate the loss
    node_loss = F.cross_entropy(node_logits, gt_target_nodes, reduction="sum")
    parent_label_loss = F.cross_entropy(
        parent_label_logits, gt_parent_labels, reduction="sum"
    )
    new_label_loss = F.cross_entropy(new_label_logits, gt_new_labels, reduction="sum")
    loss = (node_loss + parent_label_loss + new_label_loss) / batch_size

    return loss
