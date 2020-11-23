"""
Decode token representation to actions
"""
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch_geometric
import torch_scatter
from .graph_nn import GraphNeuralNetwork
from .utils import BatchIndices
from tree import InternalParseNode, DUMMY_LABEL, Label
from omegaconf import DictConfig
from env import State
from transition_systems import Action
from typing import Tuple, List, Any, Dict, Optional


def decode_actions(
    node_logits: torch.Tensor,
    parent_label_logits: torch.Tensor,
    new_label_logits: torch.Tensor,
    label_vocab: List[Label],
) -> List[Any]:
    """
    Decode the most likely actions from action logits for the attach-juxtapose transition system
    """
    batch_size = node_logits.size(0)

    node_idx = node_logits.argmax(dim=-1)
    parent_label_idx = parent_label_logits.argmax(dim=-1)
    new_label_idx = new_label_logits.argmax(dim=-1)

    if node_logits.dim() == 2:
        # decode actions for only the current time step
        return [
            Action(
                "attach" if label_vocab[new_label] == DUMMY_LABEL else "juxtapose",
                target_node,
                label_vocab[parent_label],
                label_vocab[new_label],
            )
            for target_node, parent_label, new_label in zip(
                node_idx.tolist(), parent_label_idx.tolist(), new_label_idx.tolist()
            )
        ]
    else:
        # decode actions for all time steps
        max_len = node_idx.size(1)
        return [
            [
                Action(
                    "attach"
                    if label_vocab[new_label_idx[i, j].item()] == DUMMY_LABEL  # type: ignore
                    else "juxtapose",
                    target_node=node_idx[i, j].item(),  # type: ignore
                    parent_label=label_vocab[parent_label_idx[i, j]],  # type: ignore
                    new_label=label_vocab[new_label_idx[i, j]],  # type: ignore
                )
                for j in range(max_len)
            ]
            for i in range(batch_size)
        ]


def decode_topk_actions(
    node_logit: torch.Tensor,
    parent_label_logit: torch.Tensor,
    new_label_logit: torch.Tensor,
    label_vocab: List[Label],
    k: int,
) -> Tuple[List[Action], torch.Tensor]:
    """
    Decode top-k actions from action logits
    """

    node_distr = Categorical(logits=node_logit)  # type: ignore
    parent_label_distr = Categorical(logits=parent_label_logit)  # type: ignore
    new_label_distr = Categorical(logits=new_label_logit)  # type: ignore

    node_log_prob = node_distr.probs.log()
    parent_label_log_prob = parent_label_distr.probs.log()
    new_label_log_prob = new_label_distr.probs.log()

    # sum of log probabilities for target_node, parent_label, and new_label
    batch_size = node_logit.size(0)
    first_dim_stride = parent_label_log_prob.size(-1) * new_label_log_prob.size(-1)
    second_dim_stride = new_label_log_prob.size(-1)
    actions: List[List[Action]] = []
    log_probs = []

    for i in range(batch_size):
        actions.append([])
        actions_log_prob = torch.cartesian_prod(  # type: ignore
            node_log_prob[i], parent_label_log_prob[i], new_label_log_prob[i]
        ).sum(dim=-1)
        top_log_probs, top_idxs = actions_log_prob.topk(k)
        log_probs.append(top_log_probs)
        for idx in top_idxs:
            idx = idx.item()
            node_idx = idx // first_dim_stride
            idx %= first_dim_stride
            parent_label_idx = idx // second_dim_stride
            new_label_idx = idx % second_dim_stride
            actions[-1].append(
                Action(
                    "attach"
                    if label_vocab[new_label_idx] == DUMMY_LABEL
                    else "juxtapose",
                    target_node=node_idx,
                    parent_label=label_vocab[parent_label_idx],
                    new_label=label_vocab[new_label_idx],
                )
            )

    log_probs = torch.stack(log_probs)  # type: ignore

    return actions, log_probs  # type: ignore


class SequenceDecoder(nn.Module):
    "A two-layer neural network predicting the action based on the current token feature"

    def __init__(self, vocabs: Dict[str, Any], cfg: DictConfig) -> None:
        super().__init__()  # type: ignore
        self.cfg = cfg
        self.label_vocab = vocabs["label"]

        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_decoder),
            nn.LayerNorm([cfg.d_decoder]),
            nn.ReLU(),
            nn.Linear(cfg.d_decoder, cfg.max_sentence_len + 2 * len(self.label_vocab),),
            # target_node in Action cannot exceed max_sentence_len
        )

    def forward(self, state: State) -> Tuple[List[Action], Dict[str, torch.Tensor]]:
        tokens_emb, tokens_word = state.tokens_emb, state.tokens_word
        batch_size, max_len, d_model = tokens_emb.size()

        flattened_tokens_emb = tokens_emb.view(-1, d_model)
        all_logits = self.mlp(flattened_tokens_emb).view(batch_size, max_len, -1)

        node_logits = all_logits[:, :, : self.cfg.max_sentence_len]
        num_labels = len(self.label_vocab)
        parent_label_logits = all_logits[
            :, :, self.cfg.max_sentence_len : (self.cfg.max_sentence_len + num_labels),
        ]
        new_label_logits = all_logits[
            :, :, (self.cfg.max_sentence_len + num_labels) :,
        ]

        # when in the initial state, parent_label must NOT be () and new_label must be ()
        parent_label_logits[:, 0, 0] = float("-inf")
        new_label_logits[:, 0, 1:] = float("-inf")

        actions = decode_actions(
            node_logits, parent_label_logits, new_label_logits, self.label_vocab,
        )

        # truncate action sequences to match sentence lengths
        actions = [
            action_seq[: len(words)] for action_seq, words in zip(actions, tokens_word)
        ]

        logits = {
            "target_node": node_logits,
            "parent_label": parent_label_logits,
            "new_label": new_label_logits,
        }

        return actions, logits


class GraphDecoder(nn.Module):
    """
    GCN-based decoder that maps a state (consisting of the partial tree and the next token) to an action
    """

    cfg: DictConfig
    position_table: torch.Tensor
    label_vocab: List[Label]
    label_embedding: nn.Module
    graph_embedding: nn.Module
    action_decoder: nn.Module
    dummy_node_feature: nn.Parameter

    def __init__(
        self, position_table: torch.Tensor, vocabs: Dict[str, Any], cfg: DictConfig
    ):
        super().__init__()  # type: ignore
        self.position_table = position_table
        self.label_idx_map = {label: i for i, label in enumerate(vocabs["label"])}
        self.cfg = cfg

        # embeddings of constituency labels used when initializing node embeddings in the graph
        self.label_embedding = nn.Embedding(len(vocabs["label"]), cfg.d_model // 2)

        # GCN layers
        self.graph_embedding = GraphNeuralNetwork(cfg.d_model, cfg.num_gcn_layers)

        # decode the action based on embeddings of the new token and rightmost chain
        self.action_decoder = ActionDecoder(cfg.d_model, vocabs["label"])

        # dummy feature used when the rightmost chain is empty
        self.dummy_node_feature = nn.Parameter(torch.empty(cfg.d_model))
        nn.init.normal_(self.dummy_node_feature)

    def forward(self, state: State, topk: Optional[int]) -> Tuple[Any, Any]:
        graph, is_initial_state, next_token_features = self.construct_graph(state)
        rightmost_chain_features = self.graph_embedding(graph, graph.on_rightmost_chain)
        actions, logits = self.action_decoder(
            rightmost_chain_features,
            next_token_features,
            graph.batch,
            graph.on_rightmost_chain,
            is_initial_state,
            topk,
        )
        return actions, logits

    def construct_graph(
        self, state: State
    ) -> Tuple[torch_geometric.data.Batch, bool, torch.Tensor]:
        """
        Construct a batched graph object from the state
        """
        partial_trees, tokens_emb, next_token_pos, batch_size = (
            state.partial_trees,
            state.tokens_emb,
            state.n_step,
            state.batch_size,
        )
        next_token_features = state.tokens_emb[:, next_token_pos]
        device = tokens_emb.device

        if next_token_pos == 0:  # the first step
            return (
                self.construct_init_graph(batch_size, device),
                True,
                next_token_features,
            )

        node_token_pos = []  # positions of tokens in sentences, -1 if not token
        node_label_idx = []  # labels of internal nodes, -1 if not internal node
        node_label_left = []
        node_label_right = []
        node_batch_idx = []
        edges: List[Tuple[int, int]] = []
        on_rightmost_chain: List[bool] = []

        num_nodes = 0
        for i, tree in enumerate(partial_trees):
            assert isinstance(tree, InternalParseNode)
            x, edge_index, rightmost_chain_i = self.tree2graph_preorder(
                tree, num_nodes, device
            )
            edges.extend(edge_index)
            on_rightmost_chain.extend(rightmost_chain_i)
            for node in x:
                if "label" in node:  # internal node
                    node_token_pos.append(-1)
                    node_label_idx.append(self.label_idx_map[node["label"]])
                    node_label_left.append(node["left"])
                    node_label_right.append(node["right"])
                else:  # leaf node
                    node_token_pos.append(node["token_pos"])
                    node_label_idx.append(-1)
                    node_label_left.append(-1)
                    node_label_right.append(-1)
            tree_size = len(x)
            node_batch_idx.extend([i] * tree_size)
            num_nodes += tree_size

        node_token_pos = torch.tensor(node_token_pos, device=device)  # type: ignore
        node_is_token = node_token_pos >= 0  # type: ignore
        node_label_idx = node_token_pos.new_tensor(node_label_idx)  # type: ignore
        node_label_left = node_token_pos.new_tensor(node_label_left)  # type: ignore
        node_label_right = node_token_pos.new_tensor(node_label_right)  # type: ignore
        node_is_label = node_label_idx >= 0  # type: ignore
        node_batch_idx = node_token_pos.new_tensor(node_batch_idx)  # type: ignore

        d_model = self.cfg.d_model
        node_emb = tokens_emb.new_zeros((len(node_token_pos), d_model))
        flattened_tokens_emb = tokens_emb.view(-1, d_model)
        node_emb[node_is_token] = torch.index_select(
            flattened_tokens_emb,
            0,
            node_batch_idx[node_is_token] * tokens_emb.size(1)
            + node_token_pos[node_is_token],
        )
        label_position_emb = (
            self.position_table[node_label_left[node_is_label]]
            + self.position_table[node_label_right[node_is_label]]
        )
        node_emb[node_is_label] = torch.cat(
            [self.label_embedding(node_label_idx[node_is_label]), label_position_emb],
            dim=-1,
        )

        all_edges_index = (
            torch.tensor(edges, device=device).t()
            if edges != []
            else torch.empty(2, 0, dtype=torch.int64, device=device)
        )
        graph = torch_geometric.data.Batch(
            batch=node_batch_idx, x=node_emb, edge_index=all_edges_index,
        )

        graph.on_rightmost_chain = torch.tensor(
            on_rightmost_chain, dtype=torch.bool, device=device
        )

        return graph, False, next_token_features

    def tree2graph_preorder(
        self, tree: InternalParseNode, base_idx: int, device: torch.device
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]], List[bool]]:
        rank = 0
        x = []
        edge_index = []

        def visitor(node: Any) -> None:
            nonlocal rank
            node.rank = rank
            rank += 1

            # edges between parents and children
            if node.parent is not None:
                edge_index.append((base_idx + node.parent.rank, base_idx + node.rank))
                edge_index.append((base_idx + node.rank, base_idx + node.parent.rank))

            if isinstance(node, InternalParseNode):
                x.append({"label": node.label, "left": node.left, "right": node.right})
            else:
                x.append({"token_pos": node.left})

        tree.traverse_preorder(visitor)

        on_rightmost_chain = [False] * len(x)
        assert isinstance(tree, InternalParseNode)
        for node in tree.iter_rightmost_chain():
            on_rightmost_chain[node.rank] = True  # type: ignore

        return x, edge_index, on_rightmost_chain

    def construct_init_graph(
        self, batch_size: int, device: torch.device
    ) -> torch_geometric.data.Batch:
        """
        Construct the graph for a batch of empty trees
        """
        empty_edge = torch.empty(2, 0, dtype=torch.int64, device=device)
        graph = torch_geometric.data.Batch.from_data_list(
            [
                torch_geometric.data.Data(
                    self.dummy_node_feature.unsqueeze(0), empty_edge
                )
                for _ in range(batch_size)
            ]
        ).to(device)
        graph.on_rightmost_chain = torch.ones(
            batch_size, dtype=torch.bool, device=device
        )
        return graph


class ActionDecoder(nn.Module):
    """
    Decode actions given embeddings of the new token and the rightmost chain
    """

    label_vocab: List[Label]
    attn_layers_c: nn.Module
    attn_layers_p: nn.Module
    labels_layers: nn.Module

    def __init__(self, d_model: int, label_vocab: List[Label]) -> None:
        super().__init__()  # type: ignore
        self.label_vocab = label_vocab
        self.num_labels = len(label_vocab)

        self.attn_layers_c = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm([d_model // 4]),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )
        self.attn_layers_p = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm([d_model // 4]),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )
        self.labels_layers = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),
            nn.LayerNorm([d_model // 2]),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2 * len(label_vocab)),
        )

    def forward(
        self,
        rightmost_chain_features: torch.Tensor,
        next_token_features: torch.Tensor,
        batch: torch.Tensor,
        on_rightmost_chain: torch.Tensor,
        is_initial_state: bool,
        topk: Optional[int],  # decode top-k actions for beam search
    ) -> Tuple[List[Action], Any]:

        d_model = next_token_features.size(1)

        rightmost_chain_batch = batch[on_rightmost_chain]
        extended_next_token_featuress = torch.gather(
            next_token_features,
            0,
            rightmost_chain_batch.unsqueeze(1).expand(-1, d_model),
        )  # the corresponding next_token_features for each node on the rightmost chain

        d_content = d_model // 2
        attn_features_c = torch.cat(
            [
                extended_next_token_featuress[:, :d_content],
                rightmost_chain_features[:, :d_content],
            ],
            dim=-1,
        )
        attn_features_p = torch.cat(
            [
                extended_next_token_featuress[:, d_content:],
                rightmost_chain_features[:, d_content:],
            ],
            dim=-1,
        )

        # attention weights for nodes on the rightmost chain
        node_attn = self.attn_layers_c(attn_features_c) + self.attn_layers_p(
            attn_features_p
        )
        weighted_rightmost_chain_features = torch_scatter.scatter_add(
            rightmost_chain_features * torch.sigmoid(node_attn),
            rightmost_chain_batch,
            dim=0,
        )  # batch_size x d_model

        labels_features = torch.cat(
            [next_token_features, weighted_rightmost_chain_features], dim=-1
        )
        labels_logits = self.labels_layers(labels_features)

        parent_label_logits = labels_logits[
            :, : self.num_labels
        ]  # batch_size x num_labels
        new_label_logits = labels_logits[
            :, self.num_labels :
        ]  # batch_size x num_labels
        if is_initial_state:
            # when in the initial state, parent_label must NOT be () and new_label must be ()
            parent_label_logits[:, 0] = float("-inf")
            new_label_logits[:, 1:] = float("-inf")

        batch_idxs = BatchIndices(rightmost_chain_batch)
        node_logits = batch_idxs.inflate(node_attn, fill_value=float("-inf"))[:, :, 0]

        if topk is None:
            # decode only the most likely action
            actions = decode_actions(
                node_logits, parent_label_logits, new_label_logits, self.label_vocab,
            )
            logits = {
                "target_node": node_logits,
                "parent_label": parent_label_logits,
                "new_label": new_label_logits,
            }
            return actions, logits

        else:
            # decode top-k actions for beam search
            return decode_topk_actions(
                node_logits,
                parent_label_logits,
                new_label_logits,
                self.label_vocab,
                topk,
            )
