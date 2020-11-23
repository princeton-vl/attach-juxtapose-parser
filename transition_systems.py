"""
Implementation the attach-juxtapose transition system
"""
from copy import deepcopy
from typing import NamedTuple, Union, List, Tuple, Optional
from tree import (
    Tree,
    Label,
    InternalParseNode,
    LeafParseNode,
    DUMMY_LABEL,
)


class Action(NamedTuple):
    "An action in the attach-juxtatapose transition system"

    action_type: str
    target_node: int
    parent_label: Label
    new_label: Label

    def normalize(self) -> "Action":
        return Action(
            self.action_type,
            self.target_node,
            tuple(self.parent_label),
            tuple(self.new_label),
        )


class AttachJuxtapose:
    "The attach-juxtapose transition system"

    @staticmethod
    def oracle_actions(tree: Tree, immutable: bool) -> List[Action]:
        "Algorithm 1 in the appendix"

        if tree is None:
            return []
        else:
            if immutable:
                tree = deepcopy(tree)
            last_action, last_tree = AttachJuxtapose._undo_last_action(tree)
            return AttachJuxtapose.oracle_actions(last_tree, False) + [last_action]

    @staticmethod
    def execute(
        tree: Tree, action: Action, pos: int, tag: str, word: str, immutable: bool,
    ) -> InternalParseNode:
        "Sec. 3 in the paper"

        new_leaf = LeafParseNode(pos, tag, word)
        assert tree is None or tree.right == new_leaf.left
        if immutable:
            tree = deepcopy(tree)
        action_type, target_node_idx, parent_label, new_label = action

        # create the subtree to be inserted
        new_subtree: Union[
            InternalParseNode, LeafParseNode
        ] = new_leaf if parent_label == DUMMY_LABEL else InternalParseNode(
            parent_label, [new_leaf], None
        )

        # find the target position at which to insert the new subtree
        target_node = tree
        if target_node is not None:
            for _ in range(target_node_idx):
                next_node = target_node.children[-1]
                if not isinstance(next_node, InternalParseNode):  # truncate
                    break
                target_node = next_node

        if action_type == "attach":
            return AttachJuxtapose._execute_attach(tree, target_node, new_subtree)
        else:
            assert (
                action_type == "juxtapose"
                and target_node is not None
                and new_label != DUMMY_LABEL
            )
            return AttachJuxtapose._execute_juxtapose(
                tree, target_node, new_subtree, new_label
            )

    @staticmethod
    def actions2tree(
        word_seq: List[str], tag_seq: List[str], action_seq: List[Action]
    ) -> InternalParseNode:
        "Sec. 3 in the paper"

        tree = None
        for pos, (word, tag, action) in enumerate(zip(word_seq, tag_seq, action_seq)):
            tree = AttachJuxtapose.execute(
                tree, action, pos, tag, word, immutable=False
            )
        assert tree is not None
        return tree

    @staticmethod
    def _undo_last_action(tree: InternalParseNode) -> Tuple[Action, Tree]:
        "Algorithm 1 in the appendix"

        last_leaf = tree.last_leaf()
        siblings = last_leaf.get_siblings()

        last_subtree: Union[InternalParseNode, LeafParseNode]
        if siblings != []:
            last_subtree = last_leaf
            last_subtree_siblings = siblings
            parent_label: Tuple[str, ...] = DUMMY_LABEL
        else:
            assert last_leaf.parent is not None
            last_subtree = last_leaf.parent
            last_subtree_siblings = (
                last_subtree.get_siblings() if not last_subtree.is_root() else []
            )
            parent_label = last_subtree.label

        if last_subtree.is_root():
            return Action("attach", 0, parent_label, ()), None

        elif len(last_subtree_siblings) == 1 and isinstance(
            last_subtree_siblings[0], InternalParseNode
        ):
            assert last_subtree.parent is not None
            new_label = last_subtree.parent.label
            target_node = last_subtree_siblings[0]
            grand_parent = last_subtree.parent.parent
            if grand_parent is None:
                tree = target_node
                target_node.parent = None
            else:
                grand_parent.children = [
                    target_node if child is last_subtree.parent else child
                    for child in grand_parent.children
                ]
                target_node.parent = grand_parent
            return (
                Action("juxtapose", target_node.depth, parent_label, new_label),
                tree,
            )

        else:
            assert last_subtree.parent is not None
            target_node = last_subtree.parent
            target_node.children.remove(last_subtree)
            return Action("attach", target_node.depth, parent_label, ()), tree

    @staticmethod
    def _execute_attach(
        tree: Tree,
        target_node: Optional[InternalParseNode],
        new_subtree: Union[InternalParseNode, LeafParseNode],
    ) -> InternalParseNode:
        "Sec. 3 in the paper"

        if target_node is None:  # attach the first token
            assert (
                isinstance(new_subtree, InternalParseNode)
                and new_subtree.left == 0
                and new_subtree.right == 1
            )
            tree = new_subtree
        else:
            target_node.children.append(new_subtree)
            new_subtree.parent = target_node
            AttachJuxtapose._update_right(target_node)

        assert tree is not None
        return tree

    @staticmethod
    def _execute_juxtapose(
        tree: Tree,
        target_node: InternalParseNode,
        new_subtree: Union[InternalParseNode, LeafParseNode],
        new_label: Label,
    ) -> InternalParseNode:
        "Sec. 3 in the paper"

        parent = target_node.parent
        new_node = InternalParseNode(new_label, children=[target_node, new_subtree])
        if parent is None:
            assert tree is target_node
            tree = new_node
        else:
            assert target_node is parent.children[-1]
            parent.children = parent.children[:-1] + [new_node]
            new_node.parent = parent
            AttachJuxtapose._update_right(parent)

        assert tree is not None
        return tree

    @staticmethod
    def _update_right(node: InternalParseNode) -> None:
        "Update the 'right' attribute for all ancestors of 'node', including itself"

        cur: Optional[InternalParseNode] = node
        while cur is not None:
            cur.right = cur.children[-1].right
            cur = cur.parent
