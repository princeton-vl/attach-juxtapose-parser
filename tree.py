"""
Parse trees
"""
from lark import Lark, Transformer
from lark.lexer import Token
from typing import Optional, Callable, List, Generator, Any, Tuple, Sequence


class ParseNode:
    "Base class for nodes"

    # a node corresponds to tokens [left, right) in the sentence
    left: int
    right: int
    parent: Optional["InternalParseNode"]

    def is_root(self) -> bool:
        return self.parent is None

    def linearize(self) -> str:
        "Convert to S-expression format used for evaluation"
        raise NotImplementedError

    def iter_leaves(self) -> Generator["LeafParseNode", None, None]:
        "Iterate through all leaf nodes"
        raise NotImplementedError

    def traverse_preorder(self, callback: Callable[["ParseNode"], None]) -> None:
        "pre-order traversal"
        raise NotImplementedError

    def traverse_post(self, callback: Callable[["ParseNode"], None]) -> None:
        "post-orer traversal"
        raise NotImplementedError

    def get_siblings(self) -> List["ParseNode"]:
        "Get all its siblings"
        assert self.parent is not None
        return [sib for sib in self.parent.children if sib is not self]

    @property
    def depth(self) -> int:
        "Get the depth of a ndoe"
        d = -1
        cur: Optional[ParseNode] = self
        while cur is not None:
            cur = cur.parent
            d += 1
        return d


# labels on a unary chain are collapsed into one tuple
# e.g., ("S, "VP")
Label = Tuple[str, ...]

# No label, denoted as "None" in the paper
DUMMY_LABEL = ()


class InternalParseNode(ParseNode):
    "Internal nodes"

    label: Label
    children: List[ParseNode]

    def __init__(
        self,
        label: Label,
        children: List[ParseNode],
        parent: Optional["InternalParseNode"] = None,
    ) -> None:

        self.label = label
        self.children = children

        # [left, right)
        self.left = children[0].left
        self.right = children[-1].right

        self.parent = parent
        for child in self.children:
            child.parent = self

    def linearize(self) -> str:
        return (
            " ".join(["(" + sublabel for sublabel in self.label])
            + " "
            + " ".join(child.linearize() for child in self.children)
            + ")" * len(self.label)
        )

    def iter_leaves(self) -> Generator["LeafParseNode", None, None]:
        for child in self.children:
            yield from child.iter_leaves()

    def traverse_preorder(self, callback: Callable[[ParseNode], Any]) -> None:
        callback(self)
        for c in self.children:
            c.traverse_preorder(callback)

    def traverse_post(self, callback: Callable[[ParseNode], Any]) -> None:
        for c in self.children:
            c.traverse_post(callback)
        callback(self)

    def last_leaf(self) -> "LeafParseNode":
        "Get the rightmost leaf node in the subtree"

        cur: ParseNode = self
        while isinstance(cur, InternalParseNode):
            cur = cur.children[-1]
        assert isinstance(cur, LeafParseNode)
        return cur

    def iter_rightmost_chain(self) -> Generator["InternalParseNode", None, None]:
        "Iterate through the rightmost chain of the subtree"

        cur = self
        while isinstance(cur, InternalParseNode):
            yield cur
            if isinstance(cur.children[-1], InternalParseNode):
                cur = cur.children[-1]
            else:
                break

    def node_on_rightmost_chain(self, i: int) -> "InternalParseNode":
        "Get the ith node on the rightmost chain"
        for j, node in enumerate(self.iter_rightmost_chain()):
            if j == i:
                return node
        import pdb

        pdb.set_trace()
        raise IndexError

    def is_well_formed(self) -> bool:
        good = True

        def check_well_formed(node: ParseNode) -> None:
            nonlocal good
            good &= node.is_root() or isinstance(node.parent, InternalParseNode)
            if isinstance(node, LeafParseNode):
                good &= (
                    isinstance(node.left, int)
                    and isinstance(node.right, int)
                    and 0 <= node.left
                    and node.right == node.left + 1
                )
                good &= isinstance(node.tag, str)
                good &= isinstance(node.word, str)
            else:
                assert isinstance(node, InternalParseNode)
                good &= (
                    isinstance(node.label, tuple)
                    and node.label is not None
                    and all(isinstance(sublabel, str) for sublabel in node.label)
                )
                good &= isinstance(node.children, list)
                good &= len(node.children) > 1 or isinstance(
                    node.children[0], LeafParseNode
                )
                for child in node.children:
                    good &= isinstance(child, ParseNode)
                    good &= child.parent is node
                good &= all(
                    left.right == right.left
                    for left, right in zip(node.children, node.children[1:])
                )

        self.traverse_preorder(check_well_formed)

        return good


# None means empty tree
Tree = Optional[InternalParseNode]


def rightmost_chain_length(tree: Tree) -> int:
    if tree is None:
        return 0
    else:
        return len(list(tree.iter_rightmost_chain()))


class LeafParseNode(ParseNode):
    "Leaf nodes"

    # POS tag
    tag: str
    # token
    word: str

    def __init__(
        self, pos: int, tag: str, word: str, parent: Optional[InternalParseNode] = None
    ) -> None:
        self.left = pos
        self.right = pos + 1
        self.tag = tag
        self.word = word
        self.parent = parent

    def linearize(self) -> str:
        return "({} {})".format(self.tag, self.word)

    def iter_leaves(self) -> Generator["LeafParseNode", None, None]:
        yield self

    def traverse_preorder(self, callback: Callable[["ParseNode"], None]) -> None:
        callback(self)

    def traverse_post(self, callback: Callable[["ParseNode"], None]) -> None:
        callback(self)


class TreeBuilder(Transformer):  # type: ignore
    "Construct parse trees from S-expressions used in the datasets"

    pos: int = 0

    def sexp(self, children: List[Any]) -> Any:
        children = [c for c in children if c is not None]
        if len(children) <= 1:
            return None

        if len(children) == 2 and isinstance(children[1], Token):
            if children[0].value == "-NONE-":
                return None
            leaf = LeafParseNode(
                pos=self.pos, tag=children[0].value, word=children[1].value, parent=None
            )
            self.pos += 1
            return leaf

        elif children[0].value == "TOP":
            # remove TOP
            self.pos = 0
            return children[1]

        else:
            sublabel = children[0].value
            if len(children) == 2 and isinstance(
                children[1], InternalParseNode
            ):  # unary chain
                parent = children[1]
                parent.label = (sublabel,) + parent.label
            else:
                parent = InternalParseNode(
                    label=(sublabel,), children=children[1:], parent=None
                )
            return parent


class TreeBatch:
    trees: Sequence[Tree]

    def __init__(self, trees: Sequence[Tree]) -> None:
        self.trees = trees

    @staticmethod
    def from_file(filename: str) -> "TreeBatch":
        "Read a sequence of parse trees from a file"

        sexp_ebnf = """
            sexp : "(" SYMBOL (sexp|SYMBOL)* ")"
            SYMBOL : /[^\(\)\s]+/
            %import common.WS
            %ignore WS
        """
        sexp_parser = Lark(sexp_ebnf, start="sexp", parser="lalr")
        tree_builder = TreeBuilder()
        trees: List[Tree] = [
            tree_builder.transform(sexp_parser.parse(tree_sexp.strip()))
            for tree_sexp in open(filename)
        ]
        return TreeBatch(trees)

    @staticmethod
    def empty_trees(n: int) -> "TreeBatch":
        return TreeBatch([None for _ in range(n)])

    def __getitem__(self, idx: int) -> Tree:
        return self.trees[idx]

    def __len__(self) -> int:
        return len(self.trees)

    def traverse_preorder(self, callback: Callable[[ParseNode], None]) -> None:
        for tree in self.trees:
            if tree is not None:
                tree.traverse_preorder(callback)


if __name__ == "__main__":
    trees_train = TreeBatch.from_file("data/02-21.10way.clean")
    trees_valid = TreeBatch.from_file("data/22.auto.clean")
    trees_test = TreeBatch.from_file("data/23.auto.clean")
