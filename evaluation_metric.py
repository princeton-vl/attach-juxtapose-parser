"""
A wrapper of Evalb for calculating evaluation metrics.
Taken from self-attentive-parser, see 
https://github.com/nikitakit/self-attentive-parser/blob/master/src/evaluate.py
"""
import re
import os
import subprocess
import tempfile
import math
from tree import InternalParseNode
from typing import Optional, List


class FScore(object):
    def __init__(
        self, recall: float, precision: float, fscore: float, complete_match: float
    ) -> None:
        self.recall = recall
        self.precision = precision
        self.fscore = fscore
        self.complete_match = complete_match

    def __str__(self) -> str:
        return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f}, CompleteMatch={:.2f})".format(
            self.recall, self.precision, self.fscore, self.complete_match
        )


def evalb(
    evalb_dir: str,
    gold_trees: List[InternalParseNode],
    predicted_trees: List[InternalParseNode],
    ref_gold_path: Optional[str] = None,
) -> FScore:
    if not os.path.exists(evalb_dir):
        raise FileNotFoundError(
            "Cannot locate %s. Please check if Evalb has been compiled." % evalb_dir
        )
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    evalb_spmrl_program_path = os.path.join(evalb_dir, "evalb_spmrl")
    assert os.path.exists(evalb_program_path) or os.path.exists(
        evalb_spmrl_program_path
    )

    if os.path.exists(evalb_program_path):
        evalb_param_path = os.path.join(evalb_dir, "nk.prm")
    else:
        evalb_program_path = evalb_spmrl_program_path
        evalb_param_path = os.path.join(evalb_dir, "spmrl.prm")

    assert os.path.exists(evalb_program_path)
    assert os.path.exists(evalb_param_path)

    assert len(gold_trees) == len(predicted_trees)
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        assert isinstance(gold_tree, InternalParseNode)
        assert isinstance(predicted_tree, InternalParseNode)
        gold_leaves = list(gold_tree.iter_leaves())
        predicted_leaves = list(predicted_tree.iter_leaves())
        assert len(gold_leaves) == len(predicted_leaves)
        assert all(
            gold_leaf.word == predicted_leaf.word
            for gold_leaf, predicted_leaf in zip(gold_leaves, predicted_leaves)
        )

    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    gold_path = os.path.join(temp_dir.name, "gold.txt")
    predicted_path = os.path.join(temp_dir.name, "predicted.txt")
    output_path = os.path.join(temp_dir.name, "output.txt")

    with open(gold_path, "w") as outfile:
        if ref_gold_path is None:
            for tree in gold_trees:
                outfile.write("{}\n".format(tree.linearize()))
        else:
            # For the SPMRL dataset our data loader performs some modifications
            # (like stripping morphological features), so we compare to the
            # raw gold file to be certain that we haven't spoiled the evaluation
            # in some way.
            with open(ref_gold_path) as goldfile:
                outfile.write(goldfile.read())

    with open(predicted_path, "w") as outfile:
        for tree in predicted_trees:
            outfile.write("{}\n".format(tree.linearize()))

    command = "{} -p {} {} {} > {}".format(
        evalb_program_path, evalb_param_path, gold_path, predicted_path, output_path,
    )
    subprocess.run(command, shell=True)

    fscore = FScore(math.nan, math.nan, math.nan, math.nan)
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.fscore = float(match.group(1))
            match = re.match(r"Complete match\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.complete_match = float(match.group(1))
                break

    success = (
        not math.isnan(fscore.fscore) or fscore.recall == 0.0 or fscore.precision == 0.0
    )

    if success:
        temp_dir.cleanup()
    else:
        print("Error reading EVALB results.")
        print("Gold path: {}".format(gold_path))
        print("Predicted path: {}".format(predicted_path))
        print("Output path: {}".format(output_path))

    return fscore
