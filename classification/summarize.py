"""Summarize the classification accuracies over the seeds.

For each fraction f and perturbation, reports the mean accuracy, the 95% confidence
interval (t distribution) and a one-tailed Welch t-test checking whether the accuracy
is smaller than the one obtained on the original dataset D.

Example:
    python summarize.py --results_dir results
    # Checkpoints produced by the original code of the dissertation
    python summarize.py --results_dir <path>/checkpoints --legacy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from perturbations import LEGACY_NAMES, PERTURBATIONS
from train_classification import FRACTIONS, SEEDS

DATASET_NAMES = {"none": "D", "i": "D_{-i}", "t": "D_{-t}", "it": "D_{-i,-t}"}


def load_accuracy(results_dir, perturbation, fraction, seed, legacy):

    if legacy:
        import torch
        perc = "p100" if fraction == 1.0 else f"p{fraction*100}"
        file = Path(results_dir) / f"{LEGACY_NAMES[perturbation]} {perc} seed_{seed}.pickle"
        checkpoint = torch.load(file, map_location="cpu", weights_only=False)
        return checkpoint["accTest"][-1]

    with open(Path(results_dir) / f"{perturbation}_f{int(fraction*100)}_seed{seed}.json") as f:
        return json.load(f)["accuracy"]


def summarize(results_dir, legacy=False):

    rows = []
    for fraction in FRACTIONS:
        accs = {p: np.array([load_accuracy(results_dir, p, fraction, s, legacy) for s in SEEDS])
                for p in PERTURBATIONS}
        for p, values in accs.items():
            mean = values.mean()
            ci = stats.t.ppf(0.975, len(values)-1)*stats.sem(values)
            if p == "none":
                p_value = np.nan
            else:
                p_value = stats.ttest_ind(values, accs["none"], equal_var=False, alternative="less").pvalue
            rows.append({"fraction": fraction, "dataset": DATASET_NAMES[p], "mean_accuracy": mean,
                         "ci95": ci, "p_value_vs_D": p_value, "n_seeds": len(values)})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--legacy", action="store_true", help="Read checkpoints saved by the original code")
    parser.add_argument("--output", default="results/accuracy_summary.csv")
    args = parser.parse_args()

    df = summarize(args.results_dir, args.legacy)
    print(df.to_string(index=False, float_format="%.2f"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
