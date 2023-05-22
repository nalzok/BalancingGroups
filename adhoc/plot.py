import argparse
from pathlib import Path
import glob
import re
import json
from collections import defaultdict
import statistics
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt


numerical_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
dataset_pattern = "(celeba|chexpert-embedding|civilcomments|coloredmnist|multinli|waterbirds)"
imputation_pattern = f"impute(None|{numerical_pattern})"
method_pattern = "(erm|suby|subg|rwy|rwg|dro|jtt|ttlsa|ttlsi)"
hyperparem_pattern = f"batch({numerical_pattern})_lr({numerical_pattern})_decay({numerical_pattern})"
seed_pattern = f"seed_({numerical_pattern})_({numerical_pattern})"
pattern = re.compile(f"^{dataset_pattern}_{imputation_pattern}_{method_pattern}_{hyperparem_pattern}_{seed_pattern}.out$")


def plot(args):
    everything = collect(args)

    plot_root = Path("figures")
    plot_root.mkdir(parents=True, exist_ok=True)
    datasets = ["chexpert-embedding", "coloredmnist"]
    methods = ["erm", "dro", "subg", "ttlsi", "ttlsa"]

    imputed_set = { imputed for _, imputed, _ in everything }
    for dataset in datasets:
        for imputed in imputed_set:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axvline(1/20, color="black", linestyle="dotted", linewidth=3)

            for method in methods:
                if (dataset, imputed, method) not in everything:
                    continue
                v = everything[(dataset, imputed, method)]

                x = np.linspace(0, 1, 21)
                mean = np.array([v[f"te-{split}"][0] for split in range(21)])
                std = np.array([v[f"te-{split}"][1] for split in range(21)])
                ax.errorbar(x, mean, std, label=method)

            plt.ylim((0.5, 1))
            plt.xlabel("Shift parameter")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy on {dataset}")
            plt.legend()
            plt.grid(True)
            fig.tight_layout()

            format_axes(ax)
            for suffix in ("png", "pdf"):
                plt.savefig(plot_root / f"{dataset}_imputed{imputed}.{suffix}", bbox_inches='tight', dpi=300)

            plt.close(fig)


def collect(args):
    root = Path(args.path)
    files = glob.glob(str(root / "*.out"))

    agg = defaultdict(lambda: defaultdict(lambda: []))
    for file in files:
        name = Path(file).name
        matching = pattern.fullmatch(name)
        assert matching is not None

        dataset, imputation, method, _, _, _, _, _ = matching.groups()
        imputation = 0 if imputation == "None" else float(imputation)
        key = dataset, imputation, method
        
        records = []
        with open(file) as f:
            for i, line in enumerate(f):
                if line.startswith("="):
                    assert i == 0
                    continue
                records.append(json.loads(line))

        optimal_epoch = 0
        optimal_acc_va = 0
        for epoch, record in enumerate(records):
            assert epoch == record["epoch"]

            corrects_va = record["corrects_va"]
            totals_va = record["totals_va"]
            if len(corrects_va) == len(totals_va) == 16:
                # for CivilComments, we convert fine to coarse for a fair comparison
                corrects_va = [corrects_va[0], sum(corrects_va[1:8]), corrects_va[8], sum(corrects_va[6:16])]
                totals_va = [totals_va[0], sum(totals_va[1:8]), totals_va[8], sum(totals_va[6:16])]

            if args.selector1 == "min":
                acc_va = min(1 if t == 0 else c/t for c, t in zip(corrects_va, totals_va))
            elif args.selector1 == "avg":
                acc_va = sum(corrects_va) / sum(totals_va)
            else:
                raise ValueError(f"Unknown selector '{args.selector1}'")

            if acc_va > optimal_acc_va:
                optimal_epoch = record["epoch"]
                optimal_acc_va = acc_va

        optimal_record = records[optimal_epoch]
        test_sets = set()
        for name in optimal_record.keys():
            if name.startswith("corrects_") or name.startswith("totals_"):
                _, loader_name = name.split("_")
                test_sets.add(loader_name)

        for test_set in test_sets:
            corrects, totals = optimal_record[f"corrects_{test_set}"], optimal_record[f"totals_{test_set}"]

            if len(corrects) == len(totals) == 16:
                # for CivilComments, we convert fine to coarse for a fair comparison
                corrects = [corrects[0], sum(corrects[1:8]), corrects[8], sum(corrects[6:16])]
                totals = [totals[0], sum(totals[1:8]), totals[8], sum(totals[6:16])]

            agg[key][test_set].append((corrects, totals))

    everything = {}
    for key, by_key in agg.items():
        aggregated = {}
        for loader_name, results in by_key.items():
            acc_list = []
            for corrects, totals in results:
                if args.selector2 == "min":
                    acc = min(1 if t == 0 else c/t for c, t in zip(corrects, totals))
                elif args.selector2 == "avg":
                    acc = sum(corrects) / sum(totals)
                else:
                    raise ValueError(f"Unknown selector '{args.selector2}'")
                acc_list.append(acc)

            acc_mean, acc_std = statistics.mean(acc_list), statistics.stdev(acc_list)
            std_err = acc_std / sqrt(len(acc_list))
            aggregated[loader_name] = acc_mean, std_err

        everything[key] = aggregated

    return everything


DEFAULT_WIDTH = 6.0
DEFAULT_HEIGHT = 1.5

# Font sizes
SIZE_SMALL = 10
SIZE_MEDIUM = 12
SIZE_LARGE = 16

SPINE_COLOR = 'gray'


def latexify(
    width_scale_factor=1,
    height_scale_factor=1,
    fig_width=None,
    fig_height=None,
):
    f"""
    width_scale_factor: float, DEFAULT_WIDTH will be divided by this number, DEFAULT_WIDTH is page width: {DEFAULT_WIDTH} inches.
    height_scale_factor: float, DEFAULT_HEIGHT will be divided by this number, DEFAULT_HEIGHT is {DEFAULT_HEIGHT} inches.
    fig_width: float, width of the figure in inches (if this is specified, width_scale_factor is ignored)
    fig_height: float, height of the figure in inches (if this is specified, height_scale_factor is ignored)
    """
    if fig_width is None:
        fig_width = DEFAULT_WIDTH / width_scale_factor
    if fig_height is None:
        fig_height = DEFAULT_HEIGHT / height_scale_factor

    # use TrueType fonts so they are embedded
    # https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib
    # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
    plt.rcParams["pdf.fonttype"] = 42

    # https://stackoverflow.com/a/39566040
    plt.rc("font", size=SIZE_MEDIUM)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_LARGE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_LARGE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SIZE_LARGE)  # legend fontsize
    plt.rc("figure", titlesize=SIZE_LARGE)  # fontsize of the figure title

    # latexify: https://nipunbatra.github.io/blog/posts/2014-06-02-latexify.html
    plt.rcParams["backend"] = "ps"
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("figure", figsize=(fig_width, fig_height))


def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate results')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--selector1', type=str, choices=['min', 'avg'])
    parser.add_argument('--selector2', type=str, choices=['min', 'avg'])
    args = parser.parse_args()

    latexify(width_scale_factor=2, fig_height=2)
    plot(args)