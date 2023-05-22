import argparse
from pathlib import Path
import glob
import re
import json


numerical_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
dataset_pattern = "(celeba|chexpert-embedding|civilcomments|coloredmnist|multinli|waterbirds)"
missing_pattern = f"missing({numerical_pattern})"
hyperparem_pattern = f"batch({numerical_pattern})_lr({numerical_pattern})_decay({numerical_pattern})"
seed_pattern = f"seed_({numerical_pattern})_({numerical_pattern})"
pattern = re.compile(f"^{dataset_pattern}_{missing_pattern}_{hyperparem_pattern}_{seed_pattern}.out$")


def aggregate(args):
    root = Path(args.path)
    files = glob.glob(str(root / "*.out"))

    everything = {}
    for file in files:
        name = Path(file).name
        matching = pattern.fullmatch(name)
        assert matching is not None

        dataset, missing, _, _, _, _, _ = matching.groups()
        missing = float(missing)
        
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

            if args.selector == "min":
                acc_va = min(1 if t == 0 else c/t for c, t in zip(corrects_va, totals_va))
            elif args.selector == "avg":
                acc_va = sum(corrects_va) / sum(totals_va)
            else:
                raise ValueError(f"Unknown selector '{args.selector}'")

            if acc_va > optimal_acc_va:
                optimal_epoch = record["epoch"]
                optimal_acc_va = acc_va

        everything[dataset, missing] = optimal_epoch


    for k in sorted(everything):
        print(k, everything[k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate results')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--selector', type=str, choices=['min', 'avg'])
    args = parser.parse_args()
    aggregate(args)
