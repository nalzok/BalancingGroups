import argparse
from pathlib import Path
import glob
import re
import json
from collections import defaultdict
import statistics


numerical_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
dataset_pattern = "(celeba|chexpert-embedding|civilcomments|coloredmnist|multinli|waterbirds)"
imputation_pattern = f"impute(None|{numerical_pattern})"
method_pattern = "(erm|suby|subg|rwy|rwg|dro|jtt|ttlsa|ttlsi)"
hyperparem_pattern = f"batch({numerical_pattern})_lr({numerical_pattern})_decay({numerical_pattern})"
seed_pattern = f"seed_({numerical_pattern})_({numerical_pattern})"
pattern = re.compile(f"^{dataset_pattern}_{imputation_pattern}_{method_pattern}_{hyperparem_pattern}_{seed_pattern}.out$")


def aggregate(args):
    root = Path(args.path)
    files = glob.glob(str(root / "*.out"))

    agg = defaultdict(lambda: defaultdict(lambda: []))
    for file in files:
        name = Path(file).name
        matching = pattern.fullmatch(name)
        assert matching is not None

        dataset, imputation, method, batch_size, lr, wd, _, _ = matching.groups()
        imputation = 0 if imputation == "None" else float(imputation)
        batch_size = int(batch_size)
        lr = float(lr)
        wd = float(wd)
        key = dataset, imputation, method, batch_size, lr, wd
        
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

        if args.split is not None:
            test_sets = { args.split }

        for test_set in test_sets:
            corrects, totals = optimal_record[f"corrects_{test_set}"], optimal_record[f"totals_{test_set}"]

            if len(corrects) == len(totals) == 16:
                # for CivilComments, we convert fine to coarse for a fair comparison
                corrects = [corrects[0], sum(corrects[1:8]), corrects[8], sum(corrects[6:16])]
                totals = [totals[0], sum(totals[1:8]), totals[8], sum(totals[6:16])]

            agg[key][test_set].append((corrects, totals))

    imputed_set = set()
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
            aggregated[loader_name] = f"{acc_mean*100:.2f} / {acc_std*100:.2f}"

        dataset, imputed, method, _, _, _ = key
        imputed_set.add(imputed)
        everything[(dataset, imputed, method)] = aggregated

    datasets = ["celeba", "waterbirds", "multinli", "civilcomments"]
    methods = ["erm", "ttlsi", "dro", "subg", "ttlsa"]

    for dataset in datasets:
        for imputed in imputed_set:
            print(dataset, imputed)
            for method in methods:
                v = everything[(dataset, imputed, method)]
                out = v[args.split] if args.split is not None else v
                print("&", method, "&", out, "\\\\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate results')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--selector1', type=str, choices=['min', 'avg'])
    parser.add_argument('--selector2', type=str, choices=['min', 'avg'])
    parser.add_argument('--split', type=str, default=None)
    args = parser.parse_args()
    aggregate(args)
