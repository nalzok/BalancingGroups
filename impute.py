# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/usr/bin/env python

import os
import sys
import json
import time
import torch
import argparse
import numpy as np
import pandas as pd
import lightning as L
from pathlib import Path

import models
from datasets import get_loaders
from utils import Tee, chosen_hparams_best


datasets = {"waterbirds", "celeba", "chexpert-embedding", "coloredmnist", "multinli", "civilcomments"}


def parse_args():
    parser = argparse.ArgumentParser(description='Balancing baselines')
    parser.add_argument('--dataset', type=str, choices=datasets)
    parser.add_argument('--missing', type=float, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--slurm_output_dir', type=str, default='slurm_outputs')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--slurm_partition', type=str, default=None)
    parser.add_argument('--max_time', type=int, default=3*24*60)
    parser.add_argument('--hparams_seed', type=int, required=True)
    parser.add_argument('--init_seed', type=int, required=True)
    parser.add_argument('--selector', type=str, default='min_acc_va')
    return vars(parser.parse_args())


def run_experiment(args):
    L.seed_everything(args["init_seed"])
    start_time = time.time()
    dataset, loaders = get_loaders(args["data_path"], args["dataset"], args["batch_size"], "erm", None, missing=args["missing"])

    stem = "{}_missing{}_batch{}_lr{}_decay{}_seed_{}_{}".format(
        args["dataset"],
        args["missing"],
        args["batch_size"],
        args["lr"],
        args["weight_decay"],
        args["hparams_seed"],
        args["init_seed"],
    )
    sys.stdout = Tee(os.path.join(args["output_dir"], f"{stem}.out"), sys.stdout)
    sys.stderr = Tee(os.path.join(args["output_dir"], f"{stem}.err"), sys.stderr)

    model = models.ERM(args, loaders["tr"])

    for epoch in range(args["num_epochs"]):
        for i, x, y, g in loaders["tr"]:
            # HACK: swapping y and g because we want to impute g
            model.update(i, x, g, y, epoch)

        result = {"args": args, "epoch": epoch, "time": time.time() - start_time}
        for loader_name, loader in loaders.items():
            if loader_name == "va":
                avg_acc, corrects, totals, group_accs = model.accuracy(loader, predict_g=True)
                result["acc_" + loader_name] = group_accs
                result["corrects_" + loader_name] = corrects
                result["totals_" + loader_name] = totals
                result["avg_acc_" + loader_name] = avg_acc

        print(json.dumps(result))

    metadata = []
    for i, x, y, g in loaders["te"]:
        with torch.inference_mode():
            ghat = torch.softmax(model.predict(x.cuda()), -1)

        for ii, gghat in zip(i.tolist(), ghat[:, 1].tolist()):
            metadata.append((dataset.index[ii], gghat))

    metadata = pd.DataFrame.from_records(metadata, index="id", columns=["id", "a"])
    metadata_imputed = dataset.metadata_full.join(metadata, rsuffix="_impute")
    mask = np.isnan(metadata_imputed["a_impute"])
    metadata_imputed.loc[mask, "a_impute"] = metadata_imputed["a"][mask]
    metadata_imputed = metadata_imputed.rename(columns={"a_impute": "a", "a": "a_orig"})

    metadata_path = Path(dataset.metadata_path)
    metadata_path = metadata_path.with_stem(f"{metadata_path.stem}_impute{args['missing']}")
    metadata_imputed.to_csv(metadata_path)


if __name__ == "__main__":
    args = parse_args()

    commands = []
    torch.manual_seed(args["hparams_seed"])

    args["method"] = "erm"
    log_lr, log_wd, epoch, batch_size = chosen_hparams_best[args["dataset"]][args["method"]]
    args["lr"] = 10**log_lr
    args["weight_decay"] = 10**log_wd
    args["num_epochs"] = int(round(epoch))
    args["batch_size"] = batch_size

    commands.append(dict(args))

    os.makedirs(args["output_dir"], exist_ok=True)
    commands = [commands[int(p)] for p in torch.randperm(len(commands))]

    torch.set_float32_matmul_precision("high")
    for command in commands:
        run_experiment(command)
    
