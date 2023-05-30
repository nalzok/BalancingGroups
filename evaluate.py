# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/usr/bin/env python

import os
import sys
import json
import time
import torch
import submitit
import argparse
import lightning as L

import models
from datasets import get_loaders
from utils import Tee, randl, chosen_hparams_best


datasets = {"waterbirds", "celeba", "chexpert-embedding", "coloredmnist", "multinli", "civilcomments"}
methods = {"erm", "suby", "subg", "rwy", "rwg", "dro", "jtt", "ttlsi", "ttlsa", "ttlsa-oracle", "ttlsa-batch-oracle"}


def parse_args():
    parser = argparse.ArgumentParser(description='Balancing baselines')
    parser.add_argument('--dataset', type=str, choices=datasets)
    parser.add_argument('--imputed', type=float, default=None)
    parser.add_argument('--method', type=str, choices=methods)
    parser.add_argument('--input_dir', type=str, default='inputs')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--slurm_output_dir', type=str, default='slurm_outputs')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--slurm_partition', type=str, default=None)
    parser.add_argument('--max_time', type=int, default=3*24*60)
    parser.add_argument('--hparams_seed', type=int, required=True)
    parser.add_argument('--init_seed', type=int, required=True)
    parser.add_argument('--selector', type=str, choices=['min', 'avg'])
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    return vars(parser.parse_args())


def run_experiment(args):
    L.seed_everything(args["init_seed"])
    start_time = time.time()

    _, loaders = get_loaders(args["data_path"], args["dataset"], args["batch_size"], args["method"], imputed=args["imputed"])

    stem = "{}_impute{}_{}_batch{}_lr{}_decay{}_seed_{}_{}".format(
        args["dataset"],
        args["imputed"],
        args["method"],
        args["batch_size"],
        args["lr"],
        args["weight_decay"],
        args["hparams_seed"],
        args["init_seed"],
    )
    selector = args["selector"]
    sys.stdout = Tee(os.path.join(args["output_dir"], f"{stem}_{selector}.out"), sys.stdout)
    sys.stderr = Tee(os.path.join(args["output_dir"], f"{stem}_{selector}.err"), sys.stderr)

    model = {
        "erm": models.ERM,
        "suby": models.ERM,
        "subg": models.ERM,
        "rwy": models.ERM,
        "rwg": models.ERM,
        "dro": models.GroupDRO,
        "jtt": models.JTT,
        "ttlsi": models.TTLSI,
        "ttlsa": models.TTLSA,
        "ttlsa-oracle": models.Oracle,
        "ttlsa-batch-oracle": models.BatchOracle,
    }[args["method"]](args, loaders["tr"])

    records = []
    file = os.path.join(args["input_dir"], f"{stem}.out")
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

        if selector == "min":
            acc_va = min(1 if t == 0 else c/t for c, t in zip(corrects_va, totals_va))
        elif selector == "avg":
            acc_va = sum(corrects_va) / sum(totals_va)
        else:
            raise ValueError(f"Unknown selector '{args.selector1}'")

        if acc_va > optimal_acc_va:
            optimal_epoch = record["epoch"]
            optimal_acc_va = acc_va

    ckpt_path = f"ckpt-final/{stem}_epoch{optimal_epoch}.pt"
    model.load(ckpt_path)
    result = {
        "args": args,
        "epoch": optimal_epoch,
        "time": time.time() - start_time,
        "loss_tr": 0
    }
    for loader_name, loader in loaders.items():
        auc, avg_acc, corrects, totals, group_accs = model.accuracy(loader)
        result["auc_" + loader_name] = auc
        result["avg_acc_" + loader_name] = avg_acc
        result["corrects_" + loader_name] = corrects
        result["totals_" + loader_name] = totals
        result["acc_" + loader_name] = group_accs

    print(json.dumps(result))


if __name__ == "__main__":
    args = parse_args()

    commands = []
    torch.manual_seed(args["hparams_seed"])

    args["num_epochs"] = {
        "waterbirds": 300 + 60,
        "celeba": 50 + 10,
        "chexpert-embedding": 300 + 60,
        "coloredmnist": 300 + 60,
        "multinli": 5 + 2,
        "civilcomments": 5 + 2
    }[args["dataset"]]

    log_lr, log_wd, _, batch_size = chosen_hparams_best[args["dataset"]][args["method"]]
    if args["batch_size"] is None:
        args["batch_size"] = batch_size
    if args["lr"] is None:
        args["lr"] = 10**log_lr
    if args["weight_decay"] is None:
        args["weight_decay"] = 10**log_wd

    # Group DRO
    args["eta"] = 0.1

    # JTT
    args["up"] = randl([4, 5, 6, 20, 50, 100])
    args["T"] = {
        "waterbirds": randl([40, 50, 60]),
        "celeba": randl([1, 5, 10]),
        "chexpert-embedding": randl([40, 50, 60]),
        "coloredmnist": randl([40, 50, 60]),
        "multinli": randl([1, 2]),
        "civilcomments": randl([1, 2])
    }[args["dataset"]]

    commands.append(dict(args))

    os.makedirs(args["output_dir"], exist_ok=True)
    commands = [commands[int(p)] for p in torch.randperm(len(commands))]

    if args['slurm_partition'] is not None:
        executor = submitit.SlurmExecutor(folder=args['slurm_output_dir'])
        executor.update_parameters(
            time=args["max_time"],
            gpus_per_node=1,
            array_parallelism=512,
            cpus_per_task=4,
            partition=args["slurm_partition"])
        executor.map_array(run_experiment, commands)
    else:
        torch.set_float32_matmul_precision("high")
        for command in commands:
            run_experiment(command)
    
