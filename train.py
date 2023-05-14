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
methods = {"erm", "suby", "subg", "rwy", "rwg", "dro", "jtt", "ttlsa"}


def parse_args():
    parser = argparse.ArgumentParser(description='Balancing baselines')
    parser.add_argument('--dataset', type=str, choices=datasets)
    parser.add_argument('--imputed', type=float, default=None)
    parser.add_argument('--method', type=str, choices=methods)
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
    start_time = time.time()
    L.seed_everything(args["init_seed"])
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
    sys.stdout = Tee(os.path.join(args["output_dir"], f"{stem}.out"), sys.stdout)
    sys.stderr = Tee(os.path.join(args["output_dir"], f"{stem}.err"), sys.stderr)

    model = {
        "erm": models.ERM,
        "suby": models.ERM,
        "subg": models.ERM,
        "rwy": models.ERM,
        "rwg": models.ERM,
        "dro": models.GroupDRO,
        "jtt": models.JTT,
        "ttlsa": models.TTLSA,
    }[args["method"]](args, loaders["tr"])

    bcts_optimizer_initial = None   # suppress warning
    if args["method"] == "ttlsa":
        bcts_optimizer_initial = model.bcts_optimizer.state_dict()

    for epoch in range(args["num_epochs"]):
        if epoch == args["T"] + 1 and args["method"] == "jtt":
            loaders = get_loaders(
                args["data_path"],
                args["dataset"],
                args["batch_size"],
                args["method"],
                model.weights.tolist())

        if args["method"] == "ttlsa":
            with torch.inference_mode():
                model.T.zero_()
                model.b.zero_()

        train_loss = 0
        for i, x, y, g in loaders["tr"]:
            train_loss += model.update(i, x, y, g, epoch)

        if args["method"] == "ttlsa":
            model.bcts_optimizer.load_state_dict(bcts_optimizer_initial)

            for i, x, y, g in loaders["va"]:
                model.calibrate(i, x, y, g, epoch)

        result = {
            "args": args,
            "epoch": epoch,
            "time": time.time() - start_time,
            "loss_tr": train_loss
        }
        for loader_name, loader in loaders.items():
            if loader_name == "tr":
                continue
            avg_acc, corrects, totals, group_accs = model.accuracy(loader)
            result["acc_" + loader_name] = group_accs
            result["corrects_" + loader_name] = corrects
            result["totals_" + loader_name] = totals
            result["avg_acc_" + loader_name] = avg_acc

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
        "waterbirds": 300 + 60,
        "multinli": 5 + 2,
        "civilcomments": 5 + 2
    }[args["dataset"]]

    log_lr, log_wd, epoch, batch_size = chosen_hparams_best[args["dataset"]][args["method"]]
    args["lr"] = 10**log_lr
    args["weight_decay"] = 10**log_wd
    args["batch_size"] = batch_size

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
    
