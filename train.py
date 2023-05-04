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


class Tee:
    def __init__(self, fname, stream, mode="a+"):
        self.stream = stream
        self.file = open(fname, mode)

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def randl(l_):
    return l_[torch.randperm(len(l_))[0]]


def parse_args():
    parser = argparse.ArgumentParser(description='Balancing baselines')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--slurm_output_dir', type=str, default='slurm_outputs')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--slurm_partition', type=str, default=None)
    parser.add_argument('--max_time', type=int, default=3*24*60)
    parser.add_argument('--num_hparams_seeds', type=int, default=20)
    parser.add_argument('--num_init_seeds', type=int, default=5)
    parser.add_argument('--selector', type=str, default='min_acc_va')
    return vars(parser.parse_args())


def run_experiment(fabric, args):
    start_time = time.time()
    L.seed_everything(args["init_seed"])
    loaders = get_loaders(args["data_path"], args["dataset"], args["batch_size"], args["method"])

    sys.stdout = Tee(os.path.join(
        args["output_dir"], 'seed_{}_{}.out'.format(
            args["hparams_seed"], args["init_seed"])), sys.stdout)
    sys.stderr = Tee(os.path.join(
        args["output_dir"], 'seed_{}_{}.err'.format(
            args["hparams_seed"], args["init_seed"])), sys.stderr)
    checkpoint_file = os.path.join(
        args["output_dir"], 'seed_{}_{}.pt'.format(
            args["hparams_seed"], args["init_seed"]))
    best_checkpoint_file = os.path.join(
        args["output_dir"],
        "seed_{}_{}.best.pt".format(args["hparams_seed"], args["init_seed"]),
    )

    model = {
        "erm": models.ERM,
        "suby": models.ERM,
        "subg": models.ERM,
        "rwy": models.ERM,
        "rwg": models.ERM,
        "dro": models.GroupDRO,
        "jtt": models.JTT,
        "ttlsa": models.TTLSA,
    }[args["method"]](fabric, args, loaders["tr"])

    last_epoch = 0
    best_selec_val = float('-inf')
    # if os.path.exists(checkpoint_file):
    #     model.load(checkpoint_file)
    #     last_epoch = model.last_epoch
    #     best_selec_val = model.best_selec_val

    bcts_optimizer_initial = None   # suppress warning
    if args["method"] == "ttlsa":
        model, model.optimizer, model.bcts_optimizer = fabric.setup(model, model.optimizer, model.bcts_optimizer)
        bcts_optimizer_initial = model.bcts_optimizer.state_dict()
    else:
        model, model.optimizer = fabric.setup(model, model.optimizer)

    for epoch in range(last_epoch, args["num_epochs"]):
        if epoch == args["T"] + 1 and args["method"] == "jtt":
            loaders = get_loaders(
                args["data_path"],
                args["dataset"],
                args["batch_size"],
                args["method"],
                model.weights.tolist())

        if args["method"] == "ttlsa":
            with torch.no_grad():
                model.T.zero_()
                model.b.zero_()

        loader = fabric.setup_dataloaders(loaders["tr"])
        for i, x, y, g in loader:
            model.update(i, x, y, g, epoch)

        if args["method"] == "ttlsa":
            model.bcts_optimizer.load_state_dict(bcts_optimizer_initial)

            loader = fabric.setup_dataloaders(loaders["va"])
            for i, x, y, g in loader:
                model.calibrate(i, x, y, g, epoch)

        result = {
            "args": args, "epoch": epoch, "time": time.time() - start_time}
        for loader_name, loader in loaders.items():
            loader = fabric.setup_dataloaders(loader)
            avg_acc, group_accs = model.accuracy(loader)
            result["acc_" + loader_name] = group_accs
            result["avg_acc_" + loader_name] = avg_acc

        selec_value = {
            "min_acc_va": min(result["acc_va"]),
            "avg_acc_va": result["avg_acc_va"],
        }[args["selector"]]

        if selec_value >= best_selec_val:
            model.best_selec_val = selec_value
            best_selec_val = selec_value
            # model.save(best_checkpoint_file)

        if fabric.is_global_zero:
            # model.module.save(checkpoint_file)
            print(json.dumps(result))


if __name__ == "__main__":
    args = parse_args()

    commands = []
    for hparams_seed in range(args["num_hparams_seeds"]):
        torch.manual_seed(hparams_seed)
        args["hparams_seed"] = hparams_seed

        args["dataset"] = randl(
            ["waterbirds", "celeba", "chexpert-embedding", "coloredmnist", "multinli", "civilcomments"])
        args["dataset"] = "chexpert-embedding"  # override

        args["method"] = randl(
            ["erm", "suby", "subg", "rwy", "rwg", "dro", "jtt", "ttlsa"])
        args["method"] = "ttlsa"        # override

        args["num_epochs"] = {
            "waterbirds": 300 + 60,
            "celeba": 50 + 10,
            "chexpert-embedding": 300 + 60,
            "coloredmnist": 300 + 60,
            "waterbirds": 300 + 60,
            "multinli": 5 + 2,
            "civilcomments": 5 + 2
        }[args["dataset"]]
        # args["num_epochs"] = 8          # override

        args["eta"] = 0.1
        args["lr"] = randl([1e-5, 1e-4, 1e-3])
        args["lr"] = 1e-3               # override
        args["weight_decay"] = randl([1e-4, 1e-3, 1e-2, 1e-1, 1])
        args["weight_decay"] = 1e-3     # override

        if args["dataset"] in ["waterbirds", "celeba", "chexpert-embedding", "coloredmnist"]:
            args["batch_size"] = randl([2, 4, 8, 16, 32, 64, 128])
        else:
            args["batch_size"] = randl([2, 4, 8, 16, 32])
        args["batch_size"] = 32         # override

        args["up"] = randl([4, 5, 6, 20, 50, 100])
        args["T"] = {
            "waterbirds": randl([40, 50, 60]),
            "celeba": randl([1, 5, 10]),
            "chexpert-embedding": randl([40, 50, 60]),
            "coloredmnist": randl([40, 50, 60]),
            "multinli": randl([1, 2]),
            "civilcomments": randl([1, 2])
        }[args["dataset"]]
        # args["T"] = 4          # override

        for init_seed in range(args["num_init_seeds"]):
            args["init_seed"] = init_seed
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
        fabric = L.Fabric(accelerator="cuda", devices=1, strategy="ddp")
        fabric.launch()
        for command in commands:
            run_experiment(fabric, command)
    
