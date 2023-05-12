import datetime
import torch


class Tee:
    def __init__(self, fname, stream, mode="a+"):
        self.stream = stream
        self.file = open(fname, mode)
        self.file.write(f"============ {datetime.datetime.now().isoformat()} ============\n")
        self.file.flush()

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def randl(l_):
    return l_[torch.randperm(len(l_))[0]]


# Table 5 from https://arxiv.org/abs/2110.14503
chosen_hparams_best = {
        "celeba": {
            "erm": (-4.0, -1.0, 39.8, 128),
            "jtt": (-3.0, -2.0, 31.0, 32),
            "rwy": (-4.0, -2.0, 18.8, 2),
            "suby": (-3.0, -2.0, 41.4, 128),
            "rwg": (-5.0, -1.0, 6.2, 32),
            "subg": (-4.0, -1.0, 8.2, 8),
            "dro": (-5.0, -4.0, 15.4, 64),
            },
        "civilcomments": {
            "erm": (-4.0, -4.0, 2.8, 4),
            "jtt": (-5.0, -2.0, 4.2, 32),
            "rwy": (-3.0, -4.0, 4.2, 32),
            "suby": (-3.0, -3.0, 3.8, 16),
            "rwg": (-5.0, -3.0, 3.0, 4),
            "subg": (-4.0, -4.0, 3.2, 8),
            "dro": (-3.0, -3.0, 4.2, 32),
            },
        "multinli": {
            "erm": (-4.0, -4.0, 4.6, 2),
            "jtt": (-5.0, -3.0, 5.0, 4),
            "rwy": (-3.0, -4.0, 4.6, 16),
            "suby": (-4.0, -3.0, 4.4, 4),
            "rwg": (-5.0, -3.0, 2.0, 4),
            "subg": (-4.0, -3.0, 5.6, 2),
            "dro": (-4.0, -3.0, 5.4, 8),
            },
        "waterbirds": {
            "erm": (-4.0, -3.0, 257.4, 4),
            "jtt": (-3.0, -4.0, 289.4, 4),
            "rwy": (-5.0, -1.0, 109.4, 4),
            "suby": (-5.0, -2.0, 319.8, 2),
            "rwg": (-5.0, 0.0, 3.0, 2),
            "subg": (-4.0, -2.0, 175.2, 4),
            "dro": (-5.0, 0.0, 6.0, 4),
            }
        }


chosen_hparams_best["coloredmnist"] = {
        "erm": (-4.0, -3.0, 257.4, 4),
        "jtt": (-3.0, -4.0, 289.4, 4),
        "rwy": (-5.0, -1.0, 109.4, 4),
        "suby": (-5.0, -2.0, 319.8, 2),
        "rwg": (-5.0, 0.0, 3.0, 2),
        "subg": (-4.0, -2.0, 175.2, 4),
        "dro": (-5.0, 0.0, 6.0, 4),
        }


chosen_hparams_best["chexpert-embedding"] = {
        "erm": (-4.0, -3.0, 257.4, 4),
        "jtt": (-3.0, -4.0, 289.4, 4),
        "rwy": (-5.0, -1.0, 109.4, 4),
        "suby": (-5.0, -2.0, 319.8, 2),
        "rwg": (-5.0, 0.0, 3.0, 2),
        "subg": (-4.0, -2.0, 175.2, 4),
        "dro": (-5.0, 0.0, 6.0, 4),
        }


for dataset in chosen_hparams_best:
    chosen_hparams_best[dataset]["ttlsa"] = chosen_hparams_best[dataset]["erm"]
