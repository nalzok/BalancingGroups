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


