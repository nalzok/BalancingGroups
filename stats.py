from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def main(metadata):
    df = pd.read_csv(metadata, index_col="id")
    if metadata.name.startswith("metadata_civilcomments_coarse"):
        df["a"] = (df["a"] != 0).astype(float)

    mask = (df["y"] >= 0) & (df["y"] <= 1) & (df["a"] >= 0) & (df["a"] <= 1)
    df = df.loc[mask]

    splits = { "train": 0, "valid": 1, "test": 2 }
    correlations = {}
    for k, v in splits.items():
        split = df[df["split"] == v]
        mat = confusion_matrix(split["y"], split["a"])
        mat = mat.astype(float) / np.sum(mat)
        correlations[k] = np.max(mat)
    return correlations


if __name__ == "__main__":
    metadata_files = [
        "metadata_celeba.csv",
        "metadata_civilcomments_coarse.csv",
        "metadata_multinli.csv",
        "metadata_waterbirds.csv",
    ]
    root = Path("data")
    for metadata_file in metadata_files:
        print(metadata_file)
        correlations = main(root / metadata_file)
        print(" & ".join(f"{v:.3f}" for v in correlations.values()))

