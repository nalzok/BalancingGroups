# TTLSA

Code to replicate the experimental results from our TTLSA paper.
Based on the facebookresearch/BalancingGroups repo by FAIR.

## Replicating the main results

### Installing dependencies

Easiest way to have a working environment for this repo is to create a conda environement with the following commands

```bash
conda create --name ttlsa --file conda-spec/spec-file.txt
```	

If conda is not available, please install the dependencies listed in the requirements.txt file.

### Download, extract and Generate metadata for datasets

This script downloads, extracts and formats the datasets metadata so that it works with the rest of the code out of the box.

```bash
python setup_datasets.py --download --data_path data celeba waterbirds civilcomments multinli
```

### Launch jobs

To reproduce the experiments in the paper:

```bash
make train
```

### Aggregate results

The parse.py script can generate the main plots and tables from the paper. 
This script can be called while the experiments are still running. 

```bash
# worst group accuracy
python3 -m aggregate --path paper --selector1 min --selector2 min --split te
# average accuracy
python3 -m aggregate --path paper --selector1 avg --selector2 avg --split te
```

## License

This source code is released under the CC-BY-NC license, included [here](LICENSE).
