.PHONY: smoke-impute smoke-train train impute train-impute


MISSING_SWEEP = 0.5 0.75 0.875 0.9375 0.96875 0.984375


smoke-impute:
	/home/qys/miniconda3/envs/ttlsa/bin/python3.10 impute.py \
		--dataset multinli \
		--missing 0.9999 \
		--data_path data \
		--output_dir impute_sweep \
		--hparams_seed 0 \
		--init_seed 0


smoke-train:
	/home/qys/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset multinli \
		--imputed 0.75 \
		--method ttlsa \
		--data_path data \
		--output_dir main_sweep \
		--hparams_seed 0 \
		--init_seed 0


train:
	parallel \
		--eta \
		--jobs 8 \
		--joblog joblogs/train.txt \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/qys/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset {2} \
		--method {3} \
		--data_path data \
		--output_dir paper_sweep \
		--hparams_seed 0 \
		--init_seed {1} \
		::: 2023 2024 2025 2026 \
		::: waterbirds celeba chexpert-embedding coloredmnist multinli civilcomments \
		::: erm ttlsa


impute:
	parallel \
		--eta \
		--jobs 8 \
		--joblog joblogs/impute.txt \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/qys/miniconda3/envs/ttlsa/bin/python3.10 impute.py \
		--dataset {2} \
		--missing {1} \
		--data_path data \
		--output_dir impute_sweep \
		--hparams_seed 0 \
		--init_seed 0 \
		::: ${MISSING_SWEEP} \
		::: civilcomments multinli


train-impute:
	parallel \
		--eta \
		--jobs 8 \
		--joblog joblogs/train-impute.txt \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/qys/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset {2} \
		--imputed {1} \
		--method {3} \
		--data_path data \
		--output_dir main_sweep \
		--hparams_seed 0 \
		--init_seed 0 \
		::: ${MISSING_SWEEP} \
		::: civilcomments multinli \
		::: ttlsa
