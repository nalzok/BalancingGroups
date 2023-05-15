.PHONY: smoke-impute smoke-train train impute train-impute


MISSING_SWEEP = 0.5 0.75 0.875 0.9375 0.96875 0.984375 0.9921875 0.99609375


smoke-impute:
	/home/qys/miniconda3/envs/ttlsa/bin/python3.10 impute.py \
		--dataset multinli \
		--missing 0.9999 \
		--data_path data \
		--output_dir impute_sweep \
		--hparams_seed 2023 \
		--init_seed 2023


smoke-train:
	/home/qys/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset coloredmnist \
		--method ttlsa \
		--data_path data \
		--output_dir main_sweep \
		--hparams_seed 2023 \
		--init_seed 2023


hyper_cmnist:
	parallel \
		--eta \
		--jobs $$(nvidia-smi -L | wc -l) \
		--joblog joblogs/$@.txt \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/qys/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset {1} \
		--method {2} \
		--data_path data \
		--output_dir $@ \
		--lr {3} \
		--weight_decay {4} \
		--batch_size {5} \
		--hparams_seed 2023 \
		--init_seed 2023 \
		::: coloredmnist \
		::: erm dro subg \
		::: 1e-5 1e-4 1e-3 \
		::: 1e-4 1e-3 1e-2 1e-1 1e-0 \
		::: 128


hyper_chexpert:
	parallel \
		--eta \
		--jobs $$(nvidia-smi -L | wc -l) \
		--joblog joblogs/$@.txt \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/qys/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset {1} \
		--method {2} \
		--data_path data \
		--output_dir $@ \
		--lr {3} \
		--weight_decay {4} \
		--batch_size {5} \
		--hparams_seed 2023 \
		--init_seed 2023 \
		::: chexpert-embedding \
		::: erm dro subg \
		::: 1e-5 1e-4 1e-3 \
		::: 1e-4 1e-3 1e-2 1e-1 1e-0 \
		::: 128


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
		--hparams_seed 2023 \
		--init_seed 2023 \
		::: ${MISSING_SWEEP} \
		::: coloredmnist


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
		--hparams_seed 2023 \
		--init_seed 2023 \
		::: ${MISSING_SWEEP} \
		::: civilcomments multinli \
		::: ttlsa
