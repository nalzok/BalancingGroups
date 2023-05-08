.PHONY: smoke train impute

smoke:
	/home/qys/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset waterbirds \
		--imputed 0.9 \
		--method ttlsa \
		--batch_size 4 \
		--lr 1e-4 \
		--weight_decay 1e-3 \
		--data_path data \
		--output_dir main_sweep \
		--num_hparams_seeds 1 \
		--num_init_seeds 1

train:
	parallel \
		--eta \
		--jobs 8 \
		--round-robin \
		--joblog joblogs/train.txt \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/qys/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset {1} \
		--imputed {2} \
		--method {3} \
		--batch_size {4} \
		--lr {5} \
		--weight_decay {6} \
		--data_path data \
		--output_dir main_sweep \
		--num_hparams_seeds 1 \
		--num_init_seeds 1 \
		::: waterbirds celeba \
		::: 0.1 0.25 0.5 0.75 0.9 \
		::: ttlsa erm \
		::: 16 128 \
		::: 1e-3 1e-5 \
		::: 1e-1 1e-2

impute:
	parallel \
		--eta \
		--jobs 8 \
		--round-robin \
		--joblog joblogs/impute.txt \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/qys/miniconda3/envs/ttlsa/bin/python3.10 impute.py \
		--dataset {1} \
		--missing {2} \
		--data_path data \
		--output_dir impute_sweep \
		--num_hparams_seeds 1 \
		--num_init_seeds 1 \
		::: waterbirds celeba \
		::: 0.1 0.25 0.5 0.75 0.9
