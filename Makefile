.PHONY: debug

debug:
	parallel \
		--eta \
		--jobs 8 \
		--round-robin \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/qys/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset {1} \
		--method {2} \
		--batch_size {3} \
		--lr {4} \
		--weight_decay {5} \
		--data_path data \
		--output_dir main_sweep \
		--num_hparams_seeds 1 \
		--num_init_seeds 1 \
		::: waterbirds celeba chexpert-embedding coloredmnist \
		::: erm dro ttlsa \
		::: 2 16 128 \
		::: 1e-3 1e-5 \
		::: 1e-1 1e-2
