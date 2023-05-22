.PHONY: paper

MISSING_SWEEP = 0.5 0.75 0.875 0.9375 0.96875 0.984375 0.9921875 0.99609375


paper-imputed:
	parallel \
		--eta \
		--jobs $$(nvidia-smi -L | wc -l) \
		--joblog joblogs/$@.txt \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/$$USER/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset {3} \
		--imputed {2} \
		--method {4} \
		--data_path data \
		--output_dir $@ \
		--hparams_seed 2023 \
		--init_seed {1} \
		::: 2023 2024 2025 2026 \
		::: ${MISSING_SWEEP} \
		::: celeba waterbirds multinli civilcomments \
		::: ttlsi ttlsa


impute:
	parallel \
		--eta \
		--jobs $$(nvidia-smi -L | wc -l) \
		--joblog joblogs/$@.txt \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/$$USER/miniconda3/envs/ttlsa/bin/python3.10 impute.py \
		--dataset {2} \
		--missing {1} \
		--data_path data \
		--output_dir $@ \
		--hparams_seed 2023 \
		--init_seed 2023 \
		::: ${MISSING_SWEEP} \
		::: celeba waterbirds multinli civilcomments
