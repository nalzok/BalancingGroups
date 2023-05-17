.PHONY: paper

paper:
	parallel \
		--eta \
		--jobs $$(nvidia-smi -L | wc -l) \
		--joblog joblogs/$@.txt \
		--rpl '{%0} 1 $$_ = $$job->slot() - 1' \
		env CUDA_VISIBLE_DEVICES={%0} \
		/home/$$USER/miniconda3/envs/ttlsa/bin/python3.10 train.py \
		--dataset {2} \
		--method {3} \
		--data_path data \
		--output_dir $@ \
		--hparams_seed 2023 \
		--init_seed {1} \
		::: 2023 2024 2025 2026 \
		::: celeba waterbirds multinli civilcomment \
		::: erm ttlsi dro subg ttlsa
