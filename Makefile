.PHONY: debug

debug:
	pipenv run python train.py --data_path data --output_dir main_sweep --num_hparams_seeds 1 --num_init_seeds 1
