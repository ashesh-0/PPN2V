#!/bin/bash 

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=3400.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/ER

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=6800.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/ER

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=13600.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/ER

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=20400.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/ER
