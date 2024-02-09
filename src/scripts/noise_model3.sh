#!/bin/bash 
python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=3150.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/CCPs

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=6300.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/CCPs

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=12600.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/CCPs

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=18900.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/CCPs

