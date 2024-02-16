#!/bin/bash 
python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=4450.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/ER  --gmm_tolerance=1e-6

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=8900 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/ER  --gmm_tolerance=1e-6

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=17800 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/ER  --gmm_tolerance=1e-6

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=26700 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/ER  --gmm_tolerance=1e-6

