#!/bin/bash 

# # 1250
# python src/scripts/train_noise_model.py --n2v_modelpath=''  --unnormalized_version --datafname=actin-60x-noise2-highsnr.tif --hist_bins=128 --train_with_gt_as_clean_data --upperclip_quantile=1 --lowerclip_quantile=0.0 --add_gaussian_noise_std=1250.0

# python src/scripts/train_noise_model.py --n2v_modelpath=''  --unnormalized_version --datafname=mito-60x-noise2-highsnr.tif --hist_bins=128 --train_with_gt_as_clean_data --upperclip_quantile=1 --lowerclip_quantile=0.0 --add_gaussian_noise_std=1250.0

# python src/scripts/train_noise_model.py --n2v_modelpath=''  --unnormalized_version --datafname=actin-60x-noise2-highsnr.tif --hist_bins=128 --train_with_gt_as_clean_data --datafname2=mito-60x-noise2-highsnr.tif --upperclip_quantile=1 --add_gaussian_noise_std=1250.0 --lowerclip_quantile=0.0

#!/bin/bash 
python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=3400.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/CCPs

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=6800.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/CCPs

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=13600.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/CCPs

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=20400.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/CCPs

