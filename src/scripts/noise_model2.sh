#!/bin/bash 

# # 1000
# python src/scripts/train_noise_model.py --n2v_modelpath=''  --unnormalized_version --datafname=actin-60x-noise2-highsnr.tif --hist_bins=128 --train_with_gt_as_clean_data --upperclip_quantile=1 --lowerclip_quantile=0.0 --add_gaussian_noise_std=1000.0

# python src/scripts/train_noise_model.py --n2v_modelpath=''  --unnormalized_version --datafname=mito-60x-noise2-highsnr.tif --hist_bins=128 --train_with_gt_as_clean_data --upperclip_quantile=1 --lowerclip_quantile=0.0 --add_gaussian_noise_std=1000.0

# python src/scripts/train_noise_model.py --n2v_modelpath=''  --unnormalized_version --datafname=actin-60x-noise2-highsnr.tif --hist_bins=128 --train_with_gt_as_clean_data --datafname2=mito-60x-noise2-highsnr.tif --upperclip_quantile=1 --lowerclip_quantile=0.0 --add_gaussian_noise_std=1000.0
#!/bin/bash 
python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=3150.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/Microtubules/

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=6300.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/Microtubules/

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=12600.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/Microtubules/

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=GT_all.mrc  --hist_bins=128 --add_gaussian_noise_std=18900.0 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/BioSR/Microtubules/


