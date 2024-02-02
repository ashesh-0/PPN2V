#! /bin/bash
noise_std=$1
python src/scripts/train_n2v.py --fname='mito-60x-noise2-highsnr.tif' --add_gaussian_noise_std="$noise_std" 
python src/scripts/train_n2v.py --fname='actin-60x-noise2-highsnr.tif' --add_gaussian_noise_std="$noise_std"
python src/scripts/train_n2v.py --fname='mito-60x-noise2-highsnr.tif' --fname2='actin-60x-noise2-highsnr.tif' --add_gaussian_noise_std="$noise_std"