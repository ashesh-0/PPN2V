#! /bin/bash 
python src/scripts/train_noise_model.py  --unnormalized_version --datafname=OptiMEM100x014.tif  --hist_bins=128 --add_gaussian_noise_std=152 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/microscopy/ --gmm_tolerance=1e-6 --channel_idx=2

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=OptiMEM100x014.tif  --hist_bins=128 --add_gaussian_noise_std=228 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/microscopy/ --gmm_tolerance=1e-6 --channel_idx=2

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=OptiMEM100x014.tif  --hist_bins=128 --add_gaussian_noise_std=304 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/microscopy/ --gmm_tolerance=1e-6 --channel_idx=2

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=OptiMEM100x014.tif  --hist_bins=128 --add_gaussian_noise_std=608 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/microscopy/ --gmm_tolerance=1e-6 --channel_idx=2

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=OptiMEM100x014.tif  --hist_bins=128 --add_gaussian_noise_std=152 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/microscopy/ --gmm_tolerance=1e-6 --channel_idx=3

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=OptiMEM100x014.tif  --hist_bins=128 --add_gaussian_noise_std=228 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/microscopy/ --gmm_tolerance=1e-6 --channel_idx=3

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=OptiMEM100x014.tif  --hist_bins=128 --add_gaussian_noise_std=304 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/microscopy/ --gmm_tolerance=1e-6 --channel_idx=3

python src/scripts/train_noise_model.py  --unnormalized_version --datafname=OptiMEM100x014.tif  --hist_bins=128 --add_gaussian_noise_std=608 --train_with_gt_as_clean_data --datadir=/group/jug/ashesh/data/microscopy/ --gmm_tolerance=1e-6 --channel_idx=3


