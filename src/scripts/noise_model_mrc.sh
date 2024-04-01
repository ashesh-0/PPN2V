python src/scripts/train_noise_model.py --datafname='F-actin/GT_all_a.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=3050 --datadir=/group/jug/ashesh/data/BioSR/   --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

python src/scripts/train_noise_model.py --datafname='F-actin/GT_all_a.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=4575 --datadir=/group/jug/ashesh/data/BioSR/   --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

python src/scripts/train_noise_model.py --datafname='F-actin/GT_all_a.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=6100 --datadir=/group/jug/ashesh/data/BioSR/   --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

python src/scripts/train_noise_model.py --datafname='F-actin/GT_all_a.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=12200 --datadir=/group/jug/ashesh/data/BioSR/   --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

sleep 5

python src/scripts/train_noise_model.py --datafname='CCPs/GT_all.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=3050  --datadir=/group/jug/ashesh/data/BioSR/   --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

python src/scripts/train_noise_model.py --datafname='CCPs/GT_all.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=4575  --datadir=/group/jug/ashesh/data/BioSR/   --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

python src/scripts/train_noise_model.py --datafname='CCPs/GT_all.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=6100  --datadir=/group/jug/ashesh/data/BioSR/   --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

python src/scripts/train_noise_model.py --datafname='CCPs/GT_all.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=12200  --datadir=/group/jug/ashesh/data/BioSR/   --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

sleep 5

python src/scripts/train_noise_model.py --datafname='F-actin/GT_all_a.mrc' --datafname2='CCPs/GT_all.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=3050   --datadir=/group/jug/ashesh/data/BioSR/  --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

python src/scripts/train_noise_model.py --datafname='F-actin/GT_all_a.mrc' --datafname2='CCPs/GT_all.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=4575   --datadir=/group/jug/ashesh/data/BioSR/  --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

python src/scripts/train_noise_model.py --datafname='F-actin/GT_all_a.mrc' --datafname2='CCPs/GT_all.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=6100   --datadir=/group/jug/ashesh/data/BioSR/  --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

python src/scripts/train_noise_model.py --datafname='F-actin/GT_all_a.mrc' --datafname2='CCPs/GT_all.mrc' --unnormalized_version   --hist_bins=128 --add_gaussian_noise_std=12200   --datadir=/group/jug/ashesh/data/BioSR/  --gmm_tolerance=1e-10 --train_with_gt_as_clean_data

sleep 5
