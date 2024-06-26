import argparse
import os
import pickle
import socket
import sys
import urllib
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from scipy.stats import norm
from torch.distributions import normal
from tqdm import tqdm

import src.ppn2v.pn2v.gaussianMixtureNoiseModel
import src.ppn2v.pn2v.histNoiseModel
import src.ppn2v.pn2v.prediction
from src.ppn2v.experiment_saving import add_git_info, dump_config, get_workdir, load_config
from src.ppn2v.pn2v import *
from src.ppn2v.pn2v.utils import *
from src.scripts.train_n2v import get_bestmodelname, get_mrc_data, load_data
from tifffile import imread


def slashstrip(path):
    if path[-1] == '/':
        return path[:-1]
    else:
        return path


dtype = torch.float
device = torch.device("cuda:0")


def evaluate_n2v(net, data, patchsize=96, overlap=80):
    results = []
    meanRes = []
    resultImgs = []
    dataTest = data
    print('Evaluating using N2V model')
    for index in tqdm(range(dataTest.shape[0])):

        im = dataTest[index]
        # We are using tiling to fit the image into memory
        # If you get an error try a smaller patch size (ps)
        means = src.ppn2v.pn2v.prediction.tiledPredict(im,
                                                       net,
                                                       ps=patchsize,
                                                       overlap=overlap,
                                                       device=device,
                                                       noiseModel=None)
        resultImgs.append(means)

    return np.array(resultImgs)


def get_hist_model_name(dataName, normalized_version, bins):
    return f"HistNoiseModel_{dataName}_Norm{int(normalized_version)}_Bins{int(bins)}_bootstrap"


def get_gmm_model_name(dataName, normalized_version, n_gaussian, n_coeff, gmm_lowerClip, gmm_upperClip, gmm_min_sigma,
                       hard_upper_threshold):
    nameGMMNoiseModel = f"GMMNoiseModel_{dataName}_{n_gaussian}_{n_coeff}"
    nameGMMNoiseModel += f"_Clip{gmm_lowerClip}-{gmm_upperClip}"
    nameGMMNoiseModel += f"_Sig{gmm_min_sigma}"
    nameGMMNoiseModel += f"_Up{hard_upper_threshold}"

    nameGMMNoiseModel += f"_Norm{int(normalized_version)}_bootstrap"
    return nameGMMNoiseModel


def get_trained_n2v_model(n2v_modelpath):
    net = torch.load(n2v_modelpath)
    return net


def get_frame_count(data_shape, datausage_fraction=1.0):
    """
    In case of partial data usage, we need to know how many frames to use.
    """
    framepixelcount = np.prod(data_shape[1:3])
    pixelcount = int(data_shape[0] * framepixelcount * datausage_fraction)
    frame_count = int(np.ceil(pixelcount / framepixelcount))
    return frame_count


def cleanup_name(fname):
    return fname.replace('/', '_').replace('.mrc', '').replace('.tif', '')


def train_noise_model(
    n2v_modelpath,
    noise_model_rootdirectory,
    data_dir,
    data_fileName,
    channel_idx=None,
    channel_dim=None,
    normalized_version=True,
    n_gaussian=6,
    n_coeff=4,
    gmm_min_sigma=0.125,
    hard_upper_threshold=None,
    hist_bins=64,
    val_fraction=0.2,
    upperclip_quantile=0.995,
    lowerclip_quantile=0.005,
    input_is_sum=False,
    train_dataset_fraction=1.0,
    train_with_gt_as_clean_data=False,
    add_gaussian_noise_std=-1,
    poisson_noise_factor=-1,
    data_usage_fraction=1.0,
    gmm_tolerance=None,
    train_pure_noise_model=False,
    clean_datapath=None,
):
    """
    Args:
        data_usage_fraction: Fraction of the data to use. This is useful when we want to use only a fraction of the data.
    """
    hostname = socket.gethostname()

    exp_directory = get_workdir(noise_model_rootdirectory, False)
    # sanity checks
    if clean_datapath is not None:
        assert os.path.exists(clean_datapath), f'clean_datapath does not exist: {clean_datapath}'
        assert train_with_gt_as_clean_data is False, 'train_with_gt_as_clean_data should be False when clean_datapath is provided'
        assert n2v_modelpath is None, 'n2v_modelpath should be None when clean_datapath is provided'
        assert poisson_noise_factor == -1, 'poisson_noise_factor should be -1 when clean_datapath is provided'
        assert add_gaussian_noise_std == -1, 'add_gaussian_noise_std should be -1 when clean_datapath is provided'
        assert val_fraction == 0.0, 'val_fraction should be 0.0 when clean_datapath is provided'

    config = {
        'datadir': data_dir,
        'fname': data_fileName,
        'normalized_version': normalized_version,
        'n_gaussian': n_gaussian,
        'n_coeff': n_coeff,
        'gmm_min_sigma': gmm_min_sigma,
        'hard_upper_threshold': hard_upper_threshold,
        'hist_bins': hist_bins,
        'upperclip_quantile': upperclip_quantile,
        'lowerclip_quantile': lowerclip_quantile,
        'val_fraction': val_fraction,
        'exp_directory': exp_directory,
        'n2v_modelpath': n2v_modelpath,
        'input_is_sum': input_is_sum,
        'train_with_gt_as_clean_data': train_with_gt_as_clean_data,
        'gmm_tolerance': gmm_tolerance,
        'channel_idx': channel_idx,
        'channel_dim': channel_dim,
        'train_pure_noise_model': train_pure_noise_model,
        'poisson_noise_factor': poisson_noise_factor,
        'clean_datapath': clean_datapath,
    }
    n2v_config = load_config(os.path.dirname(n2v_modelpath)) if n2v_modelpath is not None else None
    if add_gaussian_noise_std > 0:
        config['add_gaussian_noise_std'] = add_gaussian_noise_std

    if n2v_config is not None:
        if n2v_config.get('add_gaussian_noise_std', 0.0) > 0.0:
            config['add_gaussian_noise_std'] = n2v_config['add_gaussian_noise_std']
            add_gaussian_noise_std = n2v_config['add_gaussian_noise_std']
        n2v_fnames = set({n2v_config['fname'], n2v_config.get('fname2', '')})
        fnames = set(data_fileName)
        assert n2v_fnames == fnames, f'N2V should have been trained on the same data!!, Found {n2v_fnames} for N2V and {fnames} for noise model'
        if poisson_noise_factor != -1:
            assert 'poisson_noise_factor' in n2v_config and n2v_config[
                'poisson_noise_factor'] == poisson_noise_factor, 'Poisson noise factor should be same for N2V and noise model'

    add_git_info(config)
    dump_config(config, exp_directory)

    wandb.init(name=os.path.join(hostname, 'noise_model',
                                 *exp_directory.split('/')[-2:]),
               dir=noise_model_rootdirectory,
               project="N2V",
               config=config)

    noisy_data = None
    assert isinstance(data_fileName, tuple)
    count = 0
    for fName, c_idx, c_dim in zip(data_fileName, channel_idx, channel_dim):
        if fName == '':
            assert c_idx is None, 'filename needs to be provided if channel index is provided.'
            continue
        assert train_pure_noise_model is False, 'train_pure_noise_model does not need any data.'
        fpath = os.path.join(data_dir, fName)
        if fpath.endswith('.mrc'):
            data = get_mrc_data(fpath)
        else:
            data = load_data(fpath)
        if c_idx is not None:
            print('Using only channel', c_idx, ' at dimension', c_dim, ' from the data', data.shape)
            assert c_dim is not None, 'channel_dim should be provided if channel_idx is provided'
            if c_dim == 1:
                data = data[:, c_idx]
            elif c_dim == 2:
                data = data[:, :, c_idx]
            elif c_dim == 3:
                data = data[:, :, :, c_idx]
            else:
                raise ValueError('Invalid channel dimension', c_dim)

        if noisy_data is None:
            noisy_data = data
        elif noisy_data is not None and data.shape == noisy_data.shape:
            noisy_data += data
        elif noisy_data is not None and data.shape != noisy_data.shape:
            min_N = min(data.shape[0], noisy_data.shape[0])
            noisy_data = noisy_data[:min_N] + data[:min_N]

        count += 1

    if train_pure_noise_model:
        print('Training pure noise model')
        assert count == 0, 'train_pure_noise_model should not have any data'
        assert add_gaussian_noise_std > 0 or poisson_noise_factor > 0, 'train_pure_noise_model should have some noise'
        signal = np.random.uniform(0, 65535, size=(100, 128, 128)).astype(np.uint32)

        if poisson_noise_factor > 0:
            print('Enabling poisson noise with factor', poisson_noise_factor)
            # The higher this factor, the more the poisson noise.
            signal = signal // poisson_noise_factor
            noisy_data = np.tile(signal[:, None], (1, 100, 1, 1))
            noisy_data = noisy_data.reshape(-1, 128, 128)
            noisy_data = np.random.poisson(noisy_data)  #* poisson_noise_factor

        if add_gaussian_noise_std > 0.0:
            print('Adding gaussian noise', add_gaussian_noise_std)
            noisy_data = np.tile(signal[:, None], (1, 100, 1, 1))
            noisy_data = noisy_data.reshape(-1, 128, 128)
            noisy_data = noisy_data + np.random.normal(0, add_gaussian_noise_std, noisy_data.shape)
    else:
        # Here, we are averaging the data. Because, this is what we will do when working with usplit.
        if input_is_sum is False:
            noisy_data = noisy_data // count

        raw_data = noisy_data
        # I think clipping should be done on original data. After that we can add noise. Otherwise
        # it is incorrect in multiple ways => now, I realized that this is incorrect. We should clip the data after adding noise.:
        # 1. N2V does exactly that: first clips the data and then adds noise. => n2v should also clip the data after adding noise.
        # 2. If after adding noise, we clip the data, then for some portions we will see not noisy but saturated data which is incorrect. this is correct.
        # 3. Now, in the current data loader, we are clipping the data before adding noise. => we were doing wrong.
        if poisson_noise_factor > 0:
            print('Enabling poisson noise for N2V model with factor', poisson_noise_factor)
            # The higher this factor, the more the poisson noise.
            noisy_data = np.random.poisson(noisy_data / poisson_noise_factor) * poisson_noise_factor

        if add_gaussian_noise_std > 0.0:
            print('Adding gaussian noise for N2V model', add_gaussian_noise_std)
            noisy_data = noisy_data + np.random.normal(0, add_gaussian_noise_std, noisy_data.shape)

        # upperclip data
        max_val = np.quantile(noisy_data, upperclip_quantile)
        noisy_data[noisy_data > max_val] = max_val

        # lowerclip
        min_val = np.quantile(noisy_data, lowerclip_quantile)
        noisy_data[noisy_data < min_val] = min_val
        # save
        # dataName = f"{os.path.basename(slashstrip(data_dir))}-{'_'.join([cleanup_name(fname).split('-')[0] for fname in data_fileName])}"
        # if add_gaussian_noise_std > 0:
        #     dataName += f'Gaus{int(add_gaussian_noise_std)}'
        # if poisson_noise_factor > 0:
        #     dataName += f'Pois{int(poisson_noise_factor)}'
        # np.save(os.path.join(exp_directory, f'noisy_{dataName}.npy'), noisy_data)
        # print('Noisy data saved at', os.path.join(exp_directory, f'noisy_{dataName}.npy'))
        # import pdb; pdb.set_trace()

        val_N = int(noisy_data.shape[0] * val_fraction)
        noisy_data = noisy_data[val_N:].copy()
        raw_data = raw_data[val_N:].copy()
        if data_usage_fraction < 1.0:
            frame_count = get_frame_count(noisy_data.shape, data_usage_fraction)
            noisy_data = noisy_data[:frame_count].copy()
            raw_data = raw_data[:frame_count].copy()
            print(f'Using only a fraction: {data_usage_fraction} of the training data', frame_count, 'New shape',
                  noisy_data.shape)

        if train_dataset_fraction < 1.0:
            original_shape = noisy_data.shape
            noisy_data = noisy_data[:int(len(noisy_data) * train_dataset_fraction)]
            print(f'Using only a fraction: {train_dataset_fraction} of the training data', original_shape, 'New shape',
                  noisy_data.shape)

        if train_with_gt_as_clean_data:
            signal = raw_data
            assert upperclip_quantile == 1.0, 'upperclip_quantile should be 1.0 when using ground truth as clean data'
            assert lowerclip_quantile == 0.0, 'lowerclip_quantile should be 0.0 when using ground truth as clean data'
        elif clean_datapath is not None:
            print('Using externally provided clean data as denoised data')
            signal = load_data(clean_datapath)
            # NOTE: this works for now, but we should have a better way to handle this.
            signal = signal[:noisy_data.shape[0]].astype(np.uint32)
            noisy_data = noisy_data.astype(np.float64)
        else:
            net = get_trained_n2v_model(n2v_modelpath)
            signal = evaluate_n2v(net, noisy_data)

    if hard_upper_threshold is not None:
        noisy_data[noisy_data > hard_upper_threshold] = hard_upper_threshold
        signal[signal > hard_upper_threshold] = hard_upper_threshold
        assert signal.max() <= hard_upper_threshold
        assert noisy_data.max() <= hard_upper_threshold

    if normalized_version:
        norm_signal = (signal - noisy_data.mean()) / noisy_data.std()
        norm_obs = (noisy_data - noisy_data.mean()) / noisy_data.std()
    else:
        norm_signal = signal.copy()
        norm_obs = noisy_data.copy()

    min_obs = np.percentile(norm_obs, 0.0)
    max_obs = np.percentile(norm_obs, 100)
    min_sig = np.percentile(norm_signal, 0.0)
    max_sig = np.percentile(norm_signal, 100)
    min_val = min(min_obs, min_sig)
    max_val = max(max_obs, max_sig)
    dataName = f"{os.path.basename(slashstrip(data_dir))}-{'_'.join([cleanup_name(fname).split('-')[0] for fname in data_fileName])}"
    histogram = src.ppn2v.pn2v.histNoiseModel.createHistogram(hist_bins, min_val, max_val, norm_obs, norm_signal)
    hist_path = os.path.join(exp_directory, get_hist_model_name(dataName, normalized_version, hist_bins) + '.npy')
    np.save(hist_path, histogram)
    print('Histogram model saved at', hist_path)

    # Code below ensures that both GMM and histogram based models use the same min max signal.
    min_signal = np.min(histogram[1, ...])
    max_signal = np.max(histogram[2, ...])
    # print(min_signal, max_signal)
    # min_signal = np.percentile(norm_signal, 0.0)
    # max_signal = np.percentile(norm_signal, 100)
    gaussianMixtureNoiseModel = src.ppn2v.pn2v.gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(
        min_signal=min_signal,
        max_signal=max_signal,
        path=exp_directory,
        weight=None,
        n_gaussian=n_gaussian,
        n_coeff=n_coeff,
        device=device,
        min_sigma=gmm_min_sigma)

    if gmm_tolerance is not None:
        gaussianMixtureNoiseModel.set_tolerance(gmm_tolerance)
    gaussianMixtureNoiseModel.train(norm_signal,
                                    norm_obs,
                                    batchSize=250000,
                                    n_epochs=4000,
                                    learning_rate=0.1,
                                    name=get_gmm_model_name(dataName, normalized_version, n_gaussian, n_coeff,
                                                            lowerclip_quantile, upperclip_quantile, gmm_min_sigma,
                                                            hard_upper_threshold),
                                    lowerClip=0,
                                    upperClip=100)


if __name__ == '__main__':
    # Let's look at the training and validation loss
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/group/jug/ashesh/data/ventura_gigascience')
    parser.add_argument('--datafname', type=str, default='mito-60x-noise2-lowsnr.tif')
    parser.add_argument('--channel_idx', type=int, default=None)
    parser.add_argument('--channel_dim', type=int, default=None)
    parser.add_argument('--clean_datapath', type=str, default=None)
    parser.add_argument('--n2v_modelpath', type=str)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--datafname2', type=str, default='')
    parser.add_argument('--channel_idx2', type=int, default=None)
    parser.add_argument('--channel_dim2', type=int, default=None)
    parser.add_argument('--input_is_sum', action='store_true')
    parser.add_argument('--noise_model_directory', type=str, default='/home/ashesh.ashesh/training/noise_model/')
    parser.add_argument('--gmm_min_sigma', type=float, default=0.125)
    parser.add_argument('--n_gaussian', type=int, default=6)
    parser.add_argument('--n_coeff', type=int, default=4)
    parser.add_argument('--hist_bins', type=int, default=64)
    parser.add_argument('--normalized_version', action='store_true')
    parser.add_argument('--upperclip_quantile', type=float, default=1.0)
    parser.add_argument('--lowerclip_quantile', type=float, default=0.0)
    parser.add_argument('--train_dataset_fraction', type=float, default=1.0)
    parser.add_argument('--add_gaussian_noise_std', type=float, default=-1)
    parser.add_argument('--train_with_gt_as_clean_data', action='store_true')
    parser.add_argument('--poisson_noise_factor', type=float, default=-1)
    parser.add_argument('--gmm_tolerance', type=float, default=1e-6)
    parser.add_argument('--train_pure_noise_model', action='store_true')
    parser.add_argument('--data_usage_fraction', type=float, default=1.0)

    def remove_extension(fname):
        output = fname.replace('.tif', '').replace('.npy', '')
        return output

    args = parser.parse_args()
    if args.clean_datapath is not None:
        assert remove_extension(os.path.basename(args.clean_datapath.replace('_pred', ''))) == remove_extension(
            args.datafname), 'clean_datapath should have the same name as datafname'

    train_noise_model(
        args.n2v_modelpath,
        args.noise_model_directory,
        args.datadir,
        (args.datafname, args.datafname2),
        channel_idx=(args.channel_idx, args.channel_idx2),
        channel_dim=(args.channel_dim, args.channel_dim2),
        normalized_version=args.normalized_version,
        n_gaussian=args.n_gaussian,
        n_coeff=args.n_coeff,
        gmm_min_sigma=args.gmm_min_sigma,
        hard_upper_threshold=None,
        hist_bins=args.hist_bins,
        upperclip_quantile=args.upperclip_quantile,
        lowerclip_quantile=args.lowerclip_quantile,
        input_is_sum=args.input_is_sum,
        train_dataset_fraction=args.train_dataset_fraction,
        train_with_gt_as_clean_data=args.train_with_gt_as_clean_data,
        add_gaussian_noise_std=args.add_gaussian_noise_std,
        poisson_noise_factor=args.poisson_noise_factor,
        gmm_tolerance=args.gmm_tolerance,
        train_pure_noise_model=args.train_pure_noise_model,
        data_usage_fraction=args.data_usage_fraction,
        clean_datapath=args.clean_datapath,
        val_fraction=args.val_fraction,
    )
