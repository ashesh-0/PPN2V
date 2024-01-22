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
from src.scripts.train_n2v import get_bestmodelname, load_data
from tifffile import imread


def slashstrip(path):
    if path[-1] == '/':
        return path[:-1]
    else:
        return path


dtype = torch.float
device = torch.device("cuda:0")


def evaluate_n2v(net, data):
    results = []
    meanRes = []
    resultImgs = []
    dataTest = data
    print('Evaluating using N2V model')
    for index in tqdm(range(dataTest.shape[0])):

        im = dataTest[index]
        # We are using tiling to fit the image into memory
        # If you get an error try a smaller patch size (ps)
        means = src.ppn2v.pn2v.prediction.tiledPredict(im, net, ps=128, overlap=48, device=device, noiseModel=None)
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


def train_noise_model(
    n2v_modelpath,
    noise_model_rootdirectory,
    data_dir,
    data_fileName,
    normalized_version=True,
    n_gaussian=6,
    n_coeff=4,
    gmm_min_sigma=0.125,
    hard_upper_threshold=None,
    hist_bins=64,
    val_fraction=0.2,
    upperclip_quantile=0.995,
    lowerclip_quantile=0.005,
):

    hostname = socket.gethostname()

    exp_directory = get_workdir(noise_model_rootdirectory, False)

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
    }

    n2v_config = load_config(os.path.dirname(n2v_modelpath))
    if n2v_config is not None:
        assert n2v_config[
            'fname'] == data_fileName, f'N2V should have been trained on the same data!!, Found {n2v_config["fname"]} for N2V and {data_fileName} for noise model'

    add_git_info(config)
    dump_config(config, exp_directory)

    wandb.init(name=os.path.join(hostname, 'noise_model',
                                 *exp_directory.split('/')[-2:]),
               dir=noise_model_rootdirectory,
               project="N2V",
               config=config)

    fpath = os.path.join(data_dir, data_fileName)
    noisy_data = load_data(fpath)

    if n2v_config.get('enable_poisson_noise', False):
        print('Enabling poisson noise for N2V model')
        noisy_data = np.random.poisson(noisy_data)
    elif n2v_config.get('add_gaussian_noise_std', 0.0) > 0.0:
        print('Adding gaussian noise for N2V model', n2v_config['add_gaussian_noise_std'])
        noisy_data = noisy_data + np.random.normal(0, n2v_config['add_gaussian_noise_std'], noisy_data.shape)

    val_N = int(noisy_data.shape[0] * val_fraction)
    noisy_data = noisy_data[val_N:].copy()

    # upperclip data
    max_val = np.quantile(noisy_data, upperclip_quantile)
    noisy_data[noisy_data > max_val] = max_val

    # lowerclip
    min_val = np.quantile(noisy_data, lowerclip_quantile)
    noisy_data[noisy_data < min_val] = min_val

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
    dataName = f"{os.path.basename(slashstrip(data_dir))}-{data_fileName.split('-')[0]}"
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
    print(min_signal, max_signal)
    # import pdb;pdb.set_trace()
    gaussianMixtureNoiseModel = src.ppn2v.pn2v.gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(
        min_signal=min_signal,
        max_signal=max_signal,
        path=exp_directory,
        weight=None,
        n_gaussian=n_gaussian,
        n_coeff=n_coeff,
        device=device,
        min_sigma=gmm_min_sigma)

    gaussianMixtureNoiseModel.train(norm_signal,
                                    norm_obs,
                                    batchSize=100,
                                    n_epochs=2000,
                                    learning_rate=0.01,
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
    parser.add_argument('--n2v_modelpath', type=str)
    parser.add_argument('--noise_model_directory', type=str, default='/home/ashesh.ashesh/training/noise_model/')
    parser.add_argument('--gmm_min_sigma', type=float, default=0.125)
    parser.add_argument('--n_gaussian', type=int, default=6)
    parser.add_argument('--n_coeff', type=int, default=4)
    parser.add_argument('--hist_bins', type=int, default=64)
    parser.add_argument('--unnormalized_version', action='store_true')
    parser.add_argument('--upperclip_quantile', type=float, default=0.999)
    parser.add_argument('--lowerclip_quantile', type=float, default=0.001)

    args = parser.parse_args()
    train_noise_model(
        args.n2v_modelpath,
        args.noise_model_directory,
        args.datadir,
        args.datafname,
        normalized_version=(not args.unnormalized_version),
        n_gaussian=args.n_gaussian,
        n_coeff=args.n_coeff,
        gmm_min_sigma=args.gmm_min_sigma,
        hard_upper_threshold=None,
        hist_bins=args.hist_bins,
        upperclip_quantile=args.upperclip_quantile,
        lowerclip_quantile=args.lowerclip_quantile,
    )
