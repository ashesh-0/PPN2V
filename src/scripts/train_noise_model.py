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
from src.scripts.train_n2v import get_bestmodelname, get_noisy_data
from tifffile import imread

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
        means = src.ppn2v.pn2v.prediction.tiledPredict(im, net, ps=256, overlap=48, device=device, noiseModel=None)
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


def get_trained_n2v_model(n2v_modeldirectory, data_dir, data_fileName):
    net_fpath = os.path.join(n2v_modeldirectory, get_bestmodelname(data_dir, data_fileName))
    net = torch.load(net_fpath)
    return net


def train_noise_model(n2v_modeldirectory,
                      noise_model_rootdirectory,
                      data_dir,
                      data_fileName,
                      normalized_version=True,
                      n_gaussian=6,
                      n_coeff=4,
                      gmm_lowerClip=0.5,
                      gmm_upperClip=100,
                      gmm_min_sigma=0.125,
                      hard_upper_threshold=None,
                      hist_bins=64):

    hostname = socket.gethostname()

    exp_directory = get_workdir(noise_model_rootdirectory, False)

    config = {
        'datadir': data_dir,
        'fname': data_fileName,
        'normalized_version': normalized_version,
        'n_gaussian': n_gaussian,
        'n_coeff': n_coeff,
        'gmm_lowerClip': gmm_lowerClip,
        'gmm_upperClip': gmm_upperClip,
        'gmm_min_sigma': gmm_min_sigma,
        'hard_upper_threshold': hard_upper_threshold,
        'hist_bins': hist_bins,
    }
    n2v_config = load_config(n2v_modeldirectory)
    if n2v_config is not None:
        assert n2v_config[
            'fname'] == data_fileName, f'N2V should have been trained on the same data!!, Found {n2v_config["fname"]} for N2V and {data_fileName} for noise model'

    add_git_info(config)
    dump_config(config, exp_directory)

    wandb.init(name=os.path.join(hostname, 'noise_models',
                                 *exp_directory.split('/')[-2:]),
               dir=noise_model_rootdirectory,
               project="N2V",
               config=config)

    fpath = os.path.join(data_dir, data_fileName)
    noisy_data = get_noisy_data(fpath)
    noisy_data = noisy_data[:10].copy()
    net = get_trained_n2v_model(n2v_modeldirectory, data_dir, data_fileName)
    signal = evaluate_n2v(net, noisy_data)

    if hard_upper_threshold is not None:
        noisy_data[noisy_data > hard_upper_threshold] = hard_upper_threshold
        signal[signal > hard_upper_threshold] = hard_upper_threshold
        assert signal.max() <= hard_upper_threshold
        assert noisy_data.max() <= hard_upper_threshold

    if normalized_version:
        norm_signal = (signal - signal.mean()) / signal.std()
        norm_obs = (noisy_data - noisy_data.mean()) / noisy_data.std()
    else:
        norm_signal = signal.copy()
        norm_obs = noisy_data.copy()

    min_signal = np.percentile(norm_signal, 0.0)
    max_signal = np.percentile(norm_signal, 100)

    min_obs = np.percentile(norm_obs, 0.0)
    max_obs = np.percentile(norm_obs, 100)

    dataName = f"ventura_gigascience-{data_fileName.split('-')[0]}"
    histogram = src.ppn2v.pn2v.histNoiseModel.createHistogram(hist_bins, min_obs, max_obs, norm_obs, norm_signal)
    hist_path = os.path.join(exp_directory, get_hist_model_name(dataName, normalized_version, hist_bins) + '.npy')
    np.save(hist_path, histogram)
    print('Histogram model saved at', hist_path)

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
                                                            gmm_lowerClip, gmm_upperClip, gmm_min_sigma,
                                                            hard_upper_threshold),
                                    lowerClip=gmm_lowerClip,
                                    upperClip=gmm_upperClip)


if __name__ == '__main__':
    # Let's look at the training and validation loss
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/group/jug/ashesh/data/ventura_gigascience')
    parser.add_argument('--datafname', type=str, default='mito-60x-noise2-lowsnr.tif')
    parser.add_argument('--n2v_model_directory', type=str)
    parser.add_argument('--noise_model_directory', type=str)
    parser.add_argument('--gmm_lowerClip', type=float, default=0.5)
    parser.add_argument('--gmm_upperClip', type=float, default=100)
    parser.add_argument('--gmm_min_sigma', type=float, default=0.125)
    parser.add_argument('--n_gaussian', type=int, default=6)
    parser.add_argument('--n_coeff', type=int, default=4)
    parser.add_argument('--hist_bins', type=int, default=64)
    parser.add_argument('--normalized_version', action='store_false')

    args = parser.parse_args()

    n2v_modeldirectory = ''
    noise_model_directory = '/home/ashesh.ashesh/training/noise_model/'
    data_dir = '/group/jug/ashesh/data/ventura_gigascience/'
    data_fileName = 'actin-60x-noise2-highsnr.tif'

    train_noise_model(args.n2v_model_directory,
                      args.noise_model_directory,
                      args.datadir,
                      args.datafname,
                      normalized_version=args.normalized_version,
                      n_gaussian=args.n_gaussian,
                      n_coeff=args.n_coeff,
                      gmm_lowerClip=args.gmm_lowerClip,
                      gmm_upperClip=args.gmm_upperClip,
                      gmm_min_sigma=args.gmm_min_sigma,
                      hard_upper_threshold=None,
                      hist_bins=args.hist_bins)
