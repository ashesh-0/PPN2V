import warnings

warnings.filterwarnings('ignore')
import argparse
import os
import socket
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import wandb

import git
from src.ppn2v.experiment_saving import add_git_info, dump_config, get_workdir
from src.ppn2v.pn2v import histNoiseModel, training, utils
from src.ppn2v.unet.model import UNet
from tifffile import imread

# See if we can use a GPU
device = utils.getDevice()


def get_bestmodelname(datadir, fileName):
    fname = get_modelname(datadir, fileName)
    return f'best_{fname}.net'


def get_modelname(datadir, fileName: Union[str, Tuple[str, str]]):
    """
    datadir: /mnt/data/ventura_gigascience
    fileName: 'mito-60x-noise2-lowsnr.tif'
    """
    dset = os.path.basename(datadir)
    if isinstance(fileName, tuple):
        fname1 = fileName[0]
        fname1 = fname1.split('.')[0]
        fname1 = fname1.split('-')[0]

        fname2 = fileName[1]
        fname2 = fname2.split('.')[0]
        fname2 = fname2.split('-')[0]

        dataName = f"{dset}-{fname1}-{fname2}"
    else:
        fileName = fileName.split('.')[0]
        dataName = f"{dset}-{fileName.split('-')[0]}"  # This will be used to name the noise2void model

    nameModel = dataName + '-n2v'
    return nameModel


def load_data(datapath):
    print('Loading data from: ', datapath)
    if datapath.split('.')[-1] == 'npy':
        try:
            return np.load(datapath)
        except:
            # for BSD68 test dataset, allow_pickle=True is needed.
            return np.load(datapath, allow_pickle=True)

    elif datapath.split('.')[-1] == 'tif':
        return imread(datapath)
    else:
        raise ValueError('Unknown file type')


def train(
    datadir,
    fname,
    unet_depth=3,
    val_fraction=0.05,
    numOfEpochs=200,
    stepsPerEpoch=10,
    virtualBatchSize=20,
    batchSize=1,
    learningRate=1e-3,
    traindir=None,
    add_gaussian_noise_std=0.0,
    poisson_noise_factor=-1,
    upperclip_quantile=0.995,
    lowerclip_quantile=0.0,
    train_dataset_fraction=1.0,
    patchSize=128,
):
    hostname = socket.gethostname()
    exp_directory = get_workdir(traindir, False)
    print('Experiment directroy: ', exp_directory)
    print('')

    config = {
        'datadir': datadir,
        # 'fname': fname,
        'unet_depth': unet_depth,
        'val_fraction': val_fraction,
        'numOfEpochs': numOfEpochs,
        'stepsPerEpoch': stepsPerEpoch,
        'virtualBatchSize': virtualBatchSize,
        'batchSize': batchSize,
        'learningRate': learningRate,
        'exp_directory': exp_directory,
        'poisson_noise_factor': poisson_noise_factor,
        'add_gaussian_noise_std': add_gaussian_noise_std,
        'upperclip_quantile': upperclip_quantile,
        'lowerclip_quantile': lowerclip_quantile,
        'patchSize': patchSize,
    }
    fname1 = fname2 = None

    if isinstance(fname, tuple) and fname[1] == '':
        fname = fname[0]

    if isinstance(fname, tuple):
        config['fname'] = fname[0]
        config['fname2'] = fname[1]
        assert len(fname) == 2
        fname1 = fname[0]
        fname2 = fname[1]
    else:
        assert isinstance(fname, str)
        config['fname'] = fname

    add_git_info(config)
    dump_config(config, exp_directory)
    wandb.init(name=os.path.join(hostname, *exp_directory.split('/')[-2:]), dir=traindir, project="N2V", config=config)

    net = UNet(2, depth=unet_depth)
    if fname1 is not None:
        assert fname2 is not None
        noisy_data1 = load_data(os.path.join(datadir, fname1))
        noisy_data2 = load_data(os.path.join(datadir, fname2))
        noisy_data = noisy_data1 + noisy_data2
    else:
        noisy_data = load_data(os.path.join(datadir, fname))

    assert noisy_data.shape[-1] >= patchSize, 'Patch size is larger than the image size'

    assert poisson_noise_factor == -1 or add_gaussian_noise_std == 0.0, 'Cannot enable both poisson and gaussian noise'
    if poisson_noise_factor:
        noisy_data = np.random.poisson(noisy_data / poisson_noise_factor) * poisson_noise_factor

    elif add_gaussian_noise_std > 0.0:
        print('Adding gaussian noise with std: ', add_gaussian_noise_std)
        noisy_data = noisy_data + np.random.normal(0, add_gaussian_noise_std, noisy_data.shape)
        # we make sure that the noisy data is positive and the entire noise distribution is above zero
        # noisy_data = noisy_data - noisy_data.min()
        # however, this is no longer needed since I know the cause of the issue with histogram noise model.
        # `ra` was not being correctly set.

    # upperclip
    max_val = np.quantile(noisy_data, upperclip_quantile)
    noisy_data[noisy_data > max_val] = max_val
    # lowerclip
    min_val = np.quantile(noisy_data, lowerclip_quantile)
    noisy_data[noisy_data < min_val] = min_val

    nameModel = get_modelname(datadir, fname)

    # Split training and validation data.
    val_count = int(val_fraction * len(noisy_data))
    my_train_data = noisy_data[val_count:].copy()
    my_val_data = noisy_data[:val_count].copy()
    if train_dataset_fraction < 1.0:
        original_shape = my_train_data.shape
        my_train_data = my_train_data[:int(len(my_train_data) * train_dataset_fraction)]
        print(f'Using only a fraction: {train_dataset_fraction} of the training data', original_shape, 'New shape',
              my_train_data.shape)

    # Start training.
    trainHist, valHist = training.trainNetwork(net=net,
                                               trainData=my_train_data,
                                               valData=my_val_data,
                                               postfix=nameModel,
                                               directory=exp_directory,
                                               noiseModel=None,
                                               device=device,
                                               numOfEpochs=numOfEpochs,
                                               stepsPerEpoch=stepsPerEpoch,
                                               virtualBatchSize=virtualBatchSize,
                                               batchSize=batchSize,
                                               patchSize=patchSize,
                                               learningRate=learningRate)
    return trainHist, valHist


if __name__ == '__main__':
    # Let's look at the training and validation loss
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/group/jug/ashesh/data/ventura_gigascience')
    parser.add_argument('--fname', type=str, default='mito-60x-noise2-lowsnr.tif')
    parser.add_argument('--fname2', type=str, default='')
    parser.add_argument('--unet_depth', type=int, default=3)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--numOfEpochs', type=int, default=200)
    parser.add_argument('--stepsPerEpoch', type=int, default=10)
    parser.add_argument('--virtualBatchSize', type=int, default=200)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--learningRate', type=float, default=1e-3)
    parser.add_argument('--traindir', type=str, default=os.path.expanduser('~/training/N2V/'))
    parser.add_argument('--add_gaussian_noise_std', type=float, default=0.0)
    parser.add_argument('--poisson_noise_factor', action='store_true')
    parser.add_argument('--upperclip_quantile', type=float, default=1.0)
    parser.add_argument('--lowerclip_quantile', type=float, default=0.0)
    parser.add_argument('--train_dataset_fraction', type=float, default=1.0)
    parser.add_argument('--patchSize', type=int, default=1024)

    args = parser.parse_args()

    train(args.datadir, (args.fname, args.fname2),
          unet_depth=args.unet_depth,
          val_fraction=args.val_fraction,
          numOfEpochs=args.numOfEpochs,
          stepsPerEpoch=args.stepsPerEpoch,
          virtualBatchSize=args.virtualBatchSize,
          batchSize=args.batchSize,
          learningRate=args.learningRate,
          traindir=args.traindir,
          add_gaussian_noise_std=args.add_gaussian_noise_std,
          poisson_noise_factor=args.poisson_noise_factor,
          upperclip_quantile=args.upperclip_quantile,
          lowerclip_quantile=args.lowerclip_quantile,
          train_dataset_fraction=args.train_dataset_fraction,
          patchSize=args.patchSize)

    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(valHist, label='validation loss')
    # plt.plot(trainHist, label='training loss')
    # plt.legend()
    # plt.show()
