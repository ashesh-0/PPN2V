import warnings

warnings.filterwarnings('ignore')
import argparse
import os
import socket

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


def get_modelname(datadir, fileName):
    """
    datadir: /mnt/data/ventura_gigascience
    fileName: 'mito-60x-noise2-lowsnr.tif'
    """
    dset = os.path.basename(datadir)
    dataName = f"{dset}-{fileName.split('-')[0]}"  # This will be used to name the noise2void model
    # print(path,'\n',dataName)
    nameModel = dataName + '-n2v'
    return nameModel


def get_noisy_data(datapath):
    return imread(datapath)


def train(datadir,
          fname,
          unet_depth=3,
          val_fraction=0.05,
          numOfEpochs=200,
          stepsPerEpoch=10,
          virtualBatchSize=20,
          batchSize=1,
          learningRate=1e-3,
          traindir=None):
    hostname = socket.gethostname()
    exp_directory = get_workdir(traindir, False)

    config = {
        'datadir': datadir,
        'fname': fname,
        'unet_depth': unet_depth,
        'val_fraction': val_fraction,
        'numOfEpochs': numOfEpochs,
        'stepsPerEpoch': stepsPerEpoch,
        'virtualBatchSize': virtualBatchSize,
        'batchSize': batchSize,
        'learningRate': learningRate
    }
    add_git_info(config)
    dump_config(config, exp_directory)
    wandb.init(name=os.path.join(hostname, *exp_directory.split('/')[-2:]), dir=traindir, project="N2V", config=config)

    net = UNet(1, depth=unet_depth)
    data = get_noisy_data(os.path.join(datadir, fname))
    nameModel = get_modelname(datadir, fname)

    # Split training and validation data.
    val_count = int(val_fraction * len(data))
    my_train_data = data[:-1 * val_count].copy()
    my_val_data = data[-1 * val_count:].copy()

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
                                               learningRate=learningRate)
    return trainHist, valHist


if __name__ == '__main__':
    # Let's look at the training and validation loss
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/group/jug/ashesh/data/ventura_gigascience')
    parser.add_argument('--fname', type=str, default='mito-60x-noise2-lowsnr.tif')
    parser.add_argument('--unet_depth', type=int, default=3)
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--numOfEpochs', type=int, default=200)
    parser.add_argument('--stepsPerEpoch', type=int, default=10)
    parser.add_argument('--virtualBatchSize', type=int, default=200)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--learningRate', type=float, default=1e-3)
    parser.add_argument('--traindir', type=str, default=os.path.expanduser('~/training/N2V/'))
    args = parser.parse_args()

    train(args.datadir,
          args.fname,
          unet_depth=args.unet_depth,
          val_fraction=args.val_fraction,
          numOfEpochs=args.numOfEpochs,
          stepsPerEpoch=args.stepsPerEpoch,
          virtualBatchSize=args.virtualBatchSize,
          batchSize=args.batchSize,
          learningRate=args.learningRate,
          traindir=args.traindir)

    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(valHist, label='validation loss')
    # plt.plot(trainHist, label='training loss')
    # plt.legend()
    # plt.show()
