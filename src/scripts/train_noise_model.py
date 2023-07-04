import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import git
import socket
import wandb

# import sys
# sys.path.append('../../')

from src.ppn2v.unet.model import UNet

from src.ppn2v.pn2v import utils
from src.ppn2v.pn2v import histNoiseModel
from src.ppn2v.pn2v import training
from src.ppn2v.experiment_saving import  get_workdir, add_git_info
from tifffile import imread
import os
# See if we can use a GPU
device=utils.getDevice()

def get_modelname(datadir, fileName):
    """
    datadir: /mnt/data/ventura_gigascience
    fileName: 'mito-60x-noise2-lowsnr.tif'
    """
    dset = os.path.basename(datadir)
    dataName=f"{dset}-{fileName.split('-')[0]}" # This will be used to name the noise2void model
    # print(path,'\n',dataName)
    nameModel=dataName+'-n2v'
    return nameModel

def get_noisy_data(datapath):
    return imread(datapath)


def train(datadir, fname, unet_depth=3, val_fraction=0.05, numOfEpochs=200, 
          stepsPerEpoch=10, virtualBatchSize=20,batchSize=1, learningRate=1e-3,
          traindir=None):
    hostname = socket.gethostname()
    exp_directory = get_workdir(traindir, False)
    config = {
        'datadir':datadir,
        'fname':fname,
        'unet_depth':unet_depth,
        'val_fraction':val_fraction,
        'numOfEpochs':numOfEpochs,
        'stepsPerEpoch':stepsPerEpoch,
        'virtualBatchSize':virtualBatchSize,
        'batchSize':batchSize,
        'learningRate':learningRate
    }
    add_git_info(config)
    
    wandb.init(name=os.path.join(hostname),
                         dir=traindir,
                         project="N2V",
                         config=config)

    net = UNet(1, depth=unet_depth)
    data = get_noisy_data(os.path.join(datadir, fname))
    nameModel = get_modelname(datadir, fname)

    # Split training and validation data.
    val_count = int(val_fraction*len(data))
    my_train_data=data[:-1*val_count].copy()
    my_val_data=data[-1*val_count:].copy()

    # Start training.
    trainHist, valHist = training.trainNetwork(net=net, trainData=my_train_data, valData=my_val_data,
                                            postfix= nameModel, directory=exp_directory, noiseModel=None,
                                            device=device, numOfEpochs= numOfEpochs, stepsPerEpoch = stepsPerEpoch, 
                                            virtualBatchSize=virtualBatchSize, batchSize=batchSize, 
                                            learningRate=learningRate)
    return trainHist, valHist

if __name__ == '__main__':
    # Let's look at the training and validation loss
    datadir = '/group/jug/ashesh/data/ventura_gigascience'
    fname = 'mito-60x-noise2-lowsnr.tif'
    unet_depth=3
    val_fraction=0.05
    numOfEpochs=20 
    stepsPerEpoch=10
    virtualBatchSize=20
    batchSize=1
    learningRate=1e-3
    traindir=os.path.expanduser('~/training/N2V/')

    train(datadir, fname, unet_depth=unet_depth, val_fraction=val_fraction, numOfEpochs=numOfEpochs, 
          stepsPerEpoch=stepsPerEpoch, virtualBatchSize=virtualBatchSize,batchSize=batchSize, learningRate=learningRate,
          traindir=traindir)
    
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(valHist, label='validation loss')
    # plt.plot(trainHist, label='training loss')
    # plt.legend()
    # plt.show()