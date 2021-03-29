import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import copy
import os

import numpy as np
import torch
from torch import nn

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # plotting
import seaborn as sns

import V1FreeViewingCode.models.datasets as dd
from torch.utils.data import Dataset, DataLoader, random_split

import V1FreeViewingCode.Analysis.notebooks.Utils as U
import argparse

# import modeling
import V1FreeViewingCode.models.encoders as encoders
import V1FreeViewingCode.models.cores as cores
import V1FreeViewingCode.models.readouts as readouts
import V1FreeViewingCode.models.regularizers as regularizers
import V1FreeViewingCode.models.utils as ut

import pickle
import pandas as pd

from pathlib import Path


def fit_shifter(name='20200304_kilowf', path='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/', tdownsample=2, numlags=12, lengthscale=1):

    num_lags = numlags
    sessid = name # '20200304_kilowf'

    save_dir='../../checkpoints/v1calibration_ls{}'.format(lengthscale)
    figDir = "/home/jake/Data/Repos/V1FreeViewingCode/Figures/2021_pytorchmodeling"
    
    outfile = Path(save_dir) / sessid / 'best_shifter.p'
    if outfile.exists():
        print("fit_shifter: shifter model already fit for this session")
        return

    #%% LOAD ALL DATASETS

    # build tent basis for saccades
    n = 40
    num_basis = 15
    B = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,n)), axis=1) - np.arange(0,n,n/num_basis))/n*num_basis, 0)
    t_downsample = tdownsample

    
    gd = dd.PixelDataset(sessid, stims=["Gabor", "Dots", "BackImage", "Grating", "FixRsvpStim"],
        stimset="Train", num_lags=num_lags,
        downsample_t=t_downsample,
        downsample_s=1,
        valid_eye_rad=5.2,
        dirname=path,
        include_frametime={'num_basis': 40, 'full_experiment': False},
        include_saccades=[{'name':'sacon', 'basis':B, 'offset':-20}, {'name':'sacoff', 'basis':B, 'offset':0}],
        include_eyepos=True,
        preload=True)

    sample = gd[:]

    #%% compute STAS
    """
    Compute STAS using einstein summation through pytorch
    """
    # # use gabor here if it exists, else, use dots
    # if 'Gabor' in gd.stims:
    #     stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Gabor' == s][0]
    # elif 'Dots' in gd.stims:
    #     stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Dots' == s][0]
    # use first stim in list (automatically goes in Gabor, Dots order)
    index = np.where(gd.stim_indices==0)[0]
    sample = gd[index] # load sample 

    stas = torch.einsum('nlwh,nc->lwhc', sample['stim'], sample['robs']-sample['robs'].mean(dim=0))
    sta = stas.detach().cpu().numpy()

    # plot STAs / get RF centers
    """
    Plot space/time STAs 
    """
    NC = sta.shape[3]
    mu = np.zeros((NC,2))
    sx = np.ceil(np.sqrt(NC*2))
    sy = np.round(np.sqrt(NC*2))

    mod2 = sy % 2
    sy += mod2
    sx -= mod2

    tdiff = np.zeros((num_lags, NC))
    blag = np.zeros(NC)

    plt.figure(figsize=(sx*2,sy*2))
    for cc in range(NC):
        w = sta[:,:,:,cc]

        wt = np.std(w, axis=0)
        wt /= np.max(np.abs(wt)) # normalize for numerical stability
        # softmax
        wt = wt**10
        wt /= np.sum(wt)
        sz = wt.shape
        xx,yy = np.meshgrid(np.linspace(-1, 1, sz[1]), np.linspace(1, -1, sz[0]))

        mu[cc,0] = np.minimum(np.maximum(np.sum(xx*wt), -.5), .5) # center of mass after softmax
        mu[cc,1] = np.minimum(np.maximum(np.sum(yy*wt), -.5), .5) # center of mass after softmax

        w = (w -np.mean(w) )/ np.std(w)

        bestlag = np.argmax(np.std(w.reshape( (gd.num_lags, -1)), axis=1))
        blag[cc] = bestlag
        plt.subplot(sx,sy, cc*2 + 1)
        v = np.max(np.abs(w))
        plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=(-1,1,-1,1))
        plt.plot(mu[cc,0], mu[cc,1], '.b')
        plt.title(cc)
        plt.subplot(sx,sy, cc*2 + 2)
        i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
        t1 = w[:,i[0], j[0]]
        plt.plot(t1, '-ob')
        i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
        t2 = w[:,i[0], j[0]]
        plt.plot(t2, '-or')
        yd = plt.ylim()
        tdiff[:,cc] = t1 - t2

    plt.savefig(figDir + "/rawstas_" + gd.id + ".pdf", bbox_inches='tight')

    #%% Refit DivNorm Model on all datasets
    """
    Fit single layer DivNorm model with modulation
    """
    for version in range(3): # range of version numbers
        

        #% Model: convolutional model
        input_channels = gd.num_lags
        hidden_channels = 16
        input_kern = 19
        hidden_kern = 5
        core = cores.Stacked2dDivNorm(input_channels,
                hidden_channels,
                input_kern,
                hidden_kern,
                layers=1,
                gamma_hidden=1e-6, # group sparsity
                gamma_input=1,
                gamma_center=0,
                skip=0,
                final_nonlinearity=True,
                bias=False,
                pad_input=True,
                hidden_padding=hidden_kern//2,
                group_norm=True,
                num_groups=4,
                weight_norm=True,
                hidden_dilation=1,
                input_regularizer="RegMats",
                input_reg_types=["d2x", "center", "d2t"],
                input_reg_amt=[.000005, .01, 0.00001],
                hidden_reg_types=["d2x", "center"],
                hidden_reg_amt=[.000005, .01],
                stack=None,
                use_avg_reg=True)


        # initialize input layer to be centered
        regw = regularizers.gaussian2d(input_kern,sigma=input_kern//4)
        core.features[0].conv.weight.data = torch.einsum('ijkm,km->ijkm', core.features[0].conv.weight.data, torch.tensor(regw))
            
        # Readout
        in_shape = [core.outchannels, gd.NY, gd.NX]
        bias = True
        readout = readouts.Point2DGaussian(in_shape, gd.NC, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
                        gamma_l1=0,gamma_l2=0.00001,
                        align_corners=True, gauss_type='uncorrelated',
                        constrain_positive=False,
                        shifter= {'hidden_features': 20,
                                'hidden_layers': 1,
                                'final_tanh': False,
                                'activation': "softplus",
                                'lengthscale': lengthscale}
                                )

        
        modifiers = {'stimlist': ['frametime', 'sacoff'],
                'gain': [sample['frametime'].shape[1], sample['sacoff'].shape[1]],
                'offset':[sample['frametime'].shape[1],sample['sacoff'].shape[1]],
                'stage': "readout",
                'outdims': gd.NC}

        # combine core and readout into model
        model = encoders.EncoderMod(core, readout, modifiers=modifiers,
            gamma_mod=0,
            weight_decay=.001, optimizer='AdamW', learning_rate=.01, # high initial learning rate because we decay on plateau
            betas=[.9, .999], amsgrad=False)

        # initialize readout based on spike rate and STA centers
        model.readout.bias.data = sample['robs'].mean(dim=0) # initialize readout bias helps
        model.readout._mu.data[0,:,0,:] = torch.tensor(mu.astype('float32')) # initiaalize mus

        #% check that the forward runs (for debugging)
        

        #% Train
        trainer, train_dl, valid_dl = ut.get_trainer(gd, version=version,
                    save_dir=save_dir,
                    name=gd.id,
                    auto_lr=False,
                    batchsize=1000,
                    num_workers=64,
                    earlystopping=False)

        trainpath = Path(save_dir) / gd.id / "version_{}".format(version)
        if not trainpath.exists():
            trainer.fit(model, train_dl, valid_dl)
        
    # Load best version
    pth = Path(save_dir) / gd.id

    valloss = np.array([])
    ind = np.array([])
    for v in pth.rglob('version*'):

        try:
            df = pd.read_csv(str(v / "metrics.csv"))
            ind = np.append(ind, int(v.name.split('_')[1]))
            valloss = np.append(valloss, np.nanmin(df.val_loss.to_numpy()))
        except:
            "skip"

    sortind = np.argsort(ind)
    ind = ind[sortind]
    valloss = valloss[sortind]

    vernum = ind[np.argmin(valloss)]

    # load all shifters and save them
    shifters = nn.ModuleList()
    for vernum in ind:
        # load linear model version
        ver = "version_" + str(int(vernum))
        chkpath = pth / ver / 'checkpoints'
        best_epoch = ut.find_best_epoch(chkpath)
        model2 = encoders.EncoderMod.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))
        shifters.append(model2.readout.shifter.cpu())
   
    # load best model version
    ver = "version_" + str(int(vernum))
    chkpath = pth / ver / 'checkpoints'
    best_epoch = ut.find_best_epoch(chkpath)
    model2 = encoders.EncoderMod.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))
    shifter = model2.readout.shifter
    shifter.cpu()

    pickle.dump( {'cids': gd.cids, 'shifter': shifter, 'shifters': shifters, 'vernum': ind, 'vallos': valloss, 'numlags': num_lags, 'tdownsample': tdownsample}, open(str(outfile), "wb" ) )

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Use V1 to correct the ')
    parser.add_argument("-n", "--name", required=True,
                        help='The id tag for the session')
    parser.add_argument("-p", '--path', nargs=1, default='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/',
                        type=str,
                        help='path to data')
    parser.add_argument("--numlags", nargs=1, default=12, type=int, help="number of lags in model")
    parser.add_argument("--tdownsample", nargs=1, default=2, type=int, help="temporal downsampling factor (default=2)")
    parser.add_argument("--lengthscale", nargs=1, default=1, type=int, help="lengthscale (smoothness) for shifter (default=1)")

    args = parser.parse_args()
    fit_shifter(**vars(args))



