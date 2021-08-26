#%% imports
from re import L
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import copy
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# import matplotlib
# matplotlib.use('pdf')
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

#%% helper functions

def shift_stim(im, shift, gd):
    print("inside shift stim")
    print(im.shape)
    affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    aff = torch.tensor([[1,0,0],[0,1,0]])
    print("inside shift stim")
    affine_trans = shift[:,:,None]+aff[None,:,:]
    affine_trans[:,0,0] = 1
    affine_trans[:,0,1] = 0
    affine_trans[:,1,0] = 0
    affine_trans[:,1,1] = 1
    print("inside shift stim")
    n = im.shape[0]
    grid = F.affine_grid(affine_trans, torch.Size((n, gd.num_lags, gd.NY,gd.NX)), align_corners=True)
    print("inside shift stim")
    im2 = F.grid_sample(im, grid, mode='bilinear', align_corners=True)
    return im2

def plot_sta_fig(sta, gd):
    # plot STAs / get RF centers
    """
    Plot space/time STAs 
    """
    num_lags = gd.num_lags
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
        # plt.plot(mu[cc,0], mu[cc,1], '.b')
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

def make_figures(name='20200304_kilowf', path='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/', tdownsample=2, numlags=12, lengthscale=1, nspace=100):
    """
    make_figures
    Make STA figures and dump analyses for Figure 5 of Yates et al., 2021
    Inputs:
        gd      <Dataset>   loaded pixel dataset
        outdict <dict>      meta data from shifter fit (saved by fit_shifter)
        figDir  <str>       path to directory to save figures
        nspace  <int>       number of spatial positions in shifter plots (default = 100)
    
    Output:
        None
    Saves a file mat file with all analyses and dumps a pdf for each cell
    """
    print("Running Make Figures")

    import scipy.io as sio # for saving matlab files

    figDir = "/home/jake/Data/Repos/V1FreeViewingCode/Figures/2021_pytorchmodeling"

    save_dir='../../checkpoints/v1calibration_ls{}'.format(lengthscale)

    outfile = Path(save_dir) / name / 'best_shifter.p'
    if outfile.exists():
        print("fit_shifter was already run. Loading [%s]" %name)
        tmp = pickle.load(open(str(outfile), "rb"))
        cids = tmp['cids']
        shifter = tmp['shifter']
        shifters = tmp['shifters']
        vernum = tmp['vernum']
        valloss = tmp['vallos']
        num_lags = tmp['numlags']
        t_downsample = tmp['tdownsample']
    else:
        print("v1_tracker_calibration: need to run fit_shifter for %s" %name)
        return

    
    n = 40
    num_basis = 15
    B = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,n)), axis=1) - np.arange(0,n,n/num_basis))/n*num_basis, 0)
    gd = dd.PixelDataset(name, stims=["Dots", "Gabor"], #, "BackImage", "Grating", "FixRsvpStim"],
        stimset="Train", num_lags=num_lags,
        downsample_t=t_downsample,
        downsample_s=1,
        valid_eye_rad=5.2,
        include_frametime={'num_basis': 40, 'full_experiment': False},
        include_saccades=[{'name':'sacon', 'basis':B, 'offset':-20}, {'name':'sacoff', 'basis':B, 'offset':0}],
        include_eyepos=True,
        preload=True)

    sample = gd[:]

    # --- Plot all shifters

    # build inputs for shifter plotting
    xax = np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,nspace)
    xx,yy = np.meshgrid(xax,xax)
    xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
    ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

    inputs = torch.cat( (xgrid,ygrid), dim=1)

    nshifters = len(shifters)
    
    print("Start Plot")
    fig = plt.figure(figsize=(7,nshifters*2))
    shiftX = []
    shiftY = []

    for i in range(nshifters):
        y = shifters[i](inputs)

        y2 = y.detach().cpu().numpy()
        y2/=gd.valid_eye_rad/60 # convert to arcmin
        vmin = np.min(y2)
        vmax = np.max(y2)

        plt.subplot(nshifters,2,i*2+1)
        im = plt.contourf(xax, xax, y2[:,0].reshape((nspace,nspace)), vmin=vmin, vmax=vmax)

        if i == 0:
            plt.title("Horizontal")

        plt.colorbar(im)

        plt.subplot(nshifters,2,i*2+2)
        im = plt.contourf(xax, xax, y2[:,1].reshape((nspace,nspace)), vmin=vmin, vmax=vmax) #, extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
        if i == 0:
            plt.title("Vertical")

        plt.colorbar(im)

        shiftX.append(y2[:,0].reshape((nspace,nspace)))
        shiftY.append(y2[:,1].reshape((nspace,nspace)))


    print("Save fig")
    plt.savefig(figDir + "/shifters_" + gd.id + ".pdf", bbox_inches='tight')
    print("Success")
    # calculate mean and standard deviation across shifters
    Xarray = np.asarray(shiftX)
    Yarray = np.asarray(shiftY)
    mux = Xarray.mean(axis=0)
    sdx = Xarray.std(axis=0)

    muy = Yarray.mean(axis=0)
    sdy = Yarray.std(axis=0)

    
    print("Select Stim")
    # select stimulus
    if 'Dots' in gd.stims:
        stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Dots' == s][0]
    elif 'Gabor' in gd.stims:
        stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Gabor' == s][0]
    
    index = np.where(gd.stim_indices==stimuse)[0]
    sample = gd[index] # load sample 
    
    # use best model
    bestver = np.argmin(valloss)
    shift = shifters[bestver](sample['eyepos']).detach()
    y = shifters[bestver](inputs)
    y2 = y.detach().cpu().numpy()

    print("Shift Stim")
    # shift stimulus
    im = sample['stim'] #.detach().clone() # original
    im2 = shift_stim(im, shift, gd) # shifted

    print("Get STAs")
    # compute new STAs on shifted stimulus
    stas = torch.einsum('nlwh,nc->lwhc', im, sample['robs']-sample['robs'].mean(dim=0))
    sta = stas.detach().cpu().numpy()
    stas = torch.einsum('nlwh,nc->lwhc', im2, sample['robs']-sample['robs'].mean(dim=0))
    sta2 = stas.detach().cpu().numpy()

    print("Save Mat File")
    # Save output for matlab    
    fname = figDir + "/rfs_" + gd.id + ".mat"
    mdict = {'cids': cids, 'xspace': xx, 'yspace': yy, 'shiftx': y2[:,0].reshape((100,100)), 'shifty': y2[:,1].reshape((100,100)), 'stas_pre': sta, 'stas_post': sta2,
        'valloss': valloss, 'mushiftx': mux, 'mushifty': muy, 'sdshiftx': sdx, 'sdshifty': sdy}

    sio.savemat(fname, mdict)
    
    # compute dimensions
    extent = np.round(gd.rect/gd.ppd*60)
    extent = np.asarray([extent[i] for i in [0,2,3,1]])

    print("Plot STAs")
    # Plot STAs Before and After
    NC = sta.shape[3]

    # Loop over cells
    for cc in range(NC):
        plt.figure(figsize=(8,2))
        
        w = sta[:,:,:,cc]    
        w2 = sta2[:,:,:,cc]

        bestlag = np.argmax(np.std(w2.reshape( (gd.num_lags, -1)), axis=1))
        
        w = (w -np.mean(w[bestlag,:,:]) )/ np.std(w) # before
        w2 = (w2 -np.mean(w2[bestlag,:,:]) )/ np.std(w2) # after

        plt.subplot(1,4,3) # After Space
        v = np.max(np.abs(w2))
        plt.imshow(w2[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=extent)
        plt.title("After")
        plt.xlabel("arcmin")
        plt.ylabel("arcmin")

        plt.subplot(1,4,4) # After Time
        i,j=np.where(w2[bestlag,:,:]==np.max(w2[bestlag,:,:]))
        plt.plot(w2[:,i[0], j[0]], '-ob')
        i,j=np.where(w2[bestlag,:,:]==np.min(w2[bestlag,:,:]))
        plt.plot(w2[:,i[0], j[0]], '-or')
        yd = plt.ylim()
        plt.xlabel("Lag (frame=8ms)")

        plt.subplot(1,4,1) # Before Space
        plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=extent)
        plt.title("Before")
        plt.xlabel("arcmin")
        plt.ylabel("arcmin")

        plt.subplot(1,4,2) # Before Time
        i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
        plt.plot(w[:,i[0], j[0]], '-ob')
        i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
        plt.plot(w[:,i[0], j[0]], '-or')
        plt.axvline(bestlag, color='k', linestyle='--')
        plt.ylim(yd)
        
        plt.xlabel("Lag (frame=8ms)")
        
        plt.savefig(figDir + "/sta_shift" + gd.id + "_" + str(cc) + ".pdf", bbox_inches='tight')
        plt.close('all')


    print("Save Mat File")
    # save all STAs as one figure    
    plot_sta_fig(sta, gd) # Before
    plt.savefig(figDir + "/rawstas_" + gd.id + ".pdf", bbox_inches='tight')

    plot_sta_fig(sta2, gd) # After
    plt.savefig(figDir + "/shiftstas_" + gd.id + ".pdf", bbox_inches='tight')
    plt.close('all')

#%%
name='20200304_kilowf'
path='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'
tdownsample=2
numlags=12
lengthscale=1
stimlist=["Gabor", "Dots", "BackImage", "Grating", "FixRsvpStim"]
    
num_lags = numlags
sessid = name # '20200304_kilowf'

save_dir='../../checkpoints/v1calibration_ls%d_version%d' %(lengthscale, 1)
figDir = "/home/jake/Data/Repos/V1FreeViewingCode/Figures/2021_pytorchmodeling"

outfile = Path(save_dir) / sessid / 'best_shifter.p'
if outfile.exists():
    print("fit_shifter: shifter model already fit for this session")

#%% LOAD ALL DATASETS

# build tent basis for saccades
n = 40
num_basis = 15
B = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,n)), axis=1) - np.arange(0,n,n/num_basis))/n*num_basis, 0)
t_downsample = tdownsample


gd = dd.PixelDataset(sessid, stims=stimlist,
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
sx = int(np.ceil(np.sqrt(NC*2)))
sy = int(np.round(np.sqrt(NC*2)))

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
for version in range(5): # range of version numbers
    

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


    #% Train
    trainer, train_dl, valid_dl = ut.get_trainer(gd, version=version,
                save_dir=save_dir,
                name=gd.id,
                auto_lr=False,
                batchsize=1000,
                max_epochs=20,
                num_workers=64,
                earlystopping=False)

    trainpath = Path(save_dir) / gd.id / "version_{}".format(version)
    if not trainpath.exists():
        trainer.fit(model, train_dl, valid_dl)

#%%

# fname = '/home/jake/Data/Repos/V1FreeViewingCode/Analysis/manuscript_freeviewingmethods/lightning_logs/version_0/checkpoints/epoch=0-step=244.ckpt'
# model2 = encoders.EncoderMod.load_from_checkpoint(fname)
model2 = model

#%%
model2.core.plot_filters()
#%%

model2
#%%
from pytorch_lightning import Trainer

trainer = Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=20)
trainer.fit(model, train_dl, valid_dl)
#%%
    
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
    model2 = encoders.EncoderMod.load_from_checkpoint(str(chkpath / best_epoch))
    shifters.append(model2.readout.shifter.cpu())

# load best model version
ver = "version_" + str(int(vernum))
chkpath = pth / ver / 'checkpoints'
best_epoch = ut.find_best_epoch(chkpath)
model2 = encoders.EncoderMod.load_from_checkpoint(str(chkpath / best_epoch))
shifter = model2.readout.shifter
shifter.cpu()

print("saving file")
outdict = {'cids': gd.cids, 'shifter': shifter, 'shifters': shifters, 'vernum': ind, 'vallos': valloss, 'numlags': num_lags, 'tdownsample': tdownsample, 'lengthscale': lengthscale}
pickle.dump( outdict, open(str(outfile), "wb" ) )


#%%



