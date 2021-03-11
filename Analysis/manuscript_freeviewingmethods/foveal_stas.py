#%% Big Import
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import copy
import os

import numpy as np
import torch
from torch import nn

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
# %% Check that it was run and load the best models

# sessid = "20200304_kilowf"
sessid = "20191119_kilowf"
lengthscale = 1 # the default is 1
save_dir='../../checkpoints/v1calibration_ls{}'.format(lengthscale)

outfile = Path(save_dir) / sessid / 'best_shifter.p'
if outfile.exists():
    print("fit_shifter was already run. Loading [%s]" %sessid)
    tmp = pickle.load(open(str(outfile), "rb"))
    cids = tmp['cids']
    shifter = tmp['shifter']
else:
    print("foveal_stas: need to run fit_shifter for %s" %sessid)


#%% LOAD ALL DATASETS

# build tent basis for saccades
n = 40
num_basis = 15
B = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,n)), axis=1) - np.arange(0,n,n/num_basis))/n*num_basis, 0)
num_lags = 12
t_downsample = 2

import V1FreeViewingCode.models.datasets as dd
import importlib
importlib.reload(dd)
gd = dd.PixelDataset(sessid, stims=["Dots", "Gabor", "BackImage", "Grating", "FixRsvpStim"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    include_frametime={'num_basis': 40, 'full_experiment': False},
    include_saccades=[{'name':'sacon', 'basis':B, 'offset':-20}, {'name':'sacoff', 'basis':B, 'offset':0}],
    include_eyepos=True,
    preload=True)

sample = gd[:]

#%% compute STAS
"""
Compute STAS using einstein summation through pytorch
"""
index = np.where(gd.stim_indices==1)[0]
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



#%% Refit DivNorm Model on all datasets
#% This is optional
import V1FreeViewingCode.models.encoders as encoders
importlib.reload(encoders)
importlib.reload(readouts)
importlib.reload(cores)

#%%
"""
Fit single layer DivNorm model with modulation
"""
for version in [1]: # range of version numbers
    

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
                    gamma_l1=0,gamma_l2=0,
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
                earlystopping=False)

    # trainpath = Path(save_dir) / gd.id / "version_{}".format(version)
    # if not trainpath.exists():
    trainer.fit(model, train_dl, valid_dl)
#%%
# Load best version
pth = Path(save_dir) / sessid

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
plt.plot(ind, valloss, '.')
plt.xlabel("Version #")
plt.ylabel("Loss")


shifters = nn.ModuleList()
for vernum in ind:
    # load linear model version
    ver = "version_" + str(int(vernum))
    chkpath = pth / ver / 'checkpoints'
    best_epoch = ut.find_best_epoch(chkpath)
    model2 = encoders.EncoderMod.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))
    shifters.append(model2.readout.shifter.cpu())



#%% load best model
vernum = ind[np.argmin(valloss)]
print("Loading version %d" %vernum)
ver = "version_" + str(int(vernum))
chkpath = pth / ver / 'checkpoints'
best_epoch = ut.find_best_epoch(chkpath)
model = encoders.EncoderMod.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))
nn.utils.remove_weight_norm(model.core.features[0].conv)

model.core.plot_filters()
#%% load data
path='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'

# build tent basis for saccades
n = 40
num_basis = model.core.input_channels
B = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,n)), axis=1) - np.arange(0,n,n/num_basis))/n*num_basis, 0)
num_lags = model.core.input_channels

# re-load dataset to fit shifter
gd = dd.PixelDataset(sessid, stims=["Gabor"],
    stimset="Train", num_lags=num_lags,
    downsample_t=2,
    downsample_s=1,
    valid_eye_rad=5.2,
    cids=cids,
    include_eyepos=True,
    dirname=path,
    include_frametime={'num_basis': 40, 'full_experiment': False},
    include_saccades=[{'name':'sacon', 'basis':B, 'offset':-20}, {'name':'sacoff', 'basis':B, 'offset':0}],
    preload=True)

# %% plot the shifter 
# build inputs for shifter plotting
xx,yy = np.meshgrid(np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,100),np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,100))
xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

inputs = torch.cat( (xgrid,ygrid), dim=1)


# y = model.readout.shifter(inputs) # best shfiter
y = shifters[0](inputs)
y2 = y.detach().cpu().numpy()
y2/=gd.valid_eye_rad/60 # conver to arcmin
vmin = np.min(y2)
vmax = np.max(y2)

fig = plt.figure(figsize=(5,3))
ax1 = plt.subplot(1,2,1)
ax1.imshow(y2[:,0].reshape((100,100)), extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
ax1.tick_params(labelsize=8)
ax1.set_title("Horizontal", fontsize=8)

ax2 = plt.subplot(1,2,2)
im = ax2.imshow(y2[:,1].reshape((100,100)), extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
ax2.set_title("Vertical", fontsize=8)

# fig.axes()
plt.colorbar(im)

# cbar = fig.colorbar(im, cax=axcbar, orientation="vertical")
# axcbar.tick_params(labelsize=7)
# axcbar.set_ylabel("Shift (Arcmin)", fontsize=8)
figDir = "/home/jake/Data/Repos/V1FreeViewingCode/Figures/2021_pytorchmodeling"
# plt.savefig(figDir + "/shifters_" + gd.id + ".pdf", bbox_inches='tight')
# %%
# model.core.plot_filters()

# %% shift stimulus

index = np.where(gd.stim_indices==0)[0]
sample = gd[index] # load sample 

import torch.nn.functional as F
affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])

aff = torch.tensor([[1,0,0],[0,1,0]])

# pick shifter (1 means use LNLN model)
shift = model.readout.shifter(sample['eyepos']).detach() # shifters[1](sample['eyepos']).detach()
affine_trans = shift[:,:,None]+aff[None,:,:]
affine_trans[:,0,0] = 1
affine_trans[:,0,1] = 0
affine_trans[:,1,0] = 0
affine_trans[:,1,1] = 1

im = sample['stim'].detach().clone()
n = im.shape[0]
grid = F.affine_grid(affine_trans, torch.Size((n, gd.num_lags, gd.NY,gd.NX)), align_corners=True)

im2 = F.grid_sample(im, grid, mode='bilinear', align_corners=True)

#%% new STAs on shifted stimulus
stas = torch.einsum('nlwh,nc->lwhc', im, sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()
stas = torch.einsum('nlwh,nc->lwhc', im2, sample['robs']-sample['robs'].mean(dim=0))
sta2 = stas.detach().cpu().numpy()

#%%
extent = np.round(gd.rect/gd.ppd*60)
extent = np.asarray([extent[i] for i in [0,2,1,3]])

# plotting
NC = sta.shape[3]
# plt.figure(figsize=(8,NC*2))
for cc in range(NC):
    plt.figure(figsize=(8,2))
    
    w = sta[:,:,:,cc]    
    w2 = sta2[:,:,:,cc]

    bestlag = np.argmax(np.std(w2.reshape( (gd.num_lags, -1)), axis=1))
    
    w = (w -np.mean(w[bestlag,:,:]) )/ np.std(w)
    w2 = (w2 -np.mean(w2[bestlag,:,:]) )/ np.std(w2)


    plt.subplot(1,4,3)
    # plt.subplot(NC,4, cc*4 + 1)
    v = np.max(np.abs(w2))
    plt.imshow(w2[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=extent)
    plt.title("After")
    plt.xlabel("arcmin")
    plt.ylabel("arcmin")
    # plt.subplot(NC,4, cc*4 + 2)
    plt.subplot(1,4,4)
    i,j=np.where(w2[bestlag,:,:]==np.max(w2[bestlag,:,:]))
    plt.plot(w2[:,i[0], j[0]], '-ob')
    i,j=np.where(w2[bestlag,:,:]==np.min(w2[bestlag,:,:]))
    plt.plot(w2[:,i[0], j[0]], '-or')
    yd = plt.ylim()
    # sns.despine(offset=0, trim=True)
    plt.xlabel("Lag (frame=8ms)")

    # plt.subplot(NC,4, cc*4 + 3)
    plt.subplot(1,4,1)
    plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm")
    plt.title("Before")
    plt.xlabel("arcmin")
    plt.ylabel("arcmin")

    plt.subplot(1,4,2)
    # plt.subplot(NC,4, cc*4 + 4)
    i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-ob')
    i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-or')
    plt.axvline(bestlag, color='k', linestyle='--')
    plt.ylim(yd)
    # sns.despine(offset=0, trim=True)
    plt.xlabel("Lag (frame=8ms)")
    # plt.savefig(figDir + "/sta_shift" + gd.id + "_" + str(cc) + ".pdf", bbox_inches='tight')
    plt.close()
# %%

# %%
