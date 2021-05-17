#%% Big Import
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import copy
import os

import numpy as np
import torch
from torch import nn

import matplotlib
# matplotlib.use("pdf")
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

from scipy.ndimage import gaussian_filter

# %% Check that it was run and load the best models
figDir = "/home/jake/Data/Repos/V1FreeViewingCode/Figures/2021_pytorchmodeling"


# sessid = "20191119_kilowf"
# sessid = "20191120a_kilowf"
# sessid = "20191121_kilowf"
# sessid = "20191122_kilowf"
# sessid = '20191205_kilowf'
sessid = "20200304_kilowf"

lengthscale = 1 # the default is 1
save_dir='../../checkpoints/v1calibration_ls{}'.format(lengthscale)

outfile = Path(save_dir) / sessid / 'best_shifter.p'
if outfile.exists():
    print("fit_shifter was already run. Loading [%s]" %sessid)
    tmp = pickle.load(open(str(outfile), "rb"))
    cids = tmp['cids']
    shifter = tmp['shifter']
    shifters = tmp['shifters']
    vernum = tmp['vernum']
    valloss = tmp['vallos']
    num_lags = tmp['numlags']
    t_downsample = tmp['tdownsample']
else:
    print("foveal_stas: need to run fit_shifter for %s" %sessid)


#%% LOAD DATASETS with shifter

cropidxs = {'20191119_kilowf': [[20,50],[0,30]],
    '20191120a_kilowf': [[20,50],[20,50]],
    '20191121_kilowf': [[20,50],[20,50]],
    '20191122_kilowf': [[10,50],[20,50]],
    '20191202_kilowf': [[10,50], [15,45]],
    '20191205_kilowf': [[10,50],[15,45]],
    '20191206_kilowf': [[10,50],[15,45]],
    '20200304_kilowf': [[10,40],[15,50]]}

import V1FreeViewingCode.models.datasets as dd
import importlib
importlib.reload(dd)

cropidx = cropidxs[sessid]

n = 40
num_basis = 15
B = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,n)), axis=1) - np.arange(0,n,n/num_basis))/n*num_basis, 0)


# Load Training / Evaluation
gd = dd.PixelDataset(sessid, stims=["Dots", "Gabor", "BackImage", "Grating", "FixRsvpStim"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    shifter=shifters[np.argmin(valloss)],
    cropidx=cropidx,
    include_frametime={'num_basis': 40, 'full_experiment': False},
    include_saccades=[{'name':'sacon', 'basis':B, 'offset':-20}, {'name':'sacoff', 'basis':B, 'offset':0}],
    include_eyepos=True,
    optics={'type': 'gausspsf', 'sigma': (0.7, 0.7, 0.0)},
    preload=True)

#%% Load test set
# print("Loading Test set")
# gd_test = dd.PixelDataset(sessid, stims=["Dots", "Gabor", "BackImage"], #, "Grating", "FixRsvpStim"],
#     stimset="Test", num_lags=num_lags,
#     downsample_t=t_downsample,
#     downsample_s=1,
#     valid_eye_rad=5.2,
#     cropidx=cropidx,
#     shifter=shifter,
#     include_frametime={'num_basis': 40, 'full_experiment': False},
#     include_saccades=[{'name':'sacon', 'basis':B, 'offset':-20}, {'name':'sacoff', 'basis':B, 'offset':0}],
#     include_eyepos=True,
#     optics={'type': 'gausspsf', 'sigma': (0.7, 0.7, 0.0)},
#     preload=True)


#%% get sample
sample = gd[:]
im = sample['stim'].detach().clone()

#%% new STAs on shifted stimulus
crop = [0, gd.NX, 0, gd.NY]
# crop = [10,50,20,50]
stas = torch.einsum('nlwh,nc->lwhc', im[:,:,crop[2]:crop[3], crop[0]:crop[1]], sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()

#%%
extent = np.round(gd.rect/gd.ppd*60)
extent = np.asarray([extent[i] for i in [0,2,3,1]])

# plotting
NC = sta.shape[3]
sx = np.ceil(np.sqrt(NC))
sy = np.round(np.sqrt(NC))

plt.figure(figsize=(sx*2, sy*4))

Ispat = 0
for cc in range(NC):
    plt.subplot(sx, sy, cc+1)
    
    w = sta[:,:,:,cc]    

    bestlag = np.argmax(np.std(w.reshape( (gd.num_lags, -1)), axis=1))
    
    w = (w -np.mean(w[bestlag,:,:]) )/ np.std(w)

    v = np.max(np.abs(w))
    plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm")
    plt.title("%d" %cc)
    plt.xlabel("arcmin")
    plt.ylabel("arcmin")

    Ispat += np.abs(w[bestlag,:,:])

    # plt.subplot(sx, sy, cc*2+2)
    # i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
    # plt.plot(w[:,i[0], j[0]], '-ob')
    # i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
    # plt.plot(w[:,i[0], j[0]], '-or')
    # yd = plt.ylim()
    # plt.xlabel("Lag (frame=8ms)")

#%%
plt.figure()
plt.imshow(Ispat)

#%%
import importlib
importlib.reload(cores)

# path to where checkpointing will occur
iteration = 1
save_dir='./checkpoints/test_v1fovea_nim{}'.format(iteration)

from pathlib import Path

importlib.reload(ut)

#%%

for version in range(10,12): #range(3): #[18]: # fit a few runs
    input_size = [gd.num_lags, gd.NY, gd.NX]
    num_channels = [NC//2 + (4 - NC//2 % 4)] # number of subunits in each layer
    # num_channels = [15, 10] # number of subunits in each layer

    # shared NIM core:
    core = cores.NimCore(input_size,
            num_channels, # this number of subunits
            ei_split=[.1], # percent that are inhibitory
            weight_norm=[True], # use weight norm when fitting
            act_funcs=["relu"], # offset for ELU nonlinearity (TODO: change how we handle activations to be more like the 2D models)
            gamma_group=0,
            group_norm=[True],
            group_norm_num=1, # if using groupnorm, number of groups
            bias=[False], # include bias parameters
            weight_regularizer="RegMats",
            weight_reg_types=[["d2xt"]],
            weight_reg_amt=[[10e-8]],
            stack=None, # None outputs all layers, otherwise spedcify which layers you want (if making a scaffold)
            use_avg_reg=False) # mean instead of sum when regularizing with group lasso

    # full readout
    readout = readouts.FullReadout(core.outchannels, NC,
            bias=True,
            constrain_positive=True, # this is a EI NIM
            weight_norm=False,
            gamma_readout=0.01) # group lasso reg?

    modifiers = {'stimlist': ['frametime', 'sacoff'],
                'gain': [sample['frametime'].shape[1], sample['sacoff'].shape[1]],
                'offset':[sample['frametime'].shape[1],sample['sacoff'].shape[1]],
                'stage': "readout",
                'outdims': gd.NC}

        # combine core and readout into model
    model = encoders.EncoderMod(core, readout, modifiers=modifiers,
        gamma_mod=0,
        weight_decay=.1, optimizer='AdamW', learning_rate=.01, # high initial learning rate because we decay on plateau
        betas=[.9, .999], amsgrad=False)


    # check that the forward runs (for debugging)
    # sample = gd[:10]
    # out = model(sample['stim'])
    # print(out.shape)
    
    #% get trainer and train/valid data loaders (data loaders are iteratable datasets)
    trainer, train_dl, valid_dl = ut.get_trainer(gd, version=version,
            save_dir=save_dir,
            num_workers=64,
            name=gd.id,
            auto_lr=False,
            gradient_clip_val=0,
            earlystopping=True,
            earlystoppingpatience=50,
            batchsize=1000)

    trainpath = Path(save_dir) / gd.id / "version_{}".format(version)
    if not trainpath.exists():
        trainer.fit(model, train_dl, valid_dl)
    # model.hparams.weight_decay = model.hparams.weight_decay/10
    
    # model2.core.hparams.gamma_group = 1
    # trainer.fit(model, train_dl, valid_dl)

print("Done Fitting")


#%% load best model

import pandas as pd
from pathlib import Path
import seaborn as sns

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

plt.plot(ind, valloss, '.')
plt.xlabel("Version")
plt.ylabel("Validation Loss")

ver = ind[np.argmin(valloss)]
print("Best version: %d" %ver)


ver = "version_" + str(int(2))
chkpath = pth / ver / 'checkpoints'
best_epoch = ut.find_best_epoch(chkpath)


model2 = encoders.EncoderMod.load_from_checkpoint(str(chkpath / best_epoch))

# you have to remove weight norm after loading to have the filter weights in the "weights" variable
# weight_norm works by splitting the weights into a unit vector and an amplitude.
# these need to be recombined for plotting and whatnot
for l in range(model2.core.layers):
    if model2.core.weight_norm[l]:
        nn.utils.remove_weight_norm(model2.core.features[l].conv)


model2.core.plot_filters()
#%% plot filters
wsub = model2.core.features.layer0.conv.weight.detach().cpu().numpy()
wreadout = model2.readout.features.weight.detach().cpu().numpy()
f = plt.imshow(wreadout.T)
#%%
# subs = np.where(np.sum(wreadout, axis=0)>2)[0]
# model2.core.plot_filters(sort=subs, cmaps=[plt.cm.coolwarm, plt.cm.gray]) # plot filters and 

#%% select stimulus


# %%
def get_null_adjusted_ll(model, sample=None, bits=False, use_shifter=True):
        '''
        get null-adjusted log likelihood
        bits=True will return in units of bits/spike
        '''
        m0 = model.cpu()
        loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')

        lnull = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy().sum(axis=0)
        if use_shifter:
            yhat = m0(sample['stim'], shifter=sample['eyepos'], sample=sample)
        else:
            yhat = m0(sample['stim'], sample=sample)
        llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
        rbar = sample['robs'].sum(axis=0).numpy()
        ll = (llneuron - lnull)/rbar
        if bits:
            ll/=np.log(2)
        return ll



#%% Evaluate test likelihood

print("Select Stim")
    # select stimulus
# if 'Dots' in gd.stims:
#     stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Dots' == s][0]
# elif 'Gabor' in gd.stims:
#     stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Gabor' == s][0]

stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Gabor' == s][0]    
    

ll0 = get_null_adjusted_ll(model2, sample=sample, bits=True)
plt.plot(ll0)

# sample = gd[:]
index = np.where(gd.stim_indices==stimuse)[0]
sample = gd[index]

# NT = 10*gd.NY*gd.NX*gd.num_lags
# X = np.random.randn(NT, gd.NY, gd.NX)*10
# # X[np.abs(X)<5] = 0
# inds = np.expand_dims(np.arange(gd.num_lags,NT), axis=1)-range(gd.num_lags)
# X = torch.tensor(X[inds,:,:].astype('float32'))

yhat = model2(sample['stim'], sample=sample)

#%%
gd = dd.PixelDataset(sessid, stims=["Gabor"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    shifter=shifters[np.argmin(valloss)],
    include_eyepos=True,
    optics={'type': 'gausspsf', 'sigma': (0.7, 0.7, 0.0)},
    preload=True)

#%%
sample = gd[:]
stas = torch.einsum('nlwh,nc->lwhc', sample['stim'], sample['robs']-sample['robs'].mean(dim=0))
staHat = torch.einsum('nlwh,nc->lwhc', sample['stim'], yhat-yhat.mean(dim=0))

#%%
# reload sample
sample = gd[:]
yhat = model2(sample['stim'], sample=sample)
gd.stims
out = {'stas': stas.detach().cpu().numpy(),
    'stasHat': staHat.detach().cpu().numpy(),
    'yhat': yhat.detach().cpu().numpy(),
    'stims': gd.stims,
    'stimid': gd.stim_indices,
    'frametime': gd.frame_time,
    'eyepos': gd.eyepos,
    'll0': ll0,
    'wreadout': wreadout,
    'wsubunits': wsub}

from scipy.io import savemat
fname = save_dir + '/' + gd.id  + '_model2.mat'
print("Saving [%s]" %fname)
savemat(fname, out)
print("Done")

#%%
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.imshow(wreadout.T)
# plt.subplot(2,1,2)
plt.plot(ll0*20, 'r')
cc = 0
# %%

cc +=1
if cc >= gd.NC:
    cc = 0
# cc = 13


plt.figure(figsize=(5,2))
plt.subplot(1,3,1)
sta = stas[:,:,:,cc].detach().cpu().numpy()
bestlag = np.argmax(np.std(sta.reshape((sta.shape[0], -1)), axis=1))
plt.imshow(sta[bestlag, :,:], cmap=plt.cm.coolwarm)
plt.subplot(1,3,2)
plt.plot(wreadout[cc,:])
plt.axhline(.1)
subunits = np.where(wreadout[cc,:]>0.1)[0]
plt.subplot(1,3,3)
sta = staHat[:,:,:,cc].detach().cpu().numpy()
bestlag = np.argmax(np.std(sta.reshape((sta.shape[0], -1)), axis=1))
plt.imshow(sta[bestlag, :,:], cmap=plt.cm.coolwarm)
plt.title(ll0[cc])
n = len(subunits)
print("%d subunits" %n)

if n > 0:
    scol = np.ceil(np.sqrt(n))
    srows = np.round(np.sqrt(n))

    plt.figure(figsize=(scol*1, srows*1))

    vs = np.asarray([np.max(wsub[s,:,:,:]) for s in subunits])
    v = np.max(vs*wreadout[cc,subunits])
    rwts = wreadout[cc,subunits]
    # wpow = np.std(wsub[subunits,:,:,:], axis=1)
    # v = np.max(wpow)*np.max(rwts)
    rf = 0
    for isub in range(n):
        sub = subunits[isub]
        plt.subplot(srows, scol, isub + 1)
        
        wtmp = np.std(wsub[sub,:], axis=0)*rwts[isub]
        rf += wtmp
        # plt.imshow(wtmp, vmin=-v, vmax=v)
        wtmp = wsub[sub,:]*wreadout[cc,sub]
        bestlag = np.argmax(np.std(wtmp.reshape((wtmp.shape[0], -1)), axis=1))
        plt.imshow(wtmp[bestlag,:,:],vmin=-v, vmax=v, cmap=plt.cm.coolwarm, interpolation=None)
        # plt.imshow(wtmp[bestlag,:,:], cmap=plt.cm.coolwarm, interpolation=None)

# wtmp = np.squeeze(wsub[sub,:])
# plt.figure(figsize=(15,15))
# rf = rf/np.max(rf)
# rf[rf<.5]=0.0
# plt.imshow(rf, interpolation='none', cmap=plt.cm.coolwarm)
# plt.colorbar()
# w[:,cc]
# stas[:,:]

#%%

# import
sub = 0
wtmp = wsub[sub,:]*wreadout[cc,sub]
bestlag = np.argmax(np.std(wtmp.reshape((wtmp.shape[0], -1)), axis=1))


#%%
i += 1
if i > wsub.shape[0]:
    i = 0
wtmp = wsub[i,:]
plt.plot(wtmp[bestlag,:].flatten())

#%%
plt.imshow(wtmp[bestlag,:,:])

# %%



# %%

# %%
