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

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


from rfest import splineLG, splineLNP, splineLNLN
from rfest.utils import fetch_data, split_data, upsample_data
from rfest.utils import get_n_samples, build_design_matrix
from rfest.plot import plot3d, plot_subunits3d, plot_nonlinearity

# %% Check that it was run and load the best models
figDir = "/home/jake/Data/Repos/V1FreeViewingCode/Figures/2021_pytorchmodeling"

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
    shifters = tmp['shifters']
    vernum = tmp['vernum']
    valloss = tmp['vallos']
    num_lags = tmp['numlags']
    t_downsample = tmp['tdownsample']
else:
    print("foveal_stas: need to run fit_shifter for %s" %sessid)


#%% LOAD DATASETS with shifter

import V1FreeViewingCode.models.datasets as dd
import importlib
importlib.reload(dd)
gd = dd.PixelDataset(sessid, stims=["Dots", "Gabor"], #, "BackImage", "Grating", "FixRsvpStim"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    shifter=shifter,
    include_eyepos=True,
    preload=True)


#%% get sample
sample = gd[:]
im = sample['stim'].detach().clone()

#%% new STAs on shifted stimulus
crop = [20, 50, 0, 30]
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
    plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=extent)
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



#%% get data for fitting
f = nn.Flatten()
dt = 1/gd.frate
dims = sample['stim'][:,:,crop[2]:crop[3], crop[0]:crop[1]].shape[1:]
df = [i - 2 for i in dims]
stim = f(sample['stim'][:,0,crop[2]:crop[3], crop[0]:crop[1]]).detach().cpu().clone().numpy()
robs = sample['robs'].detach().cpu().numpy()
del gd


# dims = [25, 20, 15]

#%%
from rfest import ALD
#%%
robs.shape
#%%
# stim.shape

X = build_design_matrix(stim, dims[0])
X.shape
#%%
import copy
dt = 1/240

df = [5, 15, 15]
sigma0 = [1.]
rho0 = [1.]
params_t0 = [3., 20., 1., 1.] # taus, nus, tauf, nuf
params_y0 = [3., 20., 1., 1.]
params_x0 = [3., 20., 1., 1.]
p0 = sigma0 + rho0 + params_t0 + params_y0 + params_x0


lsgs = []
for cc in [13]:
    
    
    y = robs[:,cc]
    y = gaussian_filter(y, sigma=2, truncate=0.25)
    # (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = split_data(X[:10000,:], y[:10000], dt, frac_train=0.8, frac_dev=0.1)
    
    #%%
    # lnp = splineLNP(X_train, y_train, dims, df, dt=dt, nonlinearity='exponential')
    # lg = splineLG(X_train, y_train, dims=dims, dt=dt, df=df)
    # lg = splineLG(X, y, dims=dims, dt=dt, df=df)
    # lg.fit(metric='corrcoef',
    #       num_iters=3000, verbose=0, tolerance=10,
    #       beta=0.01)

    # lgs.append(copy.deepcopy(lg))
    # lnp.fit(extra={'X': X_dev, 'y': y_dev}, metric='corrcoef',
    #     num_iters=300, verbose=300, tolerance=10,
    #     beta=0.01)

    ald = ALD(X[:5000], y[:5000], dims=dims)
    ald.fit(p0=p0, num_iters=30, verbose=10)
    lgs.append(copy.deepcopy(ald))
#%%
plot3d(lg, X,y, len_time=1, response_type='spike')


#%%

# select stimulus
# if 'Dots' in gd.stims:
#     stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Dots' == s][0]
# elif 'Gabor' in gd.stims:
#     stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Gabor' == s][0]


# index = np.where(gd.stim_indices==stimuse)[0]
sample = gd[:] # load sample 


# %% save everything out for matlab
import scipy.io as sio
fname = figDir + "/rfs_" + gd.id + ".mat"
mdict = {'cids': cids, 'xspace': xx, 'yspace': yy, 'shiftx': y2[:,0].reshape((100,100)), 'shifty': y2[:,1].reshape((100,100)), 'stas_pre': sta, 'stas_post': sta2,
    'valloss': valloss, 'mushiftx': mux, 'mushifty': muy, 'sdshiftx': sdx, 'sdshifty': sdy}

sio.savemat(fname, mdict)


#%% plot all STAs as one figure

def plot_sta_fig(sta, gd):
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

plot_sta_fig(sta, gd)
plt.savefig(figDir + "/rawstas_" + gd.id + ".pdf", bbox_inches='tight')
plot_sta_fig(sta2, gd)
plt.savefig(figDir + "/shiftstas_" + gd.id + ".pdf", bbox_inches='tight')
# plt.close('all')


#%%
# from V1FreeViewingCode.Analysis.manuscript_freeviewingmethods.v1_tracker_calibration import make_figures

# make_figures(gd, tmp, figDir)

#%%

# #%% Refit DivNorm Model on all datasets
# #% This is optional
# import V1FreeViewingCode.models.encoders as encoders
# importlib.reload(encoders)
# importlib.reload(readouts)
# importlib.reload(cores)

# # #%%
# # """
# # Fit single layer DivNorm model with modulation
# # """
# # for version in range(5): # range of version numbers
    

# #     #% Model: convolutional model
# #     input_channels = gd.num_lags
# #     hidden_channels = 16
# #     input_kern = 19
# #     hidden_kern = 5
# #     core = cores.Stacked2dDivNorm(input_channels,
# #             hidden_channels,
# #             input_kern,
# #             hidden_kern,
# #             layers=1,
# #             gamma_hidden=1e-6, # group sparsity
# #             gamma_input=1,
# #             gamma_center=0,
# #             skip=0,
# #             final_nonlinearity=True,
# #             bias=False,
# #             pad_input=True,
# #             hidden_padding=hidden_kern//2,
# #             group_norm=True,
# #             num_groups=4,
# #             weight_norm=True,
# #             hidden_dilation=1,
# #             input_regularizer="RegMats",
# #             input_reg_types=["d2x", "center", "d2t"],
# #             input_reg_amt=[.000005, .01, 0.00001],
# #             hidden_reg_types=["d2x", "center"],
# #             hidden_reg_amt=[.000005, .01],
# #             stack=None,
# #             use_avg_reg=True)


# #     # initialize input layer to be centered
# #     regw = regularizers.gaussian2d(input_kern,sigma=input_kern//4)
# #     core.features[0].conv.weight.data = torch.einsum('ijkm,km->ijkm', core.features[0].conv.weight.data, torch.tensor(regw))
        
# #     # Readout
# #     in_shape = [core.outchannels, gd.NY, gd.NX]
# #     bias = True
# #     readout = readouts.Point2DGaussian(in_shape, gd.NC, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
# #                     gamma_l1=0,gamma_l2=0.00001,
# #                     align_corners=True, gauss_type='uncorrelated',
# #                     constrain_positive=False,
# #                     shifter= {'hidden_features': 20,
# #                             'hidden_layers': 1,
# #                             'final_tanh': False,
# #                             'activation': "softplus",
# #                             'lengthscale': lengthscale}
# #                             )

    
# #     modifiers = {'stimlist': ['frametime', 'sacoff'],
# #             'gain': [sample['frametime'].shape[1], sample['sacoff'].shape[1]],
# #             'offset':[sample['frametime'].shape[1],sample['sacoff'].shape[1]],
# #             'stage': "readout",
# #             'outdims': gd.NC}

# #     # combine core and readout into model
# #     model = encoders.EncoderMod(core, readout, modifiers=modifiers,
# #         gamma_mod=0,
# #         weight_decay=.001, optimizer='AdamW', learning_rate=.01, # high initial learning rate because we decay on plateau
# #         betas=[.9, .999], amsgrad=False)

# #     # initialize readout based on spike rate and STA centers
# #     model.readout.bias.data = sample['robs'].mean(dim=0) # initialize readout bias helps
# #     model.readout._mu.data[0,:,0,:] = torch.tensor(mu.astype('float32')) # initiaalize mus

# #     #% check that the forward runs (for debugging)
    

# #     #% Train
# #     trainer, train_dl, valid_dl = ut.get_trainer(gd, version=version,
# #                 save_dir=save_dir,
# #                 name=gd.id,
# #                 auto_lr=False,
# #                 batchsize=1000,
# #                 num_workers=64,
# #                 earlystopping=False)

# #     trainpath = Path(save_dir) / gd.id / "version_{}".format(version)
# #     if not trainpath.exists():
# #         trainer.fit(model, train_dl, valid_dl)
    
# #%%
# # Load best version
# pth = Path(save_dir) / sessid

# valloss = np.array([])
# ind = np.array([])
# for v in pth.rglob('version*'):

#     try:
#         df = pd.read_csv(str(v / "metrics.csv"))
#         ind = np.append(ind, int(v.name.split('_')[1]))
#         valloss = np.append(valloss, np.nanmin(df.val_loss.to_numpy()))
#     except:
#         "skip"

# sortind = np.argsort(ind)
# ind = ind[sortind]
# valloss = valloss[sortind]
# plt.plot(ind, valloss, '.')
# plt.xlabel("Version #")
# plt.ylabel("Loss")


# shifters = nn.ModuleList()
# for vernum in ind:
#     # load linear model version
#     ver = "version_" + str(int(vernum))
#     chkpath = pth / ver / 'checkpoints'
#     best_epoch = ut.find_best_epoch(chkpath)
#     model2 = encoders.EncoderMod.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))
#     shifters.append(model2.readout.shifter.cpu())



# #%% load best model
# vernum = ind[np.argmin(valloss)]
# print("Loading version %d" %vernum)
# ver = "version_" + str(int(vernum))
# chkpath = pth / ver / 'checkpoints'
# best_epoch = ut.find_best_epoch(chkpath)
# model = encoders.EncoderMod.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))
# nn.utils.remove_weight_norm(model.core.features[0].conv)

# model.core.plot_filters()
# #%% load data
# # path='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'

# # # build tent basis for saccades
# # n = 40
# # num_basis = model.core.input_channels
# # B = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,n)), axis=1) - np.arange(0,n,n/num_basis))/n*num_basis, 0)
# # num_lags = model.core.input_channels

# # # re-load dataset to fit shifter
# # gd = dd.PixelDataset(sessid, stims=["Gabor"],
# #     stimset="Train", num_lags=num_lags,
# #     downsample_t=2,
# #     downsample_s=1,
# #     valid_eye_rad=5.2,
# #     cids=cids,
# #     include_eyepos=True,
# #     dirname=path,
# #     include_frametime={'num_basis': 40, 'full_experiment': False},
# #     include_saccades=[{'name':'sacon', 'basis':B, 'offset':-20}, {'name':'sacoff', 'basis':B, 'offset':0}],
# #     preload=True)


# # %%
# # model.core.plot_filters()

# # %% shift stimulus
# def shift_stim(im, shift, gd):
#     import torch.nn.functional as F
#     affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
#     aff = torch.tensor([[1,0,0],[0,1,0]])

#     affine_trans = shift[:,:,None]+aff[None,:,:]
#     affine_trans[:,0,0] = 1
#     affine_trans[:,0,1] = 0
#     affine_trans[:,1,0] = 0
#     affine_trans[:,1,1] = 1

#     n = im.shape[0]
#     grid = F.affine_grid(affine_trans, torch.Size((n, gd.num_lags, gd.NY,gd.NX)), align_corners=True)

#     im2 = F.grid_sample(im, grid, mode='bilinear', align_corners=True)
#     return im2


# if 'Dots' in gd.stims:
#     stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Dots' == s][0]
# elif 'Gabor' in gd.stims:
#     stimuse = [i for i,s in zip(range(len(gd.stims)), gd.stims) if 'Gabor' == s][0]

# index = np.where(gd.stim_indices==stimuse)[0]
# sample = gd[index] # load sample 

# # run shifter
# shift = model.readout.shifter(sample['eyepos']).detach() # shifters[1](sample['eyepos']).detach()

# im = sample['stim'].detach().clone()


# #%% new STAs on shifted stimulus
# stas = torch.einsum('nlwh,nc->lwhc', im, sample['robs']-sample['robs'].mean(dim=0))
# sta = stas.detach().cpu().numpy()
# stas = torch.einsum('nlwh,nc->lwhc', im2, sample['robs']-sample['robs'].mean(dim=0))
# sta2 = stas.detach().cpu().numpy()
# # plt.close('all')
# #%%
# extent = np.round(gd.rect/gd.ppd*60)
# extent = np.asarray([extent[i] for i in [0,2,3,1]])

# # plotting
# NC = sta.shape[3]
# # plt.figure(figsize=(8,NC*2))
# for cc in range(NC):
#     plt.figure(figsize=(8,2))
    
#     w = sta[:,:,:,cc]    
#     w2 = sta2[:,:,:,cc]

#     bestlag = np.argmax(np.std(w2.reshape( (gd.num_lags, -1)), axis=1))
    
#     w = (w -np.mean(w[bestlag,:,:]) )/ np.std(w)
#     w2 = (w2 -np.mean(w2[bestlag,:,:]) )/ np.std(w2)


#     plt.subplot(1,4,3)
#     # plt.subplot(NC,4, cc*4 + 1)
#     v = np.max(np.abs(w2))
#     plt.imshow(w2[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=extent)
#     plt.title("After")
#     plt.xlabel("arcmin")
#     plt.ylabel("arcmin")
#     # plt.subplot(NC,4, cc*4 + 2)
#     plt.subplot(1,4,4)
#     i,j=np.where(w2[bestlag,:,:]==np.max(w2[bestlag,:,:]))
#     plt.plot(w2[:,i[0], j[0]], '-ob')
#     i,j=np.where(w2[bestlag,:,:]==np.min(w2[bestlag,:,:]))
#     plt.plot(w2[:,i[0], j[0]], '-or')
#     yd = plt.ylim()
#     # sns.despine(offset=0, trim=True)
#     plt.xlabel("Lag (frame=8ms)")

#     # plt.subplot(NC,4, cc*4 + 3)
#     plt.subplot(1,4,1)
#     plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=extent)
#     plt.title("Before")
#     plt.xlabel("arcmin")
#     plt.ylabel("arcmin")

#     plt.subplot(1,4,2)
#     # plt.subplot(NC,4, cc*4 + 4)
#     i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
#     plt.plot(w[:,i[0], j[0]], '-ob')
#     i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
#     plt.plot(w[:,i[0], j[0]], '-or')
#     plt.axvline(bestlag, color='k', linestyle='--')
#     plt.ylim(yd)
#     # sns.despine(offset=0, trim=True)
#     plt.xlabel("Lag (frame=8ms)")
#     # plt.show()
#     plt.savefig(figDir + "/sta_shift" + gd.id + "_" + str(cc) + ".pdf", bbox_inches='tight')
#     plt.close('all')
# # %% save everything out for matlab
# import scipy.io as sio
# fname = figDir + "/rfs_" + gd.id + ".mat"
# mdict = {'xspace': xx, 'yspace': yy, 'shiftx': y2[:,0].reshape((100,100)), 'shifty': y2[:,1].reshape((100,100)), 'stas_pre': sta, 'stas_post': sta2}
# sio.savemat(fname, mdict)

# # %%

# %%
