#%% big import
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import copy
import os

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt  # plotting
import seaborn as sns

from V1FreeViewingCode.models.datasets import PixelDataset
from torch.utils.data import Dataset, DataLoader, random_split

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import NDN3.NDNutils as NDNutils

which_gpu = NDNutils.assign_gpu()

import tensorflow as tf

import NDN3.NDN as NDN
import NDN3.Utils.DanUtils as DU

# output_dir = '/home/jake/Data/tensorboard/tensorboard' + str(0)
# print(output_dir)
#%% Load dataset
num_lags = 10
t_downsample = 2 # temporal downsampling
sessid = '20200304'
gd = PixelDataset(sessid, stims=["Gabor"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    include_eyepos=True,
    preload=True)

#%% compute STAS
"""
Compute STAS using einstein summation through pytorch
"""
sample = gd[:] # load sample 

stas = torch.einsum('nlwh,nc->lwhc', sample['stim'], sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()

#%% plot STAs / get RF centers
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

plt.figure(figsize=(12,10))
for cc in range(NC):
    w = sta[:,:,:,cc]

    wt = np.std(w, axis=0)
    wt /= np.max(np.abs(wt)) # normalize for numerical stability
    # softmax
    wt = wt**50
    wt /= np.sum(wt)
    sz = wt.shape
    xx,yy = np.meshgrid(np.linspace(-1, 1, sz[1]), np.linspace(1, -1, sz[0]))

    mu[cc,0] = np.minimum(np.maximum(np.sum(xx*wt), -.1), .1) # center of mass after softmax
    mu[cc,1] = np.minimum(np.maximum(np.sum(yy*wt), -.1), .1) # center of mass after softmax

    w = (w -np.mean(w) )/ np.std(w)

    bestlag = np.argmax(np.std(w.reshape( (gd.num_lags, -1)), axis=1))
    plt.subplot(sx,sy, cc*2 + 1)
    v = np.max(np.abs(w))
    plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=(-1,1,-1,1))
    plt.plot(mu[cc,0], mu[cc,1], '.b')
    plt.title(cc)
    plt.subplot(sx,sy, cc*2 + 2)
    i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-ob')
    i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-or')
    yd = plt.ylim()

#%% import models
from V1FreeViewingCode.models.encoders import Encoder
import V1FreeViewingCode.models.cores as cores
import importlib
importlib.reload(cores)
import V1FreeViewingCode.models.readouts as readouts
importlib.reload(readouts)

import V1FreeViewingCode.models.regularizers as regularizers
importlib.reload(regularizers)
import V1FreeViewingCode.models.utils as ut
importlib.reload(ut)
lengthscale = 1
save_dir='./core_shifter_ls{}'.format(lengthscale)
from pathlib import Path
# %% Fit LN model with shifter 

"""
Fit LN model with shifter 10 times.
"""
for version in range(1,11): # range of version numbers
    
    print("Version %d" %version)
    #% Model: convolutional model
    input_channels = gd.num_lags
    hidden_channels = 16
    input_kern = 15
    hidden_kern = 19
    core = cores.Stacked2dDivNorm(input_channels,
            hidden_channels,
            input_kern,
            hidden_kern,
            layers=1,
            gamma_hidden=0, # group sparsity
            gamma_input=1,
            gamma_center=0,
            skip=0,
            final_nonlinearity=False,
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
            stack=None,
            use_avg_reg=True)


    # initialize input layer to be centered
    regw = regularizers.gaussian2d(input_kern,sigma=input_kern//4)
    core.features[0].conv.weight.data = torch.einsum('ijkm,km->ijkm', core.features[0].conv.weight.data, torch.tensor(regw))
        
    # Readout
    in_shape = [core.outchannels, gd.NY, gd.NX]
    bias = True
    readout = readouts.Point2DGaussian(in_shape, gd.NC, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
                    align_corners=True, gauss_type='uncorrelated',
                    constrain_positive=False,
                    shifter= {'hidden_features': 20,
                            'hidden_layers': 1,
                            'final_tanh': False,
                            'activation': "softplus",
                            'lengthscale': lengthscale}
                            )

    # combine core and readout into model
    model = Encoder(core, readout,
        weight_decay=.001, optimizer='AdamW', learning_rate=.01, # high initial learning rate because we decay on plateau
        betas=[.9, .999], amsgrad=True)

    # initialize readout based on spike rate and STA centers
    model.readout.bias.data = sample['robs'].mean(dim=0) # initialize readout bias helps

    #% check that the forward runs (for debugging)
    out = model(sample['stim'][:10,:], shifter=sample['eyepos'][:10,:])
    # out.shape

    #% Train
    trainer, train_dl, valid_dl = ut.get_trainer(gd, version=version,
                save_dir=save_dir,
                name=gd.id,
                auto_lr=False,
                batchsize=1000)
    
    trainpath = Path(save_dir) / gd.id / "version_{}".format(version)
    if not trainpath.exists():
        trainer.fit(model, train_dl, valid_dl)
    else:
        print("Already run. Skipping.")

#%%
"""
Fit DivNorm model with shifter 10 times.
"""
for version in range(11,21): # range of version numbers
    

    #% Model: convolutional model
    input_channels = gd.num_lags
    hidden_channels = 16
    input_kern = 15
    hidden_kern = 19
    core = cores.Stacked2dDivNorm(input_channels,
            hidden_channels,
            input_kern,
            hidden_kern,
            layers=2,
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
            input_reg_amt=[.000005, .001, 0.00001],
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
                    align_corners=True, gauss_type='uncorrelated',
                    constrain_positive=False,
                    shifter= {'hidden_features': 20,
                            'hidden_layers': 1,
                            'final_tanh': False,
                            'activation': "softplus",
                            'lengthscale': lengthscale}
                            )

    # combine core and readout into model
    model = Encoder(core, readout,
        weight_decay=.001, optimizer='AdamW', learning_rate=.01, # high initial learning rate because we decay on plateau
        betas=[.9, .999], amsgrad=True)

    # initialize readout based on spike rate and STA centers
    model.readout.bias.data = sample['robs'].mean(dim=0) # initialize readout bias helps
    # model.readout._mu.data[0,:,0,:] = torch.tensor(mu.astype('float32')) # initiaalize mus

    #% check that the forward runs (for debugging)
    out = model(sample['stim'][:10,:], shifter=sample['eyepos'][:10,:])
    # out.shape

    #% Train
    trainer, train_dl, valid_dl = ut.get_trainer(gd, version=version,
                save_dir=save_dir,
                name=gd.id,
                auto_lr=False,
                batchsize=1000)

    trainpath = Path(save_dir) / gd.id / "version_{}".format(version)
    if not trainpath.exists():
        trainer.fit(model, train_dl, valid_dl)

#%% get the top 5 versions that have been run
 
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

#% plot best linear shifter and best LNLN shifter
from matplotlib.patches import ConnectionPatch

# build inputs for shifter plotting
xx,yy = np.meshgrid(np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,100),np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,100))
xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

inputs = torch.cat( (xgrid,ygrid), dim=1)

# open figure
fig,ax = plt.subplots(figsize=(5,3))
ax.set_axis_off()
axins = ax.inset_axes([0.1, 0.1, 0.27, 0.85])

#--- axis 1: plot 
n1 = sum(ind < 11)
x = np.ones(n1)+np.random.rand(n1)*.1
y = valloss[ind < 11]
axins.plot(x, y, '.', alpha=.5, color="tab:cyan")
mn = np.argmin(y)
ver1 = ind[mn]
axins.plot(x[mn], y[mn], 'o', alpha=1, color="tab:cyan")

n = sum(ind > 10)
x = 2*np.ones(n)+np.random.rand(n)*.1
y = valloss[ind>10]
axins.plot(x, y, '.', alpha=.5, color="tab:green")
mn = np.argmin(y)
ver2 = ind[mn+n1-1]
axins.plot(x[mn], y[mn], 'o', alpha=1, color="tab:green")

axins.set_xlim((.5, 2.5))
axins.set_xticks([1,2])
axins.set_xticklabels(['LN', 'LNLN'])
axins.set_yticks(np.arange(.135,.152,.005))

axins.tick_params(labelsize=8)

sns.despine(ax=axins, trim=True, offset=0)
axins.set_xlabel("Model", fontsize=8)
axins.set_ylabel("Loss", fontsize=8)

# load linear model version
ver = "version_" + str(int(ver1))
chkpath = pth / ver / 'checkpoints'
best_epoch = ut.find_best_epoch(chkpath)
model2 = Encoder.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))
shifter1 = model2.readout.shifter
shifter1.cpu()
y = shifter1(inputs)
y2 = y.detach().cpu().numpy()
# y2 = y2 - np.mean(y2, axis=0)
y2/=gd.valid_eye_rad/60 # conver to arcmin
vmin = np.min(y2)
vmax = np.max(y2)

x0 = 0.425
x1 = 0.675
y0 = 0.15
y1 = 0.6
axins1 = ax.inset_axes([x0, y1, 0.3, 0.35])
axins1.imshow(y2[:,0].reshape((100,100)), extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
axins1.set_xticklabels([])
axins1.tick_params(labelsize=8)
axins1.set_title("Horizontal", fontsize=8)

axins2 = ax.inset_axes([x1, y1, 0.3, 0.35])
axins2.imshow(y2[:,1].reshape((100,100)), extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
axins2.set_yticklabels([])
axins2.set_xticklabels([])
axins2.set_title("Vertical", fontsize=8)

# trans_angle = plt.gca().transData.transform_angles(np.array((45,)),
#                                                    l2.reshape((1, 2)))[0]

# --- Load LNLN version
ver = "version_" + str(int(ver2))
chkpath = pth / ver / 'checkpoints'
best_epoch = ut.find_best_epoch(chkpath)
model2 = Encoder.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))
shifter2 = model2.readout.shifter
shifter2.cpu()
y = shifter2(inputs)
y2 = y.detach().cpu().numpy()
# y2 = y2 - np.mean(y2, axis=0)
y2/=gd.valid_eye_rad/60 # conver to arcmin

axins3 = ax.inset_axes([x0, y0, 0.3, 0.35])
im = axins3.imshow(y2[:,0].reshape((100,100)), extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
axins3.tick_params(labelsize=8)

axins4 = ax.inset_axes([x1, y0, 0.3, 0.35])
axins4.imshow(y2[:,1].reshape((100,100)), extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
axins4.set_yticklabels([])
axins4.tick_params(labelsize=8)
th2 = ax.text(x0-.035, y0+.1, 'Vertical Eye Position (d.v.a.)', fontsize=8,
               rotation=90, rotation_mode='anchor')

th2 = ax.text(x0+.05, y0-.18, 'Horizontal Eye Position (d.v.a.)', fontsize=8,
               rotation=0, rotation_mode='anchor')

axcbar = ax.inset_axes([.95, y0, 0.015, 0.8])
cbar = fig.colorbar(im, cax=axcbar, orientation="vertical")
axcbar.tick_params(labelsize=7)
axcbar.set_ylabel("Shift (Arcmin)", fontsize=8)
# cmap = plt.cm.viridis()
# cb = plt.colorbar(ax=im)
# cb1 = mpl.colorbar.ColorbarBase(axins2, cmap=cmap,
                                # orientation='vertical')
# #%% ---- plot linear model

# ver = "version_" + str(int(ver1))
# chkpath = pth / ver / 'checkpoints'
# best_epoch = ut.find_best_epoch(chkpath)
# model2 = Encoder.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))
# nn.utils.remove_weight_norm(model2.core.features.layer0.conv)
# #%%
# model2.core.plot_filters()

# #%%

# wr = model2.readout.features.detach().squeeze().cpu()
# wc = model2.core.features.layer0.conv.weight.detach().cpu()

# wrf = torch.einsum('ntxy,nc->txyc', wc, wr).numpy()

# #%%
# cc = 7

# plt.figure(figsize=(10,5))
# for l in range(num_lags):
#     plt.subplot(1, num_lags, l+1)
#     plt.imshow(wrf[l,:,:,cc], aspect='auto')


#%% run each shifter to see what they predict
iix = np.arange(0,1000) + 2000
plt.figure(figsize=(10,5))
sample = gd[:]
shifters = []
shifters.append(shifter1)
shifters.append(shifter2)

bigshift = 0
for i in range(len(shifters)):
    shift = shifters[i](sample['eyepos']).detach()
    # shift = shift - shift.mean(dim=0)
    bigshift = bigshift + shift
    f = plt.plot(shift[iix,:]/gd.ppd*60)

bigshift = bigshift / len(shifters)
plt.plot(bigshift[iix,:]/gd.ppd*60, 'k--')
plt.xlabel("Time (frames)")
plt.ylabel("Shift (arcmin)")

#%% shift stimulus and plot improvement with average shift
import torch.nn.functional as F
affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])

aff = torch.tensor([[1,0,0],[0,1,0]])

# pick shifter (1 means use LNLN model)
shift = shifters[1](sample['eyepos']).detach()
affine_trans = shift[:,:,None]+aff[None,:,:]
affine_trans[:,0,0] = 1
affine_trans[:,0,1] = 0
affine_trans[:,1,0] = 0
affine_trans[:,1,1] = 1

im = sample['stim'].detach().clone()
n = im.shape[0]
grid = F.affine_grid(affine_trans, torch.Size((n, gd.num_lags, gd.NY,gd.NX)), align_corners=True)

im2 = F.grid_sample(im, grid)

#%% new STAs on shifted stimulus
stas = torch.einsum('nlwh,nc->lwhc', im, sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()
stas = torch.einsum('nlwh,nc->lwhc', im2, sample['robs']-sample['robs'].mean(dim=0))
sta2 = stas.detach().cpu().numpy()

# plotting
NC = sta.shape[3]
plt.figure(figsize=(8,NC*2))
for cc in range(NC):
    w = sta[:,:,:,cc]
    w = (w -np.mean(w) )/ np.std(w)
    w2 = sta2[:,:,:,cc]
    w2 = (w2 -np.mean(w2) )/ np.std(w2)

    bestlag = np.argmax(np.std(w2.reshape( (gd.num_lags, -1)), axis=1))
    plt.subplot(NC,4, cc*4 + 1)
    v = np.max(np.abs(w2))
    plt.imshow(w2[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm")
    plt.title(cc)
    plt.subplot(NC,4, cc*4 + 2)
    i,j=np.where(w2[bestlag,:,:]==np.max(w2[bestlag,:,:]))
    plt.plot(w2[:,i[0], j[0]], '-ob')
    i,j=np.where(w2[bestlag,:,:]==np.min(w2[bestlag,:,:]))
    plt.plot(w2[:,i[0], j[0]], '-or')
    yd = plt.ylim()

    plt.subplot(NC,4, cc*4 + 3)
    plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm")
    plt.subplot(NC,4, cc*4 + 4)
    i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-ob')
    i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-or')
    plt.ylim(yd)

#%% plot individual units
if cc >= NC:
    cc = 0
cc = 3
print(cc)

w = sta[:,:,:,cc]
w = (w - np.min(w)) / (np.max(w)-np.min(w))
w2 = sta2[:,:,:,cc]
w2 = (w2 - np.min(w2)) / (np.max(w2)-np.min(w2))
plt.figure(figsize=(20,4))
for i in range(w.shape[0]):
    plt.subplot(2,w.shape[0],i+1)
    plt.imshow(w[i,:,:], vmin=0, vmax=1, interpolation='None')
    plt.axis("off")
    plt.subplot(2,w.shape[0],w.shape[0]+i+1)
    plt.imshow(w2[i,:,:], vmin=0, vmax=1, interpolation='None')
    plt.axis("off")

cc +=1

#%% plot individual frame
fr = 2
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(w[fr,:,:], interpolation='Nearest', extent=(0,70/gd.ppd*60,0,70/gd.ppd*60), cmap="gray")
plt.xlabel("arcmin")
plt.ylabel("arcmin")

plt.subplot(122)
plt.imshow(w2[fr,:,:], interpolation='Nearest', extent=(0,70/gd.ppd*60,0,70/gd.ppd*60), cmap="gray")
plt.xlabel("arcmin")
plt.ylabel("arcmin")


#%% Load test set
gd_test = PixelDataset(sessid, stims=["Gabor"],
    stimset="Test", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    include_eyepos=True,
    preload=True,
    shifter=None)

# %%
def get_null_adjusted_ll(model, sample, bits=False, use_shifter=True):
    '''
    get null-adjusted log likelihood
    bits=True will return in units of bits/spike
    '''
    m0 = model.cpu()
    loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
    lnull = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy().sum(axis=0)
    if use_shifter:
        yhat = m0(sample['stim'], shifter=sample['eyepos'])
    else:
        yhat = m0(sample['stim'])
    llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
    rbar = sample['robs'].sum(axis=0).numpy()
    ll = (llneuron - lnull)/rbar
    if bits:
        ll/=np.log(2)
    return ll



#%% load best model

ver = "version_" + str(int(ver2))
chkpath = pth / ver / 'checkpoints'
best_epoch = ut.find_best_epoch(chkpath)
model2 = Encoder.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))

#%% plot div norm layer
# get filters
w = model2.core.features.layer0.conv.weight.detach().cpu().numpy()
sz = w.shape

wr = w.reshape(sz[0], sz[1], sz[2]*sz[3])
nfilt = w.shape[0]
bestlag = np.zeros(sz[0])
wIm = np.zeros((sz[0], sz[2], sz[3]))
for cc in range(sz[0]):
    wtmp = np.squeeze(wr[cc,:])
    bestlag[cc] = np.argmax(np.std(wtmp, axis=1))
    wIm[cc,:,:] = w[cc,int(bestlag[cc]),:,:]


nmat = model2.core.features.layer0.norm.weight.detach().cpu().numpy()

plt.imshow(nmat)

#%% remove weight norm
nn.utils.remove_weight_norm(model2.core.features.layer0.conv)
nn.utils.remove_weight_norm(model2.core.features.layer1.conv)
#%% add weight norm
nn.utils.weight_norm(model2.core.features.layer0.conv)
nn.utils.weight_norm(model2.core.features.layer1.conv)
#%% 
"""
PLOT MODEL: should be function

1. plot subunit layer
2. plot pooling layer
3. plot readout mus
4. plot shifter
5. plot test log-likelihood
"""

# 1. plot subunits
model2.core.plot_filters()

# 2. plot pooling
w = model2.core.features.layer1.conv.weight.detach().cpu().numpy()
plt.figure(figsize=(15,15))
for i in range(w.shape[0]):
    for j in range(w.shape[1]):
        plt.subplot(w.shape[0], w.shape[1], i*w.shape[0]+j+1)
        plt.imshow(w[i,j,:,:], aspect='auto')
        plt.title(i)
        plt.axis("off")

# 3. plot readout mus
plt.figure()
xy = model2.readout.mu.detach().cpu().numpy().squeeze()
plt.plot(xy[:,0], xy[:,1], '.')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Readout')

# 4. plot shifter
xx,yy = np.meshgrid(np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,100),np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,100))
xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

inputs = torch.cat( (xgrid,ygrid), dim=1)

shifter = model2.readout.shifter
shifter.cpu()
y = shifter(inputs)

y2 = y.detach().cpu().numpy()
y2/=gd.valid_eye_rad/60

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(y2[:,0].reshape((100,100)), extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None)
plt.title("Horizontal Shift (arcmin)")
plt.xlabel("Horizontal Eye position")
plt.ylabel("Vertical Eye position")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(y2[:,1].reshape((100,100)), extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None)
plt.title("Vertical shift (arcmin)")
plt.colorbar()
plt.xlabel("Horizontal Eye position")
plt.ylabel("Vertical Eye position")

#5. plot test LL
sample = gd_test[:]
l2 = get_null_adjusted_ll(model2, sample)

plt.figure()
plt.plot(l2, '-o')
plt.axhline(0, color='k')

#%% Reload dataset in restricted position
import V1FreeViewingCode.models.datasets as dd
import importlib
importlib.reload(dd)

gd_shift = dd.PixelDataset(sessid, stims=["Gabor"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    valid_eye_ctr=(0.0,0.0),
    include_eyepos=True,
    cropidx=[(15,50),(20,50)],
    shifter=shifters[1], #model2.readout.shifter,
    preload=True)

#%% test set
gd_shift_test = dd.PixelDataset(sessid, stims=["Gabor"],
    stimset="Test", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    valid_eye_ctr=(0.0,0.0),
    include_eyepos=True,
    cropidx=[(15,50),(20,50)],
    shifter=shifters[1],
    preload=True)
# %% reload sample and compute STAs
sample = gd_shift[:] # load sample 

# shifted
stas = torch.einsum('nlwh,nc->lwhc', sample['stim'], sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()
# original (full scale)
stas = torch.einsum('nlwh,nc->lwhc', im2, sample['robs']-sample['robs'].mean(dim=0))
sta2 = stas.detach().cpu().numpy()

# plotting
NC = sta.shape[3]
plt.figure(figsize=(8,NC*2))
for cc in range(NC):
    sd = np.std(w)
    w = sta[:,:,:,cc]
    w = (w -np.mean(w) )/ sd
    w2 = sta2[:,:,:,cc]
    w2 = (w2 -np.mean(w2) )/ sd

    bestlag = np.argmax(np.std(w2.reshape( (gd.num_lags, -1)), axis=1))
    plt.subplot(NC,4, cc*4 + 1)
    v = np.max(np.abs(w2))
    plt.imshow(w2[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm")
    plt.title(cc)
    plt.subplot(NC,4, cc*4 + 2)
    i,j=np.where(w2[bestlag,:,:]==np.max(w2[bestlag,:,:]))
    plt.plot(w2[:,i[0], j[0]], '-ob')
    i,j=np.where(w2[bestlag,:,:]==np.min(w2[bestlag,:,:]))
    plt.plot(w2[:,i[0], j[0]], '-or')
    yd = plt.ylim()

    plt.subplot(NC,4, cc*4 + 3)
    plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm")
    plt.subplot(NC,4, cc*4 + 4)
    i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-ob')
    i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-or')
    plt.ylim(yd)


NT = sample['stim'].shape[0]
NY = sample['stim'].shape[2]
NX = sample['stim'].shape[3]
flat = nn.Flatten()

# train / validation
Xstim = flat(sample['stim'].permute((0,2,3,1))).detach().cpu().numpy()
Robs = sample['robs'].detach().cpu().numpy()

# test set
sample = gd_shift_test[:] # load sample 

Xstim_test = flat(sample['stim'].permute((0,2,3,1))).detach().cpu().numpy()
Robs_test = sample['robs'].detach().cpu().numpy()


#%% Fit GLM
NT = Xstim.shape[0]
Ui, Xi = NDNutils.generate_xv_folds(NT)

lbfgs_param = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_param['maxiter'] = 1000


d2t = .5
d2xt = .5
l1 = 0
loc = 10000000

NC = Robs.shape[1]

# NDN parameters for processing the stimulus
lin = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[NC],
    layer_types=['readout'], normalization=[0],
    act_funcs=['softplus'], verbose=True,
    reg_list={'d2xt': [d2xt], 'local':[loc], 'l1':[l1]})


# initialize GLM
glm0 = NDN.NDN([lin],  noise_dist='poisson')

v2f0 = glm0.fit_variables(fit_biases=True)

stas = (Xstim.T @ (Robs-np.mean(Robs, axis=0))) / np.sum(Robs, axis=0)
stas /= np.sum(stas,axis=0)
glm0.networks[0].layers[0].weights[:] = copy.deepcopy(stas[:])

glms = []
LLxs = []
d2xts,locs = np.meshgrid(1e-4*10**np.arange(0,8), 1e-1*10**np.arange(0,8))
d2xts = d2xts.flatten()
locs = locs.flatten()

N = len(locs)

pth = Path('./glms')
modname = (gd.id + '_glm_')

# modfiles = list(pth.glob(modname + '*'))

for rr in range(N):
    d2xt = d2xts[rr]
    loc = locs[rr]

    f00 = pth / (modname + str(d2xt) + '_' + str(loc))

    if f00.exists():
        glm0 = NDN.NDN.load_model(str(f00))
    else:
        glm0.set_regularization('local', loc, ffnet_target=0, layer_target=0)
        glm0.set_regularization('d2xt', d2xt, ffnet_target=0, layer_target=0)

        # train initial model
        _ = glm0.train(input_data=[Xstim], output_data=Robs,
            train_indxs=Ui, test_indxs=Xi,
            learning_alg='lbfgs', opt_params=lbfgs_param,
                fit_variables=v2f0)

        glm0.save_model(str(f00))
    
    glms.append(glm0.copy_model())
    LLx = glm0.eval_models(input_data=Xstim, output_data=Robs, data_indxs=Xi, nulladjusted=True)
    LLxs.append(copy.deepcopy(LLx))

#%% get best model
sx = np.ceil(np.sqrt(NC))
sy = np.round(np.sqrt(NC))
LLx = np.asarray(LLxs)

glm = glm0.copy_model()

plt.figure(figsize=(10,10))
for cc in range(NC):
    LLxcc = LLx[:,cc]
    best = np.argmax(LLxcc)
    d2xt = d2xts[best]
    loc = locs[best]
    glm.networks[0].layers[0].reg.vals['local'][cc] = 0 # turn local offcopy.deepcopy(loc)
    glm.networks[0].layers[0].reg.vals['d2xt'][cc] = copy.deepcopy(d2xt)
    print("%d) d2xt: %2.0f, %2.0f" %(cc,d2xt,loc))
    I = np.reshape(LLxcc, (8,8))
    I = np.maximum(I, 0)
    plt.subplot(sx, sy, cc+1)
    plt.imshow(I)
    if np.sum(I)==0:
        glm.networks[0].layers[0].weights[:,cc] = 0
        glm.networks[0].layers[0].biases[0,cc] = np.mean(Robs[Ui,cc])
    else:
        glm.networks[0].layers[0].weights[:,cc] = copy.deepcopy(glms[best].networks[0].layers[0].weights[:,cc])
        glm.networks[0].layers[0].biases[0,cc] = copy.deepcopy(glms[best].networks[0].layers[0].biases[0,cc])

#%% reg path over L1
reg_results = DU.unit_reg_test(glm, input_data=Xstim, output_data=Robs, train_indxs=Ui, test_indxs=Xi,
        reg_type='l1', ffnet_targets=[0], layer_targets=[0], reg_vals=1e-5*10**np.arange(0, 8),
        fit_variables=v2f0, opt_params=lbfgs_param, learning_alg='lbfgs', to_plot=True)

#%% assign individual unit L1 and retrain
glm = DU.unit_assign_reg(glm, reg_results)
_ = glm.train(input_data=[Xstim], output_data=Robs,
            train_indxs=Ui, test_indxs=Xi,
            learning_alg='lbfgs', opt_params=lbfgs_param,
            fit_variables=v2f0)

#%% fit GQM

d2xtlin = copy.deepcopy(glm.networks[0].layers[0].reg.vals['d2xt'])
l1lin = copy.deepcopy(glm.networks[0].layers[0].reg.vals['l1'])

d2xt = .5
loc = 100000
l1 = 0

lin = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[NC],
    layer_types=['readout'], normalization=[0],
    act_funcs=['lin'], verbose=True,
    reg_list={'d2xt':[0], 'local':[0], 'l1':[0]})

quad = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[NC],
    layer_types=['readout'], normalization=[0],
    act_funcs=['quad'], verbose=True,
    reg_list={'d2xt':[d2xt], 'local':[loc], 'l1':[l1]})

quad = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[NC],
    layer_types=['readout'], normalization=[0],
    act_funcs=['quad'], verbose=True,
    reg_list={'d2xt': [d2xt], 'local':[loc], 'l1':[l1]})

add_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1,2], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus'])

# initialize GLM
gqm0 = NDN.NDN([lin, quad, quad, add_par],  noise_dist='poisson')

# set regularization
gqm0.networks[0].layers[0].reg.vals['d2xt'] = d2xtlin
gqm0.networks[0].layers[0].reg.vals['l1'] = l1lin

v2f0 = gqm0.fit_variables(fit_biases=False)
v2f0[-1][-1]['biases'] = True
v2f0[-1][-1]['weights'] = False
v2f0[0][0]['weights'] = False

gqm0.networks[0].layers[0].weights[:] = copy.deepcopy(glm.networks[0].layers[0].weights[:])
gqm0.networks[0].layers[0].biases[:] = 0
gqm0.networks[1].layers[0].biases[:] = 0
gqm0.networks[2].layers[0].biases[:] = 0
gqm0.networks[3].layers[0].biases[:] = copy.deepcopy(glm.networks[0].layers[0].biases[:])

# train initial model
_ = gqm0.train(input_data=[Xstim], output_data=Robs,
    train_indxs=Ui, test_indxs=Xi,
    learning_alg='lbfgs', opt_params=lbfgs_param,
    fit_variables=v2f0)    

gqms = []
LLxs1 = []
d2xts,locs = np.meshgrid(1e-4*10**np.arange(0,8), 1e-2*10**np.arange(0,8))
d2xts = d2xts.flatten()
locs = locs.flatten()

N = len(locs)

pth = Path('./glms')
modname = (gd.id + '_gqm_')

# modfiles = list(pth.glob(modname + '*'))
v2f0[0][0]['weights'] = True

for rr in range(N):
    d2xt = d2xts[rr]
    loc = locs[rr]

    f00 = pth / (modname + str(d2xt) + '_' + str(loc))

    if f00.exists():
        gqm1 = NDN.NDN.load_model(str(f00))
    else:
        gqm1 = gqm0.copy_model()
        gqm1.set_regularization('local', loc, ffnet_target=0, layer_target=0)
        gqm1.set_regularization('d2xt', d2xt, ffnet_target=0, layer_target=0)
        gqm1.set_regularization('local', loc, ffnet_target=1, layer_target=0)
        gqm1.set_regularization('d2xt', d2xt, ffnet_target=1, layer_target=0)
        gqm1.set_regularization('local', loc, ffnet_target=2, layer_target=0)
        gqm1.set_regularization('d2xt', d2xt, ffnet_target=2, layer_target=0)

        # train initial model
        _ = gqm1.train(input_data=[Xstim], output_data=Robs,
            train_indxs=Ui, test_indxs=Xi,
            learning_alg='lbfgs', opt_params=lbfgs_param,
                fit_variables=v2f0)

        gqm1.save_model(str(f00))
    
    gqms.append(gqm1.copy_model())
    LLx = gqm1.eval_models(input_data=Xstim, output_data=Robs, data_indxs=Xi, nulladjusted=True)
    LLxs1.append(copy.deepcopy(LLx)) 

#%% get best model
sx = np.ceil(np.sqrt(NC))
sy = np.round(np.sqrt(NC))
LLx = np.asarray(LLxs1)

gqm = gqm1.copy_model()

plt.figure(figsize=(10,10))
for cc in range(NC):
    LLxcc = LLx[:,cc]
    best = np.argmax(LLxcc)
    d2xt = d2xts[best]
    loc = locs[best]
    gqm.networks[0].layers[0].reg.vals['local'][cc] = copy.deepcopy(loc)
    gqm.networks[0].layers[0].reg.vals['d2xt'][cc] = copy.deepcopy(d2xt)
    gqm.networks[1].layers[0].reg.vals['local'][cc] = copy.deepcopy(loc)
    gqm.networks[1].layers[0].reg.vals['d2xt'][cc] = copy.deepcopy(d2xt)
    gqm.networks[2].layers[0].reg.vals['local'][cc] = copy.deepcopy(loc)
    gqm.networks[2].layers[0].reg.vals['d2xt'][cc] = copy.deepcopy(d2xt)
    print("%d) d2xt: %2.0f, %2.0f" %(cc,d2xt,loc))
    I = np.reshape(LLxcc, (8,8))
    I = np.maximum(I, 0)
    plt.subplot(sx, sy, cc+1)
    plt.imshow(I)
    if np.sum(I)==0:
        gqm.networks[1].layers[0].weights[:,cc] = 0
        gqm.networks[1].layers[0].biases[0,cc] = 0
        gqm.networks[2].layers[0].weights[:,cc] = 0
        gqm.networks[2].layers[0].biases[0,cc] = 0
    else:
        for ll in range(3):
            gqm.networks[ll].layers[0].weights[:,cc] = copy.deepcopy(gqms[best].networks[ll].layers[0].weights[:,cc])
            gqm.networks[ll].layers[0].biases[0,cc] = copy.deepcopy(gqms[best].networks[ll].layers[0].biases[0,cc])
        
#%% refit with best regularization
v2f0 = gqm0.fit_variables(fit_biases=True)
v2f0[-1][-1]['weights'] = False
v2f0[0][0]['biases'] = False

_ = gqm.train(input_data=[Xstim], output_data=Robs,
            train_indxs=Ui, test_indxs=Xi,
            learning_alg='lbfgs', opt_params=lbfgs_param,
            fit_variables=v2f0)

# %% remove local, find L1 regularization
        
#%% plot filters
DU.plot_3dfilters(gqm0, ffnet=0)

DU.plot_3dfilters(gqm0, ffnet=1)

DU.plot_3dfilters(gqm0, ffnet=2)


#%% test set
LLx0 = glm.eval_models(input_data=Xstim_test, output_data=Robs_test, nulladjusted=True)
LLx1 = gqm.eval_models(input_data=Xstim_test, output_data=Robs_test, nulladjusted=True)
#%%

plt.plot(LLx0)
plt.plot(LLx1)
plt.plot(l2)
plt.ylim((0,.8))

#%%


# optimizer parameters
adam_params = {'use_gpu': True,
        'display': 30,
        'data_pipe_type': 'data_as_var',
        'poisson_unit_norm': None,
        'epochs_ckpt': None,
        'learning_rate': 1e-3,
        'batch_size': 1000,
        'epochs_training': 10000,
        'early_stop_mode': 1,
        'MAPest': True,
        'func_tol': 0,
        'epochs_summary': None,
        'early_stop': 100,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-08,
        'run_diagnostics': False}

# d2ts = 1e-4*10**np.arange(0, 5)

d2xs = [10] #1e-2*10**np.arange(0, 5)
gqms = []
LLxs = []
for step in range(len(d2xs)):

    d2t = .5
    d2x = d2xs[step]
    # loc = 10000
    l1 = 1e-3

    NC = Robs.shape[1]
    # NDN parameters for processing the stimulus
    lin = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags],
        layer_sizes=[NC],
        layer_types=['readout'], normalization=[0],
        act_funcs=['lin'], verbose=True,
        reg_list={'d2t': [d2t], 'd2x':[d2x], 'local':[loc], 'l1':[l1]})

    quad = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags],
        layer_sizes=[NC],
        layer_types=['readout'], normalization=[0],
        act_funcs=['quad'], verbose=True,
        reg_list={'d2t': [d2t], 'd2x':[d2x], 'local':[loc], 'l1':[l1]})

    quad = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags],
        layer_sizes=[NC],
        layer_types=['readout'], normalization=[0],
        act_funcs=['quad'], verbose=True,
        reg_list={'d2t': [d2t], 'd2x':[d2x], 'local':[loc], 'l1':[l1]})

    add_par = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1,2], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus'])

    # initialize GLM
    gqm0 = NDN.NDN([lin, quad, quad, add_par],  noise_dist='poisson')

    v2f0 = gqm0.fit_variables(fit_biases=False)
    v2f0[-1][-1]['biases'] = True
    v2f0[-1][-1]['weights'] = False

    stas = (Xstim.T @ (Robs-np.mean(Robs, axis=0))) / np.sum(Robs, axis=0)
    stas /= np.sum(stas,axis=0)
    gqm0.networks[0].layers[0].weights[:] = copy.deepcopy(stas[:])

    # train initial model
    _ = gqm0.train(input_data=[Xstim], output_data=Robs,
        train_indxs=Ui, test_indxs=Xi,
        learning_alg='adam', opt_params=adam_params,
         fit_variables=v2f0)

    LLx = gqm0.eval_models(input_data=Xstim, output_data=Robs, data_indxs=Xi, nulladjusted=True)     

    gqms.append(gqm0)
    LLxs.append(LLx)         

#%%
gqm0 = gqms[0].copy_model()

# gqml1 = 
# l1s = 
for step in range(len(l1s)):

    d2t = .5
    d2x = d2xs[step]
    loc = 10000
    l1 = l1s[step]

    NC = Robs.shape[1]
    # NDN parameters for processing the stimulus
    lin = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags],
        layer_sizes=[NC],
        layer_types=['readout'], normalization=[0],
        act_funcs=['lin'], verbose=True,
        reg_list={'d2t': [d2t], 'd2x':[d2x], 'local':[loc], 'l1':[l1]})

    quad = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags],
        layer_sizes=[NC],
        layer_types=['readout'], normalization=[0],
        act_funcs=['quad'], verbose=True,
        reg_list={'d2t': [d2t], 'd2x':[d2x], 'local':[loc], 'l1':[l1]})

    quad = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags],
        layer_sizes=[NC],
        layer_types=['readout'], normalization=[0],
        act_funcs=['quad'], verbose=True,
        reg_list={'d2t': [d2t], 'd2x':[d2x], 'local':[loc], 'l1':[l1]})

    add_par = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1,2], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus'])

    # initialize GLM
    gqm0 = NDN.NDN([lin, quad, quad, add_par],  noise_dist='poisson')

    v2f0 = gqm0.fit_variables(fit_biases=False)
    v2f0[-1][-1]['biases'] = True
    v2f0[-1][-1]['weights'] = False

    stas = (Xstim.T @ (Robs-np.mean(Robs, axis=0))) / np.sum(Robs, axis=0)
    stas /= np.sum(stas,axis=0)
    gqm0.networks[0].layers[0].weights[:] = copy.deepcopy(stas[:])

    # train initial model
    _ = gqm0.train(input_data=[Xstim], output_data=Robs,
        train_indxs=Ui, test_indxs=Xi,
        learning_alg='adam', opt_params=adam_params,
         fit_variables=v2f0)

    LLx = gqm0.eval_models(input_data=Xstim, output_data=Robs, data_indxs=Xi, nulladjusted=True)     

    gqms.append(gqm0)
    LLxs.append(LLx)      
#%% pick best regularization
bestreg = np.zeros(NC)
reg_path = np.asarray(LLxs)
for cc in range(NC):
    bestind = np.argmax(reg_path[:,cc])
    bestreg[cc] = d2xs[bestind]
plt.figure()
f = plt.plot(reg_path)




#%% plot model

gqm0 = gqms[0].copy_model()
DU.plot_3dfilters(gqm0, ffnet=0)

DU.plot_3dfilters(gqm0, ffnet=1)

DU.plot_3dfilters(gqm0, ffnet=2)


#%% test set
sample = gd_shift_test[:] # load sample 

Xstim_test = flat(sample['stim'].permute((0,2,3,1))).detach().cpu().numpy()
Robs_test = sample['robs'].detach().cpu().numpy()

LLx = gqm0.eval_models(input_data=Xstim_test, output_data=Robs_test, nulladjusted=True)

#%%
plt.plot(LLx, '-o')
yd = plt.ylim()
yd = (-.05, yd[1])
plt.ylim(yd)

#%%

stas = torch.einsum('nlwh,nc->lwhc', sample['stim'], (sample['robs']-sample['robs'].mean(dim=0))/sample['robs'].sum(dim=0))
sta = stas.detach().cpu().numpy()

#%% plot STAs / get RF centers
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

plt.figure(figsize=(12,10))
for cc in range(NC):
    w = sta[:,:,:,cc]
    w = (w -np.mean(w) )/ np.std(w)

    bestlag = np.argmax(np.std(w.reshape( (gd.num_lags, -1)), axis=1))
    plt.subplot(sx,sy, cc*2 + 1)
    v = np.max(np.abs(w))
    plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=(-1,1,-1,1))

    plt.title(cc)
    plt.subplot(sx,sy, cc*2 + 2)
    i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-ob')
    i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-or')
    yd = plt.ylim()

#%%  build GLM
import V1FreeViewingCode.models.layers as layers
importlib.reload(layers)
importlib.reload(cores)
importlib.reload(readouts)

#%%



# %%
save_dir='./test_linear_model'
from pathlib import Path
# % LN model

"""
Fit LN model
"""
for version in [8]:#range(1,6): # range of version numbers
    
    print("Version %d" %version)
    #% Model: GLM    
    input_dims = [gd_shift.num_lags, gd_shift.NY, gd_shift.NX]
    core = cores.GLM(input_size=input_dims,
            output_channels=gd.NC,
            weight_norm=True,
            bias=True,
            activation="softplus",
            input_regularizer="RegMats",
            input_reg_types=["d2x", "d2t"],
            input_reg_amt=[.001,.001])

    readout = readouts.IdentityReadout(gd.NC)
    readout.bias.data = sample['robs'].mean(dim=0)

    model = Encoder(core, readout,
            output_nl=nn.Identity(),
            weight_decay=.1, optimizer='AdamW', learning_rate=.001, # high initial learning rate because we decay on plateau
            betas=[.9, .999], amsgrad=True)

    #% check that the forward runs (for debugging)
    out = model(sample['stim'][:10,:], shifter=None)
    # out.shape
    model.core.features.layer0.conv.weight.data = stas.permute(3,0,1,2)
    model.core.features.layer0.conv.weight_v.data = stas.permute(3,0,1,2)
    # turn off training the features
    model.core.features.layer0.conv.weight_v.requires_grad=False

    #% Train
    trainer, train_dl, valid_dl = ut.get_trainer(gd_shift, version=version,
                save_dir=save_dir,
                name=gd_shift.id,
                auto_lr=False,
                batchsize=1000)
    
    trainpath = Path(save_dir) / gd_shift.id / "version_{}".format(version)
    if not trainpath.exists():
        trainer.fit(model, train_dl, valid_dl)
        model.core.features.layer0.conv.weight_v.requires_grad=True
        trainer, train_dl, valid_dl = ut.get_trainer(gd_shift, version=version,
                save_dir=save_dir,
                name=gd_shift.id,
                auto_lr=False,
                batchsize=1000)
        trainer.fit(model, train_dl, valid_dl)

    else:
        print("Already run. Skipping.")
# %%

model.core.plot_filters()
# %%
gd_shift_test = dd.PixelDataset(sessid, stims=["Gabor"],
    stimset="Test", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    valid_eye_ctr=(0.0,0.0),
    include_eyepos=True,
    cropidx=[(15,50),(20,50)],
    shifter=model2.readout.shifter,
    preload=True)
# %%
ll2 = get_null_adjusted_ll(model, gd_shift_test[:])

plt.plot(ll2, 'o-')

#%%
w = model.core.features.layer0.conv.weight.detach().cpu().numpy()

#%%
cc +=1
if cc >= gd_shift.NC:
    cc = 0

plt.figure(figsize=(10,5))
wtmp = w[cc,:,:,:]
wtmp = (wtmp - np.min(wtmp)) / (np.max(wtmp) - np.min(wtmp))

for l in range(10):
    plt.subplot(3,4,l+1)
    plt.imshow(wtmp[l,:,:], vmin=0, vmax=1, interpolation=None)
    plt.xlabel(cc)
# %%
plt.imshow(wtmp[4,:,:], interpolation=None)

#%%

import pandas as pd
from pathlib import Path
import seaborn as sns

pth = Path(save_dir) / gd.id

valloss = np.array([])
ind = np.array([])
for v in pth.rglob('version*'):
    
    try:
        df = pd.read_csv(str(v / "metrics.csv"))
        valloss = np.append(valloss, np.nanmin(df.val_loss.to_numpy()))
        ind = np.append(ind, int(v.name.split('_')[1]))
        
    except:
        "skip"

sortind = np.argsort(ind)
ind = ind[sortind]
valloss = valloss[sortind]

plt.plot(ind, valloss, 'o')
plt.ylim((0.13, .15))

# %%
mn = np.argmin(valloss)
ver1 = ind[mn]
ver = "version_" + str(int(ver1))
chkpath = pth / ver / 'checkpoints'
best_epoch = ut.find_best_epoch(chkpath)
model = Encoder.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))

#%%

nn.utils.remove_weight_norm(model.core.features.layer0.conv)
# nn.utils.remove_weight_norm(model.core.features.layer1.conv)

#%%
model.core.plot_filters()
# %%
# w = model.
import V1FreeViewingCode.models.encoders as enc
importlib.reload(enc)

version +=1
input_dims = [gd_shift.num_lags, gd_shift.NY, gd_shift.NX]
glm = enc.GLM(input_size=input_dims,
            output_size=gd.NC,
            bias=True,
            l1_strength=1e-4,
            l2_strength=1e-2,
            input_reg_types=["d2x", "d2t"],
            input_reg_amt=[1e-3,1e-4],
            weight_decay=.1,
            optimizer='AdamW',
            learning_rate=.0001,
            betas=[.9, .999], amsgrad=True)

#% check that the forward runs (for debugging)
out = glm(sample['stim'][:10,:])

# out.shape
glm.linear.weight.data = stas.permute(3,0,1,2)

#% Train
trainer, train_dl, valid_dl = ut.get_trainer(gd_shift, version=version,
            save_dir=save_dir,
            name=gd_shift.id,
            auto_lr=False,
            batchsize=1000)

trainpath = Path(save_dir) / gd_shift.id / "version_{}".format(version)

trainer.fit(glm, train_dl, valid_dl)
# %%

w = glm.linear.weight.detach().cpu().numpy()
NC = w.shape[0]

plt.figure(figsize=(20,2*NC))
for cc in range(NC):
    wtmp = w[cc,:,:,:]
    wtmp = (wtmp - np.min(wtmp))/(np.max(wtmp) - np.min(wtmp))
    for l in range(num_lags):
        plt.subplot(NC,num_lags, cc*num_lags + l+1)
        plt.imshow(wtmp[l,:,:], vmin=0, vmax=1, interpolation=None, aspect='auto')

# %%
l2 = get_null_adjusted_ll(glm, gd_shift_test, use_shifter=False)
plt.plot(l2, 'o-')
plt.axhline(0, color='k')