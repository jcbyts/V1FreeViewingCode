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

import V1FreeViewingCode.Analysis.notebooks.Utils as U


#%% Load dataset
sessid = '20200304'
# sessid = '20191231'

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
save_dir='../../checkpoints/core_shifter_ls{}'.format(lengthscale)

from pathlib import Path

#%% get the best shifter model
 
import pandas as pd
from pathlib import Path
import seaborn as sns

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

#--- axis 1: plot 
n1 = sum(ind < 11)
y = valloss[ind>10]
mn = np.argmin(y)
ver2 = ind[mn+n1-1]

# --- Load LNLN version
ver = "version_" + str(int(ver2))
chkpath = pth / ver / 'checkpoints'
best_epoch = ut.find_best_epoch(chkpath)
model2 = Encoder.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))
shifter = model2.readout.shifter
shifter.cpu()


#%% reload data with cropping and shift

import pickle
save_dir2 ='../../checkpoints/core_onshiftedstim' #_ls{}'.format(lengthscale)
fname = save_dir2 + '/' + sessid + 'cids.p'
tmp = pickle.load(open(fname, "rb"))
cids = tmp['cids']
win = tmp['win']

#%%
import V1FreeViewingCode.models.datasets as dd
import importlib
importlib.reload(dd)

# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================
# TRY FITTING SUPERIOR MODELS ON SHIFT CORRECTED STIMULUS
# ======================================================================
# ======================================================================
# ======================================================================

#%% reload data at higher temporal resolution

import pickle
save_dir2 ='../../checkpoints/core_onshiftedstim' #_ls{}'.format(lengthscale)
fname = save_dir2 + '/' + sessid + 'cids.p'
tmp = pickle.load(open(fname, "rb"))
cids = tmp['cids']
win = tmp['win']
win = [(5,65), (5,65)]

import V1FreeViewingCode.models.datasets as dd
import importlib
importlib.reload(dd)
t_downsample = 1
num_lags = 20

gd_shift = dd.PixelDataset(sessid, stims=["Gabor", "Grating", "BackImage"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5,
    valid_eye_ctr=(0.0,0.0),
    include_eyepos=True,
    cropidx=win,
    cids=cids,
    shifter=shifter, #model2.readout.shifter,
    preload=True,
    temporal=True)

#%%  test set
gab_shift_test = dd.PixelDataset(sessid, stims=["Gabor"],
    stimset="Test", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    valid_eye_ctr=(0.0,0.0),
    include_eyepos=True,
    cropidx=win,
    cids=cids,
    shifter=shifter,
    preload=True,
    temporal=True)


ni_shift_test = dd.PixelDataset(sessid, stims=["BackImage"],
    stimset="Test", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    valid_eye_ctr=(0.0,0.0),
    include_eyepos=True,
    cropidx=win,
    cids=cids,
    shifter=shifter,
    preload=True,
    temporal=True)

gr_shift_test = dd.PixelDataset(sessid, stims=["Grating"],
    stimset="Test", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    valid_eye_ctr=(0.0,0.0),
    include_eyepos=True,
    cropidx=win,
    cids=cids,
    shifter=shifter,
    preload=True,
    temporal=True)

fix_shift_test = dd.PixelDataset(sessid, stims=["FixRsvpStim"],
    stimset="Test", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    valid_eye_ctr=(0.0,0.0),
    include_eyepos=True,
    cropidx=win,
    cids=cids,
    shifter=shifter,
    preload=True,
    temporal=True)

#%% PLOT STAS
sample = gd_shift[:] # load sample 

# shifted
stas = torch.einsum('nlwh,nc->lwhc', sample['stim'].squeeze(), sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()

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

    bestlag = np.argmax(np.std(w.reshape( (gd_shift.num_lags, -1)), axis=1))
    plt.subplot(sx,sy, cc*2 + 1)
    v = np.max(np.abs(w))
    plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm", extent=(-1,1,-1,1))
    # plt.plot(mu[cc,0], mu[cc,1], '.b')
    plt.title(cc)
    plt.subplot(sx,sy, cc*2 + 2)
    i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-ob')
    i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-or')
    yd = plt.ylim()
#%% BUILD MODEL AND FIT IT


"""
SINGLE LAYER DIV NORM MODEL
"""
for version in range(25,27): # range of version numbers
    

    #% Model: convolutional model
    input_channels = gd_shift.num_lags
    hidden_channels = 16
    input_kern = 19
    hidden_kern = 19
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
            input_reg_amt=[.000005, .001, 0.00001],
            hidden_reg_types=["d2x", "center"],
            hidden_reg_amt=[.000005, .01],
            stack=None,
            use_avg_reg=True)


    # initialize input layer to be centered
    regw = regularizers.gaussian2d(input_kern,sigma=input_kern//4)
    core.features[0].conv.weight.data = torch.einsum('ijkm,km->ijkm', core.features[0].conv.weight.data, torch.tensor(regw))
        
    # Readout
    in_shape = [core.outchannels, gd_shift.NY, gd_shift.NX]
    bias = True
    readout = readouts.Point2DGaussian(in_shape, gd_shift.NC, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
                    align_corners=True, gauss_type='uncorrelated',
                    constrain_positive=False,
                    shifter= None)
                    
                    # {'hidden_features': 20,
                    #         'hidden_layers': 1,
                    #         'final_tanh': False,
                    #         'activation': "softplus",
                    #         'lengthscale': lengthscale}
                    #         )

    # combine core and readout into model
    model = Encoder(core, readout,
        weight_decay=.001, optimizer='AdamW', learning_rate=.01, # high initial learning rate because we decay on plateau
        betas=[.9, .999], amsgrad=True)

    # initialize readout based on spike rate and STA centers
    model.readout.bias.data = sample['robs'].mean(dim=0) # initialize readout bias helps
    # model.readout._mu.data[0,:,0,:] = torch.tensor(mu.astype('float32')) # initiaalize mus

    #% check that the forward runs (for debugging)
    out = model(sample['stim'][:10,:]) #, shifter=sample['eyepos'][:10,:])
    # out.shape

    #% Train
    trainer, train_dl, valid_dl = ut.get_trainer(gd_shift, version=version,
                save_dir=save_dir2,
                name=gd_shift.id,
                auto_lr=False,
                batchsize=1000)

    trainpath = Path(save_dir2) / gd_shift.id / "version_{}".format(version)
    if not trainpath.exists():
        trainer.fit(model, train_dl, valid_dl)

#%% TEMPORAL CONVOLUTION EI

for version in range(32,34): # range of version numbers
    

    #% Model: convolutional model
    input_channels = gd_shift.num_lags
    ksize = [(15,5,5), (5,15,15)] # kernel size for each layer (NT, NY, NX)
    core = cores.Stacked3dCore([2,10],ksize,input_channels=1,
        act_funcs=["relu","relu"], ei_split=[0,.5], divnorm=None,
        weight_norm=[1,1], group_norm=[1,1], bias=[1,1],
        reg_types=[["d2t", "d2x", "center"], ["d2t", "d2x","center"]],
        reg_amt=[[1e-3, 1e-3, 1e-1], [1e-3, 1e-4, 1e-4]])
        
    # Readout
    in_shape = [core.outchannels, gd_shift.NY, gd_shift.NX]
    bias = True
    readout = readouts.Point2DGaussian(in_shape, gd_shift.NC, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
                    align_corners=True, gauss_type='uncorrelated',
                    constrain_positive=True,
                    shifter= None)
                    
                    # {'hidden_features': 20,
                    #         'hidden_layers': 1,
                    #         'final_tanh': False,
                    #         'activation': "softplus",
                    #         'lengthscale': lengthscale}
                    #         )

    # combine core and readout into model
    model = Encoder(core, readout,
        weight_decay=.001, optimizer='AdamW', learning_rate=.01, # high initial learning rate because we decay on plateau
        betas=[.9, .999], amsgrad=True)

    # initialize readout based on spike rate and STA centers
    model.readout.bias.data = sample['robs'].mean(dim=0) # initialize readout bias helps
    # model.readout._mu.data[0,:,0,:] = torch.tensor(mu.astype('float32')) # initiaalize mus

    #% check that the forward runs (for debugging)
    out = model(sample['stim'][:10,:]) #, shifter=sample['eyepos'][:10,:])
    # out.shape

    #% Train
    trainer, train_dl, valid_dl = ut.get_trainer(gd_shift, version=version,
                save_dir=save_dir2,
                name=gd_shift.id,
                auto_lr=False,
                batchsize=1000)

    # trainpath = Path(save_dir2) / gd_shift.id / "version_{}".format(version)
    # if not trainpath.exists():
    trainer.fit(model, train_dl, valid_dl)

    #%%
#% Train
trainer, train_dl, valid_dl = ut.get_trainer(gd_shift, version=version,
            save_dir=save_dir2,
            name=gd_shift.id,
            auto_lr=False,
            batchsize=1000)

# trainpath = Path(save_dir2) / gd_shift.id / "version_{}".format(version)
# if not trainpath.exists():
#%%
trainer.fit(model, train_dl, valid_dl)
#%%



w = model.core.features[0].conv.weight.detach().cpu().numpy()
num_lags = w.shape[2]
nch = w.shape[1]
for cc in range(w.shape[0]):
    plt.figure(figsize=(15,nch*5))
    for i in range(num_lags):
        for j in range(nch):
            plt.subplot(nch,num_lags,i+1)
            f = plt.imshow(w[cc,j,i,:,:], vmin=np.min(w[cc,j,:,:,:]), vmax=np.max(w[cc,j,:,:,:]))
#%%
def plot_filters(model, sort=False, layer=0):
        import numpy as np
        import matplotlib.pyplot as plt  # plotting
        # ei_mask = model.features.layer0.eimask.ei_mask.detach().cpu().numpy()
        
        w = model.features[layer].conv.weight.detach().cpu().numpy()
        ei_mask = np.ones(w.shape[0])
        sz = w.shape
        # w = model.features.weight.detach().cpu().numpy()
        w = w.reshape(sz[0], sz[1], sz[2]*sz[3])
        nfilt = w.shape[0]
        if sort:
            n = np.asarray([w[i,:].abs().max().detach().numpy() for i in range(nfilt)])
            cinds = np.argsort(n)[::-1][-len(n):]
        else:
            cinds = np.arange(0, nfilt)

        sx = np.ceil(np.sqrt(nfilt*2))
        sy = np.round(np.sqrt(nfilt*2))
        # sx,sy = U.get_subplot_dims(nfilt*2)
        mod2 = sy % 2
        sy += mod2
        sx -= mod2

        plt.figure(figsize=(10,10))
        for cc,jj in zip(cinds, range(nfilt)):
            plt.subplot(sx,sy,jj*2+1)
            wtmp = np.squeeze(w[cc,:])
            bestlag = np.argmax(np.std(wtmp, axis=1))
            plt.imshow(np.reshape(wtmp[bestlag,:], (sz[2], sz[3])), interpolation=None, )
            wmax = np.argmax(wtmp[bestlag,:])
            wmin = np.argmin(wtmp[bestlag,:])
            plt.axis("off")

            plt.subplot(sx,sy,jj*2+2)
            if ei_mask[cc]>0:
                plt.plot(wtmp[:,wmax], 'b-')
                plt.plot(wtmp[:,wmin], 'b--')
            else:
                plt.plot(wtmp[:,wmax], 'r-')
                plt.plot(wtmp[:,wmin], 'r--')

            plt.axhline(0, color='k')
            plt.axvline(bestlag, color=(.5, .5, .5))
            plt.axis("off")

model.core.features[0].conv.weight.shape
#%%
# class foo(nn.Module):
#     """
#     Laplace regularizer, with a Gaussian mask, for a single 2D convolutional layer.

#     """

#     def __init__(self):
#         super().__init__()
#         setattr(self, "a", 5)

# # a = foo()
# a = nn.ModuleList()

#%%

sample = gd_shift[:10]

sample['stim'].shape
#%%

from torch.nn import functional as F

import torch.nn as nn
#%%


#%%
import V1FreeViewingCode.models.layers as ll
import importlib
importlib.reload(ll)

conv1 = nn.Conv3d(1, 5, (15, 5, 5), padding=(0,0,0))
conv2 = ll.posConv3d(5, 10, (5, 15, 15), padding=0)

sample = gd_shift[:100]
a = sample['stim']
b = a.unsqueeze(1)

y = conv1(b)
print(y.shape)

y = conv2(y)
print(y.shape)

#%%
importlib.reload(cores)
ksize = [(15,5,5), (5,15,15)]
c = cores.Stacked3dCore([6,20],ksize,input_channels=1,
    act_funcs=["relu","relu"], ei_split=[.5,.5], divnorm=None,
    weight_norm=None, group_norm=[0,0], bias=[1,1],
    reg_types=[["d2t", "d2x", "center"], ["d2t", "d2x","center"]],
    reg_amt=[[1e-4, 1e-3, 1e-1], [1e-4, 1e-4, 1e-4]])
    
#%%            
y = c(b)

y = y.detach().cpu().numpy()

#%%
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
f = plt.plot(y[:,:5,25,25])
plt.subplot(2,1,2)
f = plt.plot(y[:,5:,25,25])

#%% find best version
pth = Path(save_dir2) / gd_shift.id

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

plt.plot(ind, valloss, 'o')
# plt.xlim([10, 25])

#%% Load

bnds = [16, 20]
bnds = [25, 27]
good = np.logical_and(ind >= bnds[0], ind <= bnds[1])
goodi = ind[good]
ii = np.argmin(valloss[good])
vernum = goodi[ii]
print("best run: %d" %vernum)
#%%
ver = "version_" + str(int(vernum))
chkpath = pth / ver / 'checkpoints'
best_epoch = ut.find_best_epoch(chkpath)
model2 = Encoder.load_from_checkpoint(str(chkpath / 'epoch={}.ckpt'.format(best_epoch)))

nn.utils.remove_weight_norm(model2.core.features.layer0.conv)
# nn.utils.remove_weight_norm(model2.core.features.layer1.conv)

#%% PLOT 

# 1. plot subunits
model2.core.plot_filters()

# # 2. plot pooling
# w = model2.core.features.layer1.conv.weight.detach().cpu().numpy()
# plt.figure(figsize=(15,15))
# for i in range(w.shape[0]):
#     for j in range(w.shape[1]):
#         plt.subplot(w.shape[0], w.shape[1], i*w.shape[0]+j+1)
#         plt.imshow(w[i,j,:,:], aspect='auto')
#         plt.title(i)
#         plt.axis("off")

# 3. plot readout mus
plt.figure()
xy = model2.readout.mu.detach().cpu().numpy().squeeze()
plt.plot(xy[:,0], xy[:,1], '.')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Readout')


#%% evaluate data quality?
def get_null_adjusted_ll_temporal(model, sample, bits=False, use_shifter=True):
    '''
    get null-adjusted log likelihood
    bits=True will return in units of bits/spike
    '''
    m0 = model.cpu()
    loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
    
    if use_shifter:
        yhat = m0(sample['stim'], shifter=sample['eyepos'])
    else:
        yhat = m0(sample['stim'])
    llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy()

    return llneuron

# sample = gd_shift[:]
# LLt = get_null_adjusted_ll_temporal(model2, sample)

#%%
sample = gd_shift[:1000]
yhat = model2(sample['stim'])
# LLt = get_null_adjusted_ll_temporal(model2, sample)

#%%
w = 2*240
from scipy.signal import convolve
x = convolve(LLt, np.ones(w), mode='same') / w

#%%
#5. plot test LL
sample = gab_shift_test[:]
l3 = get_null_adjusted_ll(model2, sample)

sample = gr_shift_test[:]
l4 = get_null_adjusted_ll(model2, sample)

sample = ni_shift_test[:]
l5 = get_null_adjusted_ll(model2, sample)

#%%
sample = fix_shift_test[:]
l6 = get_null_adjusted_ll(model2, sample)

#%%
plt.figure()
# plt.plot(l2, '-o')
plt.plot(cids, l3, '-o')
plt.plot(cids, l4, '-o')
plt.plot(cids, l5, '-o')
plt.plot(cids, l6, '-o')
plt.axhline(0, color='k')
#%% 

sample = gd_shift[:]

NT = sample['stim'].shape[0]
NY = sample['stim'].shape[2]
NX = sample['stim'].shape[3]
flat = nn.Flatten()

# train / validation
Xstim = flat(sample['stim'].permute((0,2,3,1))).detach().cpu().numpy()
Robs = sample['robs'].detach().cpu().numpy()

# test set
sample = gab_shift_test[:] # load sample 

Xstim_test = flat(sample['stim'].permute((0,2,3,1))).detach().cpu().numpy()
Robs_test = sample['robs'].detach().cpu().numpy()

NT = Xstim.shape[0]
Ui, Xi = NDNutils.generate_xv_folds(NT)

num_lags = gd_shift.num_lags
NC = Robs.shape[1]

adam_params = U.def_adam_params()
#%% Make model


Greg0 = 1e-1
Greg = 1e-1
Creg0 = 1
Creg = 1e-2
Mreg0 = 1e-3
Mreg = 1e-1
L1reg0 = 1e-5
Xreg = 1e-2

num_tkern = 2
num_subs = 8

# ndn_par = NDNutils.ffnetwork_params(
#     input_dims=[1,NX,NY,num_lags],
#     layer_sizes=[num_tkern, num_subs, num_tkern, NC], time_expand=[0,0,3,0],
#     layer_types=['conv', 'conv', 'temporal', 'normal'],
#     conv_filter_widths=[1,12,None],ei_layers=[None,num_subs//2,None,None],
#     normalization=[2,1,-1,-1],act_funcs=['relu', 'relu', 'lin', 'softplus'],
#     verbose=True,
#     reg_list={'d2t':[1e-4], 'd2x':[None, Xreg], 'l1':[None, L1reg0],
#     'center':[None, Creg0], 'glocal':[None, Greg0], 'max':[None, None, Mreg0]}
# )

ndn_par = NDNutils.ffnetwork_params(
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[num_tkern, num_subs, NC],
    layer_types=['conv', 'conv', 'normal'],
    conv_filter_widths=[1,12,None],ei_layers=[None,num_subs//2],
    normalization=[2,1,-1],act_funcs=['relu', 'relu', 'softplus'],
    verbose=True,
    reg_list={'d2t':[1e-4], 'd2x':[None, Xreg], 'l1':[None, L1reg0],
    'center':[None, Creg0], 'glocal':[None, Greg0], 'max':[None, None, Mreg0]}
)


# stim only
retV1 = NDN.NDN( [ndn_par], ffnet_out=0, noise_dist='poisson')
retV1.networks[0].layers[0].weights[:,:] = 0
retV1.networks[0].layers[0].weights[2:4,0] = 1
retV1.networks[0].layers[0].weights[2:4,1] = -1

v2f = retV1.fit_variables(fit_biases=False)
v2f[0][0]['biases'] = True
v2f[-1][-1]['biases'] = True

#%% train
_ = retV1.train(input_data=[Xstim], output_data=Robs, train_indxs=Ui,
    test_indxs=Xi, silent=False, learning_alg='adam', opt_params=adam_params, fit_variables=v2f)
# %% fit
DU.plot_3dfilters(retV1)

# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================




# STRAY CODE BELOW HERE





# ======================================================================
# ======================================================================
# ======================================================================


#%%


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