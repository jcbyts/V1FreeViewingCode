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

#%% Load dataset
num_lags = 10
t_downsample = 2 # temporal downsampling
sessid = 'gru20210525' #'20200304'
# gd = PixelDataset(sessid, stims=["Gabor", "BackImage", "Dots", "DriftingGrating"],
#     stimset="Train", num_lags=num_lags,
#     downsample_t=t_downsample,
#     downsample_s=1,
#     include_eyepos=True,
#     preload=True)

n = 40
num_basis = 15
B = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,n)), axis=1) - np.arange(0,n,n/num_basis))/n*num_basis, 0)

gd = PixelDataset(sessid, stims=["Gabor", "Dots"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    include_frametime={'num_basis': 40, 'full_experiment': False},
    include_saccades=[{'name':'sacon', 'basis':B, 'offset':-20}, {'name':'sacoff', 'basis':B, 'offset':0}],
    include_eyepos=True,
    preload=True)


#%% compute STAS
"""
Compute STAS using einstein summation through pytorch
"""
sample = gd[:] # load sample 

stas = torch.einsum('nlwh,nc->lwhc', sample['stim']**2, sample['robs']-sample['robs'].mean(dim=0))
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
import V1FreeViewingCode.models.encoders as encoders
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
# save_dir='./core_shifter_ls{}'.format(lengthscale)
save_dir='../../checkpoints/v1calibration_ls{}'.format(lengthscale)
from pathlib import Path


#%%
"""
Fit single layer DivNorm model with modulation
"""
for version in range(2): # range of version numbers
    

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
                num_workers=64,
                earlystopping=True)

    trainpath = Path(save_dir) / gd.id / "version_{}".format(version)
    if not trainpath.exists():
        trainer.fit(model, train_dl, valid_dl)


print("Done")
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

plt.plot(ind, valloss)

#%%
ver1 = 1
ver = "version_" + str(int(ver1))
chkpath = pth / ver / 'checkpoints'
best_epoch = ut.find_best_epoch(chkpath)

best_epoch = ut.find_best_epoch(chkpath)
model = encoders.EncoderMod.load_from_checkpoint(str(chkpath / best_epoch))
shifter1 = model.readout.shifter

#%% plot best linear shifter and best LNLN shifter
from matplotlib.patches import ConnectionPatch
# import matplotlib as mpl

# build inputs for shifter plotting
xx,yy = np.meshgrid(np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,100),np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,100))
xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

inputs = torch.cat( (xgrid,ygrid), dim=1)

shifter1.cpu()
y = shifter1(inputs)
y2 = y.detach().cpu().numpy()
# y2 = y2 - np.mean(y2, axis=0)
y2/=gd.valid_eye_rad/60 # conver to arcmin
vmin = np.min(y2)
vmax = np.max(y2)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(y2[:,0].reshape((100,100)), extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
plt.title("Horizontal")

plt.subplot(1,2,2)
plt.imshow(y2[:,1].reshape((100,100)), extent=(-gd.valid_eye_rad,gd.valid_eye_rad,-gd.valid_eye_rad,gd.valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
plt.title("Vertical")

#%% run each shifter to see what they predict

iix = np.arange(0,1000) + 2000
plt.figure(figsize=(10,5))
sample = gd[:]
shifters = []
shifters.append(shifter1)

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

shift = shifters[0](sample['eyepos']).detach()
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
stas = torch.einsum('nlwh,nc->lwhc', im**2, sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()
stas = torch.einsum('nlwh,nc->lwhc', im2**2, sample['robs']-sample['robs'].mean(dim=0))
sta2 = stas.detach().cpu().numpy()
cc = 0
#%%

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
cc = 7
print(cc)
# cc += 1

w = sta[:,:,:,cc]
w = (w - np.min(w)) / (np.max(w)-np.min(w))
w2 = sta2[:,:,:,cc]
w2 = (w2 - np.min(w2)) / (np.max(w2)-np.min(w2))
plt.figure(figsize=(20,4))
for i in range(w.shape[0]):
    plt.subplot(2,w.shape[0],i+1)
    plt.imshow(w[i,:,:], vmin=0, vmax=1, interpolation='Nearest')
    plt.axis("off")
    plt.subplot(2,w.shape[0],w.shape[0]+i+1)
    plt.imshow(w2[i,:,:], vmin=0, vmax=1, interpolation='Nearest')
    plt.axis("off")

cc +=1

#%% plot individual frame
fr = 4
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(w[fr,:,:], interpolation='None', extent=(0,70/gd.ppd*60,0,70/gd.ppd*60), cmap="gray")
plt.xlabel("arcmin")
plt.ylabel("arcmin")

plt.subplot(122)
plt.imshow(w2[fr,:,:], interpolation='None', extent=(0,70/gd.ppd*60,0,70/gd.ppd*60), cmap="gray")
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
model2 = Encoder.load_from_checkpoint(str(chkpath / best_epoch))

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
    shifter=model2.readout.shifter,
    preload=True)
# %% reload sample and compute STAs
sample = gd_shift[:] # load sample 

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