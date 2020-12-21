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
t_downsample = 2 
sessid = '20200304'
gd = PixelDataset(sessid, stims=["Gabor"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    include_eyepos=True,
    preload=True)

#%% compute STAS
sample = gd[:] # load sample 

stas = torch.einsum('nlwh,nc->lwhc', sample['stim'], sample['robs'])
sta = stas.detach().cpu().numpy()
cc = 0
#%%

NC = sta.shape[3]
if cc >= NC:
    cc = 0
cc = 37
print(cc)
w = sta[:,:,:,cc]
w = (w - np.min(w)) / (np.max(w)-np.min(w))
# w = w[::2,:,:]
plt.figure(figsize=(10,3))
for i in range(w.shape[0]):
    plt.subplot(1,w.shape[0],i+1)
    plt.imshow(w[i,:,:], vmin=0, vmax=1, interpolation='Nearest')
    plt.axis("off")
cc +=1

#%% plot energy to find cropping indicies
w = np.sum(np.std(sta, axis=0), axis=2)
plt.imshow(w)

#%% get train/validation set
n_val = np.floor(len(gd)/5).astype(int)
n_train = (len(gd)-n_val).astype(int)

gd_train, gd_val = random_split(gd, lengths=[n_train, n_val])

# build dataloaders
bs = 1000
train_dl = DataLoader(gd_train, batch_size=bs)
valid_dl = DataLoader(gd_val, batch_size=bs)

sample = gd_train[:]
mean_rates = sample['robs'].mean(dim=0)

# %% Initialize NIM core
from V1FreeViewingCode.models.encoders import Encoder
from V1FreeViewingCode.models.cores import NimCore, Stacked2dEICore, Stacked2dGqmCore
from V1FreeViewingCode.models.readouts import FullReadout,FullGaussian2d


#%% Model 1: 
input_size = (gd.num_lags, gd.NY, gd.NX)
hidden_channels = 20
core = NimCore(input_size, hidden_channels,
    gamma_hidden=.01, gamma_input=0.1, skip=0,
    elu_yshift=1.0, group_norm=True,
    weight_norm=True,
    final_nonlinearity=True,
    ei_split=10,
    layers=2,
    laplace_padding=1,
    input_regularizer="GaussianLaplaceL2Adaptive",
    bias=False)
    
readout = FullReadout(core.outchannels, gd.NC,
    constrain_positive=True,
    gamma_readout=.01)

readout.features.bias.data = mean_rates

model = Encoder(core, readout, train_shifter=False,
    weight_decay=.01, optimizer='AdamW', learning_rate=.01,
    betas=[.9, .999], amsgrad=True)

#%% Model 2: convolutional model
input_size = (gd.num_lags, gd.NY, gd.NX)
input_channels = gd.num_lags
hidden_channels = 10
input_kern = 15
hidden_kern = 21
core = Stacked2dEICore(input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        layers=1,
        gamma_hidden=1e-6,
        gamma_input=1e-4,
        gamma_center=0,
        prop_inh=0,
        skip=0,
        final_nonlinearity=False,
        bias=False,
        pad_input=True,
        hidden_padding=hidden_kern//2,
        group_norm=True,
        num_groups=2,
        weight_norm=True,
        hidden_dilation=1,
        laplace_padding=1,
        input_regularizer="GaussianLaplaceL2Adaptive",
        stack=None,
        use_avg_reg=True)

# eye shifter is a perceptron
shifter = nn.Sequential(
          nn.Linear(2,20),
          nn.ReLU(),
          nn.Linear(20,2))


# initialize input layer to be centered
import V1FreeViewingCode.models.regularizers as regularizers
regw = regularizers.gaussian2d(input_kern,sigma=input_kern//4)
core.features[0].conv.weight.data = torch.einsum('ijkm,km->ijkm', core.features[0].conv.weight.data, torch.tensor(regw))

# initialize hidden layer to be centered and positive
# regw = regularizers.gaussian2d(hidden_kern,sigma=hidden_kern//4)
# core.features[1].conv.weight.data = torch.einsum('ijkm,km->ijkm', core.features[1].conv.weight.data.abs(), torch.tensor(regw))
    
# Readout
in_shape = [core.outchannels, gd.NY, gd.NX]
bias = True
readout = FullGaussian2d(in_shape, gd.NC, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
                 align_corners=False, gauss_type='uncorrelated', grid_mean_predictor=None,
                 constrain_positive=True,
                 shared_features=None, shared_grid=None, source_grid=None)

readout.bias.data = mean_rates

# combine core and readout into model
model = Encoder(core, readout, shifter=shifter, train_shifter=True,
    weight_decay=.01, optimizer='AdamW', learning_rate=.01,
    betas=[.9, .999], amsgrad=True)#, val_loss=Corr())

# %% Test training

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pathlib import Path

save_dir='./test_cores_devel'
save_dir = Path(save_dir)
version = 16

# Train
early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0)
checkpoint_callback = ModelCheckpoint(monitor='val_loss')

logger = TestTubeLogger(
    save_dir=save_dir,
    name=gd.id,
    version=version  # fixed to one to ensure checkpoint load
)

ckpt_folder = save_dir / sessid / 'version_{}'.format(version) / 'checkpoints'

trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
    checkpoint_callback=checkpoint_callback,
    logger=logger,
    deterministic=False,
    gradient_clip_val=0,
    accumulate_grad_batches=1,
    progress_bar_refresh_rate=20,
    max_epochs=1000,
    auto_lr_find=False)

# seed_everything(42)

# trainer.tune(model, train_dl, valid_dl)
trainer.fit(model, train_dl, valid_dl)
# %%
model.core.plot_filters()

#%%
def plot_filters(model, sort=False):
        import numpy as np
        import matplotlib.pyplot as plt  # plotting
        # ei_mask = model.features.layer0.eimask.ei_mask.detach().cpu().numpy()
        
        w = model.features.layer0.conv.weight.detach().cpu().numpy()
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


plot_filters(model.core)

#%% plot readout
xy = model.readout.mu.detach().cpu().numpy().squeeze()
plt.plot(xy[:,0], xy[:,1], '.')
# ix = np.where(l2>.05)[0]
# plt.plot(xy[ix,0], xy[ix,1], '.')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Readout')

#%% plot neuron RF at

model.readout.features.squeeze().shape
# model.core.features.layer0.conv.weight.shape

w = torch.einsum('nchw,nm->chwm', model.core.features.layer0.conv.weight.cpu(), model.readout.features.squeeze().cpu())


#%%
cc += 1
print(cc)
wtmp = w[:,:,:,cc].detach().cpu().numpy()
wtmp = (wtmp - np.min(wtmp)) / (np.max(wtmp)-np.min(wtmp))
plt.figure(figsize=(10,2))
for lag in range(nlags):
    plt.subplot(1,nlags,lag+1)
    plt.imshow(wtmp[lag,:,:], aspect='auto', vmin=0, vmax=1)
    plt.axis("off")

#%%
NC = w.shape[3]
nlags = w.shape[0]
plt.figure(figsize=(20,20))
for cc in range(NC):
    wtmp = w[:,:,:,cc].detach().cpu().numpy()
    wtmp = (wtmp - np.min(wtmp)) / (np.max(wtmp)-np.min(wtmp))
    for lag in range(nlags):
        plt.subplot(NC, nlags, cc*nlags + lag + 1)
        plt.imshow(wtmp[lag,:,:], aspect='auto')

    


#%% plot shifter
xx,yy = np.meshgrid(np.linspace(-5, 5,100),np.linspace(-5, 5,100))
xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))
# ygrid = torch.tensor( xx.astype('float32').flatten()))
inputs = torch.cat( (xgrid,ygrid), dim=1)
# inputs = torch.cat( (xgrid.repeat((1, 10)),ygrid.repeat((1,10))), dim=1)
shifter = model.shifter
shifter.cpu()
y = shifter(inputs)

y2 = y.detach().cpu().numpy()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(y2[:,0].reshape((100,100)))
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(y2[:,1].reshape((100,100)))
plt.colorbar()


#%%
import torch.nn.functional as F
affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])

shift = shifter(sample['eyepos']).detach()
aff = torch.tensor([[1,0,0],[0,1,0]])

affine_trans = shift[:,:,None]+aff[None,:,:]
affine_trans[:,0,0] = 1
affine_trans[:,0,1] = 0
affine_trans[:,1,0] = 0
affine_trans[:,1,1] = 1

#%%
im = sample['stim'].detach().clone()
n = im.shape[0]
grid = F.affine_grid(affine_trans, torch.Size((n, gd.num_lags, gd.NY,gd.NX)), align_corners=True)

im2 = F.grid_sample(im, grid)

#%%
stas = torch.einsum('nlwh,nc->lwhc', im2, sample['robs'])
sta2 = stas.detach().cpu().numpy()
cc = 0
#%%

NC = sta.shape[3]
if cc >= NC:
    cc = 0
# cc = 37
print(cc)
w = sta[:,:,:,cc]
w = (w - np.min(w)) / (np.max(w)-np.min(w))
w2 = sta2[:,:,:,cc]
w2 = (w2 - np.min(w2)) / (np.max(w2)-np.min(w2))

plt.figure(figsize=(10,2))
for i in range(w.shape[0]):
    plt.subplot(2,w.shape[0],i+1)
    plt.imshow(w[i,:,:], vmin=0, vmax=1, interpolation='Nearest')
    plt.axis("off")
    plt.subplot(2,w.shape[0],w.shape[0]+i+1)
    plt.imshow(w2[i,:,:], vmin=0, vmax=1, interpolation='Nearest')
    plt.axis("off")

cc +=1

# plt.imshow(im.numpy())

# grid = normalize(grid)
#%%
def shift_stim(Stim, xshift, yshift, nearest=False):
    """
        loop over frames and shift stimulus
        Stim is N x lags x H x W
    """

    from scipy.ndimage.interpolation import interp2d
    from copy import deepcopy
    
    StimC = deepcopy(Stim)
    dims = StimC.shape

    NT = dims[0]
    nlags = dims[1]
    NX = dims[3]
    NY = dims[2]

    if nearest:
        xax = np.arange(0,NX,1)
        yax = np.arange(0,NY,1)
        xx,yy = np.meshgrid(xax,yax)
    else:
        xax = np.arange(0,NX,1) - NX/2
        yax = np.arange(0,NY,1) - NY/2

    for iFrame in tqdm(range(NT)):
        I = np.reshape(Stim[iFrame,:], (NY, NX))

        if nearest:
            xind = np.minimum(np.maximum(xx + int(np.round(xshift[iFrame])), 0), NX-1)
            yind = np.minimum(np.maximum(yy + int(np.round(yshift[iFrame])), 0), NY-1)
            StimC[iFrame,:] = I[yind, xind].flatten()
        else:
            imshifter = interp2d(xax, yax, I)
            StimC[iFrame,:] = imshifter(xax+xshift[iFrame],yax+yshift[iFrame]).flatten()
    return StimC



#%% if load
def find_best_epoch(ckpt_folder):
    # from os import listdir
    # import glob
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    try:
        # ckpt_files = listdir(ckpt_folder)  # list of strings
        ckpt_files = list(ckpt_folder.glob('*.ckpt'))
        epochs = [int(str(filename)[str(filename).find('=')+1:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
        out = max(epochs)
    except FileNotFoundError:
        out = None
    return out