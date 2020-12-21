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
sessid = '20200304'
gd = PixelDataset(sessid, stims=["Gabor"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    include_eyepos=True,
    preload=True)

#%% compute STAS
sample = gd[:] # load sample 

stas = torch.einsum('nlwh,nc->lwhc', sample['stim'].pow(2), sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()
cc = 0
#%% plot individual STAs
NC = sta.shape[3]
if cc >= NC:
    cc = 0

print(cc)
w = sta[:,:,:,cc]
w = (w - np.min(w)) / (np.max(w)-np.min(w))
plt.figure(figsize=(10,3))
for i in range(w.shape[0]):
    plt.subplot(1,w.shape[0],i+1)
    plt.imshow(w[i,:,:], vmin=0, vmax=1, interpolation='Nearest')
    plt.axis("off")
cc +=1

#%% calculate RF locations to initialize readout
mu = np.zeros((NC,2))
for cc in range(NC):
    w = sta[:,:,:,cc]
    wt = np.std(w, axis=0)
    wt /= np.max(np.abs(wt)) # normalize for numerical stability
    # softmax
    wt = wt**50
    wt /= np.sum(wt)
    sz = wt.shape
    xx,yy = np.meshgrid(np.linspace(-1, 1, sz[1]), np.linspace(1, -1, sz[0]))

    mu[cc,0] = np.sum(xx*wt) # center of mass after softmax
    mu[cc,1] = np.sum(yy*wt)
    # plt.figure()
    # plt.imshow(wt, extent=(-1,1,-1,1))
    # plt.plot(mu[cc,0],mu[cc,1], 'or')

plt.figure()
plt.plot(mu[:,0], mu[:,1], '.')
# %% Initialize NIM core

from V1FreeViewingCode.models.encoders import Encoder
# from V1FreeViewingCode.models.cores import Stacked2dEICore
import V1FreeViewingCode.models.cores as cores
import importlib
importlib.reload(cores)

from V1FreeViewingCode.models.readouts import Point2DGaussian

#% Model: convolutional model
input_size = (gd.num_lags, gd.NY, gd.NX)
input_channels = gd.num_lags
hidden_channels = 20
input_kern = 19
hidden_kern = 21
core = cores.Stacked2dEICore(input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        layers=1,
        gamma_hidden=0,
        gamma_input=0,
        gamma_center=0,
        prop_inh=0,
        skip=0,
        final_nonlinearity=False,
        bias=False,
        pad_input=True,
        hidden_padding=hidden_kern//2,
        group_norm=False,
        num_groups=2,
        weight_norm=True,
        hidden_dilation=1,
        laplace_padding=1,
        input_regularizer="GaussianLaplaceL2Adaptive",
        stack=None,
        use_avg_reg=True)


# initialize input layer to be centered
import V1FreeViewingCode.models.regularizers as regularizers
regw = regularizers.gaussian2d(input_kern,sigma=input_kern//4)
core.features[0].conv.weight.data = torch.einsum('ijkm,km->ijkm', core.features[0].conv.weight.data, torch.tensor(regw))
    
# Readout
in_shape = [core.outchannels, gd.NY, gd.NX]
bias = True
readout = Point2DGaussian(in_shape, gd.NC, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
                 align_corners=False, gauss_type='uncorrelated',
                 constrain_positive=True,
                 shifter={'hidden_features': 20,
                         'hidden_layers': 1,
                         'final_tanh': False}
                         )

readout.bias.data = sample['robs'].mean(dim=0) # initialize readout bias helps
readout._mu.data[0,:,0,:] = torch.tensor(mu.astype('float32')) # initiaalize mus

# combine core and readout into model
model = Encoder(core, readout,
    weight_decay=.001, optimizer='AdamW', learning_rate=.01, # high initial learning rate because we decay on plateau
    betas=[.9, .999], amsgrad=True)

#%% check that the forward runs (for debugging)
out = model(sample['stim'][:10,:], shifter=sample['eyepos'][:10,:])
out.shape
#%% Train
import V1FreeViewingCode.models.utils as ut
import importlib
importlib.reload(ut)
save_dir='./test_cores_devel'
version = 13

trainer, train_dl, valid_dl = ut.get_trainer(gd, version=version,
            save_dir=save_dir,
            name=gd.id,
            auto_lr=False,
            batchsize=1000)

#%% check if loading possible
from V1FreeViewingCode.models.utils import find_best_epoch
loadversion = 2
ckpt_folder = trainer.logger.save_dir / trainer.logger.name / 'version_{}'.format(loadversion) / 'checkpoints'
best_epoch = find_best_epoch(ckpt_folder)

if best_epoch is not None:
    print("Loading version %d, epoch %d" %(loadversion, best_epoch))
    chkpath = str(ckpt_folder / 'epoch={}.ckpt'.format(best_epoch))
    model = Encoder.load_from_checkpoint(chkpath, strict=False)
    # shifter = model.readout.shifter

# model.readout.shifter = shifter

#%% Train

# model.readout._mu.requires_grad=False # freeze readout position
trainer.fit(model, train_dl, valid_dl)
#%%

model.readout._mu.requires_grad=True
trainer.fit(model, train_dl, valid_dl)



# %%
# nn.utils.remove_weight_norm(model.core.features.layer0.conv)
model.core.plot_filters()

#%% plot readout
xy = model.readout.mu.detach().cpu().numpy().squeeze()
plt.plot(xy[:,0], xy[:,1], '.')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Readout')

#%%
plt.plot(xy[:,1], mu[:,1], '.')
#%% plot neuron RF at its center

w = torch.einsum('nchw,nm->chwm', model.core.features.layer0.conv.weight.cpu(), model.readout.features.squeeze().cpu())


#%%
cc = 7
print(cc)
wtmp = w[:,:,:,cc].detach().cpu().numpy()
nlags = w.shape[0]
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

xx,yy = np.meshgrid(np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,100),np.linspace(-gd.valid_eye_rad, gd.valid_eye_rad,100))
xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

inputs = torch.cat( (xgrid,ygrid), dim=1)

shifter = model.readout.shifter
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

#%% shift stimulus
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
sample = gd[:]
im = sample['stim'].detach().clone()
n = im.shape[0]
grid = F.affine_grid(affine_trans, torch.Size((n, gd.num_lags, gd.NY,gd.NX)), align_corners=True)

im2 = F.grid_sample(im, grid)

#%% new STAs on shifted stimulus
stas = torch.einsum('nlwh,nc->lwhc', im, sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()
stas = torch.einsum('nlwh,nc->lwhc', im2, sample['robs']-sample['robs'].mean(dim=0))
sta2 = stas.detach().cpu().numpy()
cc = 0
#%%

NC = sta.shape[3]
plt.figure(figsize=(8,NC*2))
for cc in range(NC):
    w = sta[:,:,:,cc]
    # w = (w - np.min(w)) / (np.max(w)-np.min(w))
    w2 = sta2[:,:,:,cc]
    # w2 = (w2 - np.min(w2)) / (np.max(w2)-np.min(w2))

    bestlag = np.argmax(np.std(w2.reshape( (gd.num_lags, -1)), axis=1))
    plt.subplot(NC,4, cc*4 + 1)
    plt.imshow(w2[bestlag,:,:], aspect='auto')
    plt.subplot(NC,4, cc*4 + 2)
    i,j=np.where(w2[bestlag,:,:]==np.max(w2[bestlag,:,:]))
    plt.plot(w2[:,i[0], j[0]], '-ob')
    i,j=np.where(w2[bestlag,:,:]==np.min(w2[bestlag,:,:]))
    plt.plot(w2[:,i[0], j[0]], '-or')
    yd = plt.ylim()

    plt.subplot(NC,4, cc*4 + 3)
    plt.imshow(w[bestlag,:,:], aspect='auto')
    plt.subplot(NC,4, cc*4 + 4)
    i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-ob')
    i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
    plt.plot(w[:,i[0], j[0]], '-or')
    plt.ylim(yd)

#%%
if cc >= NC:
    cc = 0

print(cc)
cc = 7
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



#%%
fr = 2
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(w[fr,:,:], interpolation='None', extent=(0,70/gd.ppd*60,0,70/gd.ppd*60), cmap="gray")
plt.xlabel("arcmin")
plt.ylabel("arcmin")

plt.subplot(122)
plt.imshow(w2[fr,:,:], interpolation='None', extent=(0,70/gd.ppd*60,0,70/gd.ppd*60), cmap="gray")
plt.xlabel("arcmin")
plt.ylabel("arcmin")

#%%
plt.figure(figsize=(10,5))
plt.subplot(121)
f = plt.plot(w.reshape(10,-1))
plt.subplot(122)
f = plt.plot(w2.reshape(10,-1))


#%% if load
gd_test = PixelDataset(sessid, stims=["Gabor"],
    stimset="Test", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    include_eyepos=True,
    preload=True)

# %%
def get_null_adjusted_ll(model, sample, bits=False):
    '''
    get null-adjusted log likelihood
    bits=True will return in units of bits/spike
    '''
    m0 = model.cpu()
    loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
    lnull = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy().sum(axis=0)
    yhat = m0(sample['stim'], shifter=sample['eyepos'])
    llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
    rbar = sample['robs'].sum(axis=0).numpy()
    ll = (llneuron - lnull)/rbar
    if bits:
        ll/=np.log(2)
    return ll

sample = gd_test[:]
l2 = get_null_adjusted_ll(model, sample)

plt.plot(l2, '-o')
# %%



#%%
#%% Load dataset while applying shifter
import V1FreeViewingCode.models.datasets as dd
import importlib
importlib.reload(dd)

num_lags = 10
t_downsample = 2 # temporal downsampling
sessid = '20200304'
gd = dd.PixelDataset(sessid, stims=["Gabor"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    include_eyepos=True,
    preload=True,
    cropidx=[(20,50),(15,45)],
    shifter=model.readout.shifter)


# %% reload sample and compute STAs
sample = gd[:] # load sample 

stas = torch.einsum('nlwh,nc->lwhc', sample['stim'], sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()
cc = 0
#%%

NC = sta.shape[3]
if cc >= NC:
    cc = 0

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

# %%
plt.imshow(w[4,:,:])

# plt.imshow(stas.std(dim=0).sum(dim=2))
# %%
