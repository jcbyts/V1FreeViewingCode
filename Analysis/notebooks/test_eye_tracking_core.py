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

stas = torch.einsum('nlwh,nc->lwhc', sample['stim'], sample['robs']-sample['robs'].mean(dim=0))
sta = stas.detach().cpu().numpy()
cc = 0
#%% plot individual STAs
NC = sta.shape[3]
if cc >= NC:
    cc = 0
cc=7
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
    # plt.figure() # plotting for debugging
    # plt.imshow(wt, extent=(-1,1,-1,1))
    # plt.plot(mu[cc,0],mu[cc,1], 'or')

plt.figure()
plt.plot(mu[:,0], mu[:,1], '.')

#%%
from V1FreeViewingCode.models.encoders import Encoder
# from V1FreeViewingCode.models.cores import Stacked2dEICore
import V1FreeViewingCode.models.cores as cores
import importlib
importlib.reload(cores)
from V1FreeViewingCode.models.readouts import Point2DGaussian
import V1FreeViewingCode.models.regularizers as regularizers
importlib.reload(regularizers)
import V1FreeViewingCode.models.utils as ut
importlib.reload(ut)
save_dir='./test_cores_devel'
# %% Initialize NIM core
for version in range(52,60):
    

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
    readout = Point2DGaussian(in_shape, gd.NC, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
                    align_corners=True, gauss_type='uncorrelated',
                    constrain_positive=False,
                    shifter= {'hidden_features': 20,
                            'hidden_layers': 1,
                            'final_tanh': False}
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
    out.shape

    #% Train

    # version 32 is $$
    trainer, train_dl, valid_dl = ut.get_trainer(gd, version=version,
                save_dir=save_dir,
                name=gd.id,
                auto_lr=False,
                batchsize=1000)

    trainer.fit(model, train_dl, valid_dl)

#%% check if loading possible
loadversion = 56
trainer, train_dl, valid_dl = ut.get_trainer(gd, version=loadversion,
                save_dir=save_dir,
                name=gd.id,
                auto_lr=False,
                batchsize=1000)

from V1FreeViewingCode.models.utils import find_best_epoch

ckpt_folder = trainer.logger.save_dir / trainer.logger.name / 'version_{}'.format(loadversion) / 'checkpoints'
best_epoch = find_best_epoch(ckpt_folder)

if best_epoch is not None:
    print("Loading version %d, epoch %d" %(loadversion, best_epoch))
    chkpath = str(ckpt_folder / 'epoch={}.ckpt'.format(best_epoch))
    model2 = Encoder.load_from_checkpoint(chkpath, strict=False)
    # shifter = model.readout.shifter

# model.readout.shifter = shifter


#%% remove weight norm
nn.utils.remove_weight_norm(model2.core.features.layer0.conv)
nn.utils.remove_weight_norm(model2.core.features.layer1.conv)
#%% Train

# model.readout._mu.requires_grad=False # freeze readout position
trainer.fit(model, train_dl, valid_dl)

# %%
# nn.utils.remove_weight_norm(model.core.features.layer0.conv)
# nn.utils.remove_weight_norm(model.core.features.layer1.conv)
model2.core.plot_filters()

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


#%% plot pooling layer
w = model2.core.features.layer1.conv.weight.detach().cpu().numpy()
plt.figure(figsize=(15,15))
for i in range(w.shape[0]):
    for j in range(w.shape[1]):
        plt.subplot(w.shape[0], w.shape[1], i*w.shape[0]+j+1)
        plt.imshow(w[i,j,:,:], aspect='auto')
        plt.title(i)
        plt.axis("off")

#%%
i = 5
j = 2
plt.imshow(w[i,j,:,:], aspect='auto')
plt.colorbar()
plt.title(i)
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
# wtmp = (wtmp - np.min(wtmp)) / (np.max(wtmp)-np.min(wtmp))
plt.figure(figsize=(10,2))
for lag in range(nlags):
    plt.subplot(1,nlags,lag+1)
    plt.imshow(wtmp[lag,:,:], aspect='auto')#, vmin=0, vmax=1)
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

shifter = model2.readout.shifter
# shifter = torch.load("firstshifter.pt")
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

sample = gd[:]
shift = shifter(sample['eyepos']).detach()
shift = shift - shift.mean(dim=0)
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
    w = (w -np.mean(w) )/ np.std(w)
    # w = (w - np.min(w)) / (np.max(w)-np.min(w))
    w2 = sta2[:,:,:,cc]
    w2 = (w2 -np.mean(w2) )/ np.std(w2)
    # w2 = (w2 - np.min(w2)) / (np.max(w2)-np.min(w2))

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

#%%
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



#%%
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
    preload=True,
    shifter=model2.readout.shifter)

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

#%%
sample = gd_test[:]
l2 = get_null_adjusted_ll(model, sample)

plt.plot(l2, '-o')
#%% try to refine shifter

"""
Refine eyetracking takes a few steps
1: fix shifter / first layer of core, retrain pooling weights at position on screen with most data
2: 
"""
eyepos = gd.eyepos.detach().numpy()
cnt = plt.hist2d(eyepos[:,0], eyepos[:,1], bins=20)
mx = np.max(cnt[0])
i,j=np.where(cnt[0] == mx)
x0 = cnt[1][i[0]]
y0 = cnt[2][j[0]]
plt.plot(x0, y0, 'or')
print("x: %.2f, y: %.2f, n: %d" %(x0,y0,mx))

ix = np.hypot(eyepos[:,0]-x0, eyepos[:,1]-y0)<1
np.sum(ix)

# copy model by saving and loading (is there a better way?)
torch.save(model, "temp_eyetracking_model2.pt")

#%% Load model
model2 = torch.load("temp_eyetracking_model2.pt")
#%%

gd_test = PixelDataset(sessid, stims=["Gabor"],
    stimset="Test", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    include_eyepos=True,
    preload=True,
    shifter=None)

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

gd_restricted = dd.PixelDataset(sessid, stims=["Gabor"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=2,
    valid_eye_ctr=(0.0,0.0),
    include_eyepos=True,
    preload=True)
#%% retrain 

# don't train first layer
model2.core.features.layer0.conv.weight.requires_grad = False
model2.core.features.layer0.norm.weight.requires_grad = False
model2.core.features.layer0.norm.bias.requires_grad = False

# train pooling layer
model2.core.features.layer1.conv.weight.requires_grad = True

# do train readout
model2.readout.features.requires_grad = True
model2.readout._mu.requires_grad = True
model2.readout.sigma.requires_grad = False
model2.readout.bias.requires_grad = False

# don't train shifter
for i in range(len(model2.readout.shifter)):
    model2.readout.shifter[i].linear.weight.requires_grad=False
    if type(model2.readout.shifter[i].linear.bias) != type(None):
        model2.readout.shifter[i].linear.bias.requires_grad=False

#%% get trainer

save_dir='./test_cores_refine'
version = 35
# version 32 is $$
trainer, train_dl_r, valid_dl_r = ut.get_trainer(gd_restricted, version=version,
            save_dir=save_dir,
            name=gd.id,
            auto_lr=False,
            batchsize=1000)

trainer.fit(model2, train_dl_r, valid_dl_r)

#%% now unfreeze shifter

# don't train first layer
model2.core.features.layer0.conv.weight.requires_grad = False
model2.core.features.layer0.norm.weight.requires_grad = True
model2.core.features.layer0.norm.bias.requires_grad = False

# train pooling layer
model2.core.features.layer1.conv.weight.requires_grad = False

# do train readout
model2.readout.features.requires_grad = True
model2.readout._mu.requires_grad = False
model2.readout.sigma.requires_grad = False
model2.readout.bias.requires_grad = False

# don't train shifter
for i in range(len(model2.readout.shifter)):
    model2.readout.shifter[i].linear.weight.requires_grad=True
    if type(model2.readout.shifter[i].linear.bias) != type(None):
        model2.readout.shifter[i].linear.bias.requires_grad=True

trainer, train_dl, valid_dl = ut.get_trainer(gd, version=version*10,
            save_dir=save_dir,
            name=gd.id,
            auto_lr=False,
            batchsize=1000)

trainer.fit(model2, train_dl, valid_dl)    

#%%
gd_shift = dd.PixelDataset(sessid, stims=["Gabor"],
    stimset="Train", num_lags=num_lags,
    downsample_t=t_downsample,
    downsample_s=1,
    valid_eye_rad=5.2,
    valid_eye_ctr=(0.0,0.0),
    include_eyepos=True,
    shifter=model2.readout.shifter,
    preload=True)
# %% reload sample and compute STAs
sample = gd_shift[:] # load sample 

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
import V1FreeViewingCode.models.regularizers as reg
import importlib
importlib.reload(reg)

# %%

rmat = reg.RegMats(list(model.core.features.layer0.conv.weight.shape[1:]), type=['d2xt', 'center'], amount=[.5, .5])
# %%
rmat(model.core.features.layer0.conv.weight)
# %%

# %%
