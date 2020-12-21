# %% Import libraries
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

# import deepdish as dd
import V1FreeViewingCode.Analysis.notebooks.Utils as U
import V1FreeViewingCode.Analysis.notebooks.gratings as gt

import numpy as np

import torch

import matplotlib.pyplot as plt  # plotting
import seaborn as sns

import V1FreeViewingCode.Analysis.notebooks.neureye as ne

#%% load data

from V1FreeViewingCode.Analysis.notebooks.Datasets import GratingDataset, PixelDataset
from torch.utils.data import Dataset, DataLoader, random_split

sessid = '20200304'


#%%
save_dir = './checkpoints'

def find_best_epoch(ckpt_folder):
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    ckpt_files = listdir(ckpt_folder)  # list of strings
    epochs = [int(filename[6:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
    return max(epochs)

version = 1
logger = TestTubeLogger(
    save_dir='./checkpoints',
    name=sessid,
    version=version  # fixed to one to ensure checkpoint load
)

ckpt_folder = save_dir / sessid / 'version_{}'.format(version) / 'checkpoints'
best_epoch = find_best_epoch(ckpt_folder)
self.trainer = ptl.Trainer(
    logger=logger,
    resume_from_checkpoint=str(ckpt_folder / 'epoch={}.ckpt'.format(best_epoch)),
)
#%% get STAs
gd = PixelDataset(sessid, num_lags=10, augment=None)

x,y = gd[:]
y -= y.mean(axis=0)
stas = (x.reshape(len(gd), -1).T @ y).numpy()

#%% tag good units based on fraction of large excursions in STA

# get median absolute deviation per unit
med = np.median(stas, axis=0)
devs = np.abs(stas - med)
mad = np.median(devs, axis=0) # median absolute deviation

excursions = np.mean(devs > 4*mad, axis=0)
thresh = 0.01
cids = np.where(excursions>thresh)[0]
plt.plot(excursions, '-o')
plt.plot(cids, excursions[cids], 'o')


#%% plot stas
NC = gd.NC
sx,sy = U.get_subplot_dims(NC)
sx*=2
plt.figure(figsize=(10,20))
for cc in range(NC):
    plt.subplot(sx,sy,cc*2+1)
    wtmp = stas[:,cc].reshape((gd.num_lags, -1))
    tpower = np.std(wtmp, axis=1)
    bestlag = np.argmax(tpower)
    wspace = wtmp[bestlag,:].reshape( (gd.NY, gd.NX))
    plt.imshow(wspace, aspect='auto')
    plt.axis("off")
    if cc in cids:
        plt.title(cc)
    plt.subplot(sx,sy,cc*2+2)
    if cc in cids:
        plt.plot(wtmp[:,np.argmax(wspace)], '-ob')
        plt.plot(wtmp[:,np.argmin(wspace)], '-or')
    else:        
        plt.plot(wtmp[:,np.argmax(wspace)], '-o', color=(.5, .5, .5))
        plt.plot(wtmp[:,np.argmin(wspace)], '-o', color=(.5, .5, .5))
    

print("Found %d / %d good units" %(len(cids), NC))

#%% Get Data Loaders

# augment the data with additive guassian noise
aug = [
    {'type': 'gaussian',
    'scale': 1,
    'proportion': 1}
    ]

num_lags = 10
gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=aug, cids=cids)

# test set
gd_test = PixelDataset(sessid, num_lags=num_lags,
    train=False, cids=cids)

# get train/validation set
n_val = np.floor(len(gd)/5).astype(int)
n_train = (len(gd)-n_val).astype(int)

gd_train, gd_val = random_split(gd, lengths=[n_train, n_val])

# build dataloaders
bs = 1000
train_dl = DataLoader(gd_train, batch_size=bs)
valid_dl = DataLoader(gd_val, batch_size=bs)


#%%
from V1FreeViewingCode.models.basic import sGQM
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


D_in = gd.NX*gd.NY*gd.num_lags

gqm0 = sGQM(input_dim=D_in, n_hidden=gd.NC*2, output_dim=gd.NC,
    learning_rate=.001,betas=[.9,.999],
    weight_decay=1e-0,
    normalization=2,
    relu = True,
    filternorm = 0,
    optimizer='AdamW',
    ei_split=gd.NC)

input_dims = (gd.num_lags, gd.NY, gd.NX)

#%% if training
from pytorch_lightning.loggers import TestTubeLogger

version = 20
logger = TestTubeLogger(
    save_dir='./checkpoints',
    name=sessid
)

trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
    checkpoint_callback=checkpoint_callback,
    logger=logger,
    deterministic=False,
    progress_bar_refresh_rate=20,
    max_epochs=1000,
    auto_lr_find=True)

trainer.tune(gqm0, train_dl, valid_dl) # find learning rate
trainer.fit(gqm0, train_dl, valid_dl)


#%% or load it
gqm0 = sGQM.load_from_checkpoint('./checkpoints/model_compare/version_25/checkpoints/epoch=26.ckpt')

#%% compare models

def get_null_adjusted_ll(model, xt, yt, bits=False):
    m0 = model.cpu()
    loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
    lnull = -loss(torch.ones(yt.shape)*yt.mean(axis=0), yt).detach().cpu().numpy().sum(axis=0)
    llneuron = -loss(m0(xt),yt).detach().cpu().numpy().sum(axis=0)
    rbar = yt.sum(axis=0).numpy()
    ll = (llneuron - lnull)/rbar
    if bits:
        ll/=np.log(2)
    return ll

# # get test set
xt,yt=gd_test[:]

l2 = get_null_adjusted_ll(gqm0, xt, yt,bits=False)

# print(l2)

plt.plot(l2, '-o')
plt.axhline(0)
plt.ylabel("test LL (null-adjusted)")
plt.xlabel("Unit Id")

#%% make model convolutional

from pytorch_lightning import LightningModule
from V1FreeViewingCode.models.basic import Poisson
import torch.nn as nn
input_dims = (gd.num_lags, gd.NY, gd.NX)

class convGQM(Poisson):
    def __init__(self, input_dim=(15, 8, 6),
        gqm=None,
        **kwargs):
        
        super().__init__()
        self.save_hyperparameters()
        
        ksize = (input_dim[1], input_dim[2])
        padding = (ksize[0]//2, ksize[1]//2)
        self.conv1 = nn.Conv2d(in_channels=input_dim[0],
            out_channels=gqm.hparams.n_hidden, stride=1,
            kernel_size=ksize,
            padding=padding)

        self.nl = gqm.nl
        self.relu = gqm.relu
        
        self.hparams.ei_split = gqm.hparams.ei_split
        self.spikeNL = gqm.spikeNL

        self.hparams.relu = gqm.hparams.relu
        self.hparams.normalization = gqm.hparams.normalization
        self.norm = gqm.norm

        # self.readout = nn.Linear(gqm.hparams.n_hidden, gqm.hparams.output_dim)
        self.readout = gqm.readout
        self.hparams.output_dim = gqm.hparams.output_dim

        for i in range(gqm.hparams.n_hidden):
            self.conv1.weight.data[i,:,:,:] = gqm.linear1.weight.data[i,:].reshape(input_dims)


    def forward(self,x):
        x = self.conv1(x)
        sz = list(x.size())
        x = x.permute((0,2,3,1))
        x = x.reshape((sz[0]*sz[2]*sz[3], sz[1]))
        # apply nonlinearity
        if self.hparams.relu:
            xlin = self.relu(x[:, :self.hparams.ei_split])
        else:
            xlin = x[:, :self.hparams.ei_split]
        
        x = torch.cat( (xlin, self.nl(x[:,self.hparams.ei_split:])), axis=1)
        if self.hparams.normalization>0:
            x = self.norm(x) # hmm

        x = self.spikeNL(self.readout(x))

        x = x.reshape(sz[0], sz[2], sz[3], self.hparams.output_dim)
        # x = x.permute((0, 3, 1, 2))
        return x

cmod = convGQM(input_dim=input_dims, gqm=gqm0)

#%%

#%% reload dataset with no augmentation
gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=None, cids=cids)

loss = nn.PoissonNLLLoss(log_input=False, reduction='none')


#%% test on a small amount of data
x,y = gd[:1000] # preload some data to get dimensions
xh = cmod(x) # predict rates
sz = list(xh.size())

n = np.asarray([cmod.conv1.weight[i,:].abs().max().detach().numpy() for i in range(cmod.conv1.weight.shape[0])])
cinds = np.argsort(n)[::-1][-len(n):]
cc = cinds[0]

xn = x.numpy()

wtmp = w[cc,:,:,:].squeeze()
wtmp = (wtmp - np.min(wtmp)) / (np.max(wtmp) - np.min(wtmp))
tpower = np.std(wtmp.reshape(w.shape[1], -1), axis=1)
bestlag = np.argmax(tpower)

plt.figure(figsize=(10,3))
for iframe in range(num_lags):
    plt.subplot(2,num_lags,iframe+1)
    plt.imshow(xn[0,iframe,:,:])
    plt.subplot(2,num_lags,iframe+num_lags+1)
    plt.imshow(wtmp[iframe,:,:])

plt.figure(figsize=(10,3))
plt.subplot(1,3,1) # plot stimulus
plt.imshow(xn[0,bestlag,:,:])

plt.subplot(1,3,2) # plot filter
plt.imshow(wtmp[bestlag,:,:], vmin=0, vmax=1)

plt.subplot(1,3,3) # plot convolutional output
xc = cmod.nl(cmod.conv1(x)).detach().numpy()
plt.imshow(xc[0,cc,:].squeeze())

xh = cmod(x)
# reshape and get loss across neurons over space
sz = list(xh.size())

L = 0
for cc in range(NC):
    yc = y[:,cc][:,None].repeat((1, sz[1]*sz[2])).reshape(sz[0:3])
    L += loss(xh[:,:,:,cc], yc).sum(axis=0)
L = L.detach().cpu().numpy()

plt.figure()
plt.imshow(L)
plt.xlabel("azimuth")
plt.ylabel('elevation')
plt.title("likelihood surface")


#%% Get the likelihood surface for every location on the screen

def get_likelihood_surface(gd, cmod, Npos=20, radius=1, valid_eye_range=5.2):
    from tqdm import tqdm # progress bar
    from copy import deepcopy

    assert gd.corrected is False, "cannot get an LL surface on an already corrected stimulus"
    locs = np.linspace(-valid_eye_range, valid_eye_range, Npos) # grid up space

    x,y = gd[:10] # preload some data to get dimensions
    xh = cmod(x) # predict rates
    sz = list(xh.size()) # here's the shape

    NC = gd.NC
    NY = sz[1]
    NX = sz[2]
    LLspace1 = np.zeros([Npos,Npos,NY,NX])

    # Loop over positions (this is the main time-consuming operation)
    for xx in tqdm(range(Npos)):
        for yy in range(Npos):
            ctrXY = (locs[xx],locs[yy])

            inds = np.where(np.hypot(gd.eyeX[gd.valid]-ctrXY[0], gd.eyeY[gd.valid] - ctrXY[1]) < radius)[0]

            x,y = gd[inds] # get data from those indices

            xh = cmod(x) # predict rates

            # reshape and get loss across neurons over space
            sz = list(xh.size())

            L = 0
            for cc in range(NC):
                yc = y[:,cc][:,None].repeat((1, sz[1]*sz[2])).reshape(sz[0:3])
                L += loss(xh[:,:,:,cc], yc).sum(axis=0)

            L = L.detach().cpu().numpy()

            LLspace1[xx,yy,:,:] = deepcopy(L)
        
        
    return LLspace1, locs

LLspace, locs = get_likelihood_surface(gd, cmod, Npos=20, radius=1, valid_eye_range=5.2)


#%%
centers5,LLspace3 = ne.get_centers_from_LLspace(LLspace1, softmax=10, plot=True, interpolation_steps=2, crop_edge=2)

plt.figure()
f = plt.plot(centers5[:,:,0], centers5[:,:,1], '.')

#%% plot a few

ctrid = np.argmin(locs**2)
idxs = np.arange(2, 5)
nplot = len(idxs)


plt.figure(figsize=(15,6))
for i,j in zip(idxs, range(nplot)):
    plt.subplot(1,nplot,j+1)
    irow = 5
    I = LLspace1[irow,i,:,:]
    I = 1-ne.normalize_range(I) # flip sign and normalize between 0 and 1
    Ly,Lx = I.shape

    x,y = ne.radialcenter(I**10)
    print("(%d, %d)" %(x-Lx//2,y-Ly//2))
    plt.imshow(I, aspect='equal')
    plt.axvline(Lx/2, color='w')
    plt.axhline(Ly/2, color='w')
    plt.plot(x,y, '+r')
    plt.title('(%d,%d)' %(irow,idxs[j]))

#%% Correct stimulus

# load raw again


centers6 = deepcopy(centers5)
ctrid = np.argmin(locs**2)
centers6[:,:,0]-=centers6[ctrid,ctrid,0]
centers6[:,:,1]-=centers6[ctrid,ctrid,1]

import scipy.io as sio
from pathlib import Path
fpath = Path(gd.dirname)
fname = fpath / (sessid + "_CorrGrid.mat")

sio.savemat(str(fname.resolve()), {'centers': centers6, 'locs': locs, 'LLspace': LLspace1})

#%%

gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=None, corrected=False)

xcorrec,ycorrec = ne.get_shifter_from_centers(centers6, locs*gd.ppd, maxshift=2, nearest=False)

xshift = xcorrec(gd.eyeX*gd.ppd, gd.eyeY*gd.ppd)
yshift = ycorrec(gd.eyeX*gd.ppd, gd.eyeY*gd.ppd)

ix = np.logical_or(np.isnan(xshift), np.isnan(yshift))

xshift[ix] = 0.0
yshift[ix] = 0.0

# NEAREST option shifts shifts in integer numbers of pixels
Stim = gd.x.numpy()
StimC = ne.shift_stim(Stim, xshift, yshift, [gd.NX,gd.NY], nearest=True)


#%% plot STAS
gd.x = torch.tensor(StimC)
x,y = gd[:]
y -= y.mean(axis=0)
stas = (x.reshape(len(gd), -1).T @ y).numpy()

#%% tag good units
med = np.median(stas, axis=0)
devs = np.abs(stas - med)
mad = np.median(devs, axis=0)

excursions = np.mean(devs > 4*mad, axis=0)
thresh = 0.01
cids = np.where(excursions>thresh)[0]
plt.plot(excursions, '-o')
plt.plot(cids, excursions[cids], 'o')


#%% plot stas
NC = gd.NC
sx,sy = U.get_subplot_dims(NC)
sx*=2
plt.figure(figsize=(10,20))
for cc in range(NC):
    plt.subplot(sx,sy,cc*2+1)
    wtmp = stas[:,cc].reshape((gd.num_lags, -1))
    tpower = np.std(wtmp, axis=1)
    bestlag = np.argmax(tpower)
    wspace = wtmp[bestlag,:].reshape( (gd.NY, gd.NX))
    plt.imshow(wspace, aspect='auto')
    plt.axis("off")
    if cc in cids:
        plt.title(cc)
    plt.subplot(sx,sy,cc*2+2)
    if cc in cids:
        plt.plot(wtmp[:,np.argmax(wspace)], '-ob')
        plt.plot(wtmp[:,np.argmin(wspace)], '-or')
    else:        
        plt.plot(wtmp[:,np.argmax(wspace)], '-o', color=(.5, .5, .5))
        plt.plot(wtmp[:,np.argmin(wspace)], '-o', color=(.5, .5, .5))
    

print("Found %d / %d good units" %(len(cids), NC))

# %%
# w = snim0.l1.weight.detach().cpu().numpy()
w = sgqm0.linear1.weight.detach().cpu().numpy()
# w = snim0.l1.weight.numpy()

nfilt = w.shape[0]
sx,sy = U.get_subplot_dims(nfilt)
plt.figure(figsize=(10,10))
for cc in range(nfilt):
    plt.subplot(sx,sy,cc+1)
    wtmp = np.reshape(w[cc,:], (gd.num_lags, gd.NX*gd.NY))
    bestlag = np.argmax(np.std(wtmp, axis=1))
    # bestlag = 5
    plt.imshow(np.reshape(wtmp[bestlag,:], (gd.NY, gd.NX)))
    # plt.imshow(np.reshape(w[cc,:], (gd.NX*gd.NY, gd.num_lags)), aspect='auto')
    # plt.imshow(wtmp, aspect='auto')
    # plt.plot(wtmp)

# %%
# w2 = snim0.l2.weight.detach().cpu().numpy()
w2 = cmod.readout.weight.detach().cpu().numpy()

plt.imshow(w2)
# plt.plot(w2)

# plt.plot(np.sum(w2, axis=0))
plt.figure()
f=plt.plot(w2)

#%%
import torch.nn.utils.prune as prune

a = prune.random_unstructured(sgqm0.linear1, name="weight", amount=0.3)
sgqm1 = deepcopy(sgqm0)
w = a.weight.detach().cpu().numpy()

nfilt = w.shape[0]
sx,sy = U.get_subplot_dims(nfilt)
plt.figure(figsize=(10,10))
for cc in range(nfilt):
    plt.subplot(sx,sy,cc+1)
    wtmp = np.reshape(w[cc,:], (gd.num_lags, gd.NX*gd.NY))
    bestlag = np.argmax(np.std(wtmp, axis=1))
    # bestlag = 5
    plt.imshow(np.reshape(wtmp[bestlag,:], (gd.NY, gd.NX)))


# %% cnim plot


# plt.figure()
# tk = cnim0.temporal.linear.weight.detach().cpu().numpy()
# plt.plot(tk.T)

w = cmod.conv1.weight.detach().cpu().numpy()
n = w.shape[0]
plt.figure(figsize=(10,10))
sx,sy = U.get_subplot_dims(n)
for cc in range(n):
    plt.subplot(sx,sy,cc+1)
    wtmp = w[cc,:,:,:].squeeze()
    tpower = np.std(wtmp.reshape(w.shape[1], -1), axis=1)
    bestlag = np.argmax(tpower)
    plt.imshow(wtmp[bestlag,:,:], aspect='auto')

# %% plot readout weights

NC = gd.NC
w = cnim0.readout.weight.detach().cpu().numpy()
n = cnim0.hparams.n_hidden
nx = cnim0.hparams.readoutNX
ny = cnim0.hparams.readoutNY
plt.figure(figsize=(n*2, NC*2))
for cc in range(NC):
    wtmp = w[cc,:].reshape(n,ny,nx)
    for jj in range(n):
        plt.subplot(NC, n, n*cc + jj + 1)
        plt.imshow(wtmp[jj,:,:].squeeze(), aspect='auto')

# %%
import torch.nn as nn
from plenoptic.plenoptic.simulate import Steerable_Pyramid_Freq


class PyrConv(nn.Module):
    def __init__(self, imshape, order, scales, kernel_size, exclude = [], output_dim = 10, is_complex = True):
        super(PyrConv, self).__init__()
        
        self.imshape = imshape
        self.order = order
        self.scales = scales
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.exclude = exclude
        self.is_complex = is_complex
        
        self.rect = nn.ReLU(inplace = True)
        self.pyr = Steerable_Pyramid_Freq(height=self.scales,image_shape=self.imshape,
                                          order=self.order,is_complex = self.is_complex,twidth=1, downsample=False)
        
        #channels number of scales * number of orientations * 2 (for the real/imag)
        self.pyr_channels = (self.order+1)*self.scales*2
        self.conv = nn.Conv2d(in_channels = self.pyr_channels,kernel_size = self.kernel_size, 
                              out_channels= self.output_dim, stride = 2)
        
    def forward(self, x):
        out = self.pyr(x)
        out = convert_pyr_to_tensor(out, exclude = self.exclude, is_complex = self.is_complex)
        out = self.rect(out)
        out = self.conv(out)

        return out



#%%
# pyr_conv_model = PyrConv([256,256], 3, 2,5, exclude = ['residual_lowpass', 'residual_highpass'], output_dim = 20)
# pyr_conv_resp = pyr_conv_model(im_batch)
# print(pyr_conv_resp.shape)
pyr = Steerable_Pyramid_Freq(height='auto',image_shape=[10,10],
                                          order=2,is_complex = True,twidth=1, downsample=False)
# %%

sgqm0.