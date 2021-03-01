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


#%% load data

from V1FreeViewingCode.Analysis.notebooks.Datasets import GratingDataset, PixelDataset
from torch.utils.data import Dataset, DataLoader, random_split

sessid = 18
sessid = '20200304'

# get STAs
gd = PixelDataset(sessid, num_lags=10, augment=None)

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

#%% Get Data Loaders

# augment the data with additive guassian noise
aug = [
    {'type': 'gaussian',
    'scale': 1,
    'proportion': 1}
    ]

# {'type': 'dropout',
#     'scale': .5,
#     'proportion': .1}

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
from V1FreeViewingCode.models.basic import LNP,sNIM,cNIM,sGQM,seGQM
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


D_in = gd.NX*gd.NY*gd.num_lags
# last_model = LNP.load_from_checkpoint(checkpoint_path="./checkpoints/no_augment.ckpt")
lnp0 = LNP(input_dim=D_in, output_dim=gd.NC, learning_rate=.001, betas=[.9,.999],
    optimizer='AdamW')

snim0 = sNIM(input_dim=D_in, n_hidden=gd.NC*2, output_dim=gd.NC,
    learning_rate=.001,betas=[.9,.999],
    weight_decay=1e-1,
    normalization=2,
    optimizer='AdamW',
    ei_split=gd.NC//2)

sgqm0 = sGQM(input_dim=D_in, n_hidden=gd.NC*2, output_dim=gd.NC,
    learning_rate=.001,betas=[.9,.999],
    weight_decay=1e-0,
    normalization=2,
    relu = True,
    filternorm = 0,
    optimizer='AdamW',
    ei_split=gd.NC)

input_dims = (gd.num_lags, gd.NY, gd.NX)
cnim0 = cNIM(input_dim=input_dims,
    n_hidden=10,
    output_dim=gd.NC,
    learning_rate=.003,betas=[.9,.999],
    weight_decay=1e-1,
    n_temporal=2,
    normalization=1,
    optimizer='AdamW',
    l1reg=.0001,
    ei_split=2)

# snim0 = sNIM.load_from_checkpoint('./checkpoints/lightning_logs/version_27/checkpoints/epoch=54.ckpt')
#%% 
from pytorch_lightning.loggers import TestTubeLogger

version = 20
logger = TestTubeLogger(
    save_dir='./checkpoints',
    name='model_compare',
    version=version,  # An existing version with a saved checkpoint
)

early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0)
checkpoint_callback = ModelCheckpoint(monitor='val_loss')

#%%
# seed_everything(42)

# default_root_dir='./checkpoints',

# train shared NIM
trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
    checkpoint_callback=checkpoint_callback,
    logger=logger,
    deterministic=False,
    progress_bar_refresh_rate=20,
    max_epochs=1000,
    auto_lr_find=True)

trainer.tune(cnim0, train_dl, valid_dl) # find learning rate
trainer.fit(cnim0, train_dl, valid_dl)


trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
    checkpoint_callback=checkpoint_callback,
    logger=logger,
    deterministic=False,
    progress_bar_refresh_rate=20,
    max_epochs=1000,
    auto_lr_find=True)

trainer.tune(snim0, train_dl, valid_dl) # find learning rate
trainer.fit(snim0, train_dl, valid_dl)

#%%

version += 5
logger = TestTubeLogger(
    save_dir='./checkpoints',
    name='model_compare',
    version=version,  # An existing version with a saved checkpoint
)

trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
    checkpoint_callback=checkpoint_callback,
    logger=logger,
    deterministic=False,
    progress_bar_refresh_rate=20,
    max_epochs=1000,
    auto_lr_find=True)

trainer.tune(sgqm0, train_dl, valid_dl) # find learning rate
trainer.fit(sgqm0, train_dl, valid_dl)


#%%

# trainer.save_checkpoint("./checkpoints/augment.ckpt")

input_dims = (gd.num_lags, gd.NY, gd.NX)
cmod = cNIM(input_dim=input_dims, n_hidden = 10, output_dim = gd.NC)
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

l1 = get_null_adjusted_ll(snim0, xt, yt, bits=False)
l2 = get_null_adjusted_ll(sgqm0, xt, yt,bits=False)

print(l2)

# plt.plot(l1)
plt.plot(l2)
plt.axhline(0)


#%%

gqm0 = sGQM.load_from_checkpoint('./checkpoints/model_compare/version_25/checkpoints/epoch=26.ckpt')

from pytorch_lightning import LightningModule
from V1FreeViewingCode.models.basic import Poisson
import torch.nn as nn
input_dims = (gd.num_lags, gd.NY, gd.NX)

# w1 = gqm0.linear1.weight.detach().cpu().numpy()
# w2 = gqm0.readout.weight.detach().cpu().numpy()

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
        x = self.readout(x)
        x = x.reshape(sz[0], sz[2], sz[3], self.hparams.output_dim)
        # x = x.permute((0, 3, 1, 2))
        return x

cmod = convGQM(input_dim=input_dims, gqm=gqm0)

#%%
gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=None, cids=cids)

loss = nn.PoissonNLLLoss(log_input=False, reduction='none')

#%%
impo
nbins = 20
locs = np.arange(-5, 5, nbins)
xx,yy = np.meshgrid(locs)


ctrXY = (-3,0)
inds = np.where(np.hypot(gd.eyeX[gd.valid]-ctrXY[0], gd.eyeY[gd.valid] - ctrXY[1]) < 1)[0]

x,y = gd[inds]

xh = cmod(x)
sz = list(xh.size())

NC = gd.NC
L = 0
for cc in range(NC):
    L += loss(y[:,cc].repeat(sz[1]*sz[2]).reshape((sz[0],sz[1],sz[2])), xh[:,:,:,cc]).sum(axis=0)

L = L.detach().cpu().numpy()
plt.imshow(-L)

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
w2 = sgqm0.readout.weight.detach().cpu().numpy()

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


plt.figure()
tk = cnim0.temporal.linear.weight.detach().cpu().numpy()
plt.plot(tk.T)

w = cnim0.conv1.weight.detach().cpu().numpy()
n = w.shape[0]
plt.figure(figsize=(10,10))
sx,sy = U.get_subplot_dims(n)
for cc in range(n):
    plt.subplot(sx,sy,cc+1)
    wtmp = w[cc,:,:,:].squeeze().reshape(w.shape[-2], -1)
    plt.imshow(wtmp, aspect='auto')

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