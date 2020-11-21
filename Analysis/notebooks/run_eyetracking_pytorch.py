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

from V1FreeViewingCode.Analysis.notebooks.Datasets import GratingDataset, PixelDataset
from torch.utils.data import Dataset, DataLoader, random_split

#%% load data



sessid = '20200304'
# from pathlib import Path
# save_dir = Path('./checkpoints')

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
plt.xlabel("Unit Id")
plt.ylabel("Fraction significant excursions (MAD)")

#%% plot stas
NC = gd.NC
sx,sy = U.get_subplot_dims(NC)
sx*=2
plt.figure(figsize=(10,20))
ws = 0
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
        ws += np.abs(wspace)
    else:        
        plt.plot(wtmp[:,np.argmax(wspace)], '-o', color=(.5, .5, .5))
        plt.plot(wtmp[:,np.argmin(wspace)], '-o', color=(.5, .5, .5))
    

print("Found %d / %d good units" %(len(cids), NC))

plt.figure()
plt.imshow(ws, aspect='auto')
#%%
def crop_indx( Loriginal, xrange, yrange):
    # brain-dead way to crop things with space indexed by one dim
    # Note I'm calling x the horizontal dimension (as plotted by python and y the vertical direction)
    # Also assuming everything square
    indxs = []
    for nn in range(len(yrange)):
        indxs = np.concatenate((indxs, np.add(xrange,yrange[nn]*Loriginal)))
    return indxs.astype('int')

NX2 = 20
x0 = 5
y0 = 5

xinds = range(x0, x0+NX2)
yinds = range(y0, y0+NX2)
    
Cindx = crop_indx(gd.NX, xinds,yinds)

NC = gd.NC
sx,sy = U.get_subplot_dims(NC)
plt.figure(figsize=(sx,sy))
ws = 0
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    wtmp = stas[:,cc].reshape((gd.num_lags, -1))
    tpower = np.std(wtmp, axis=1)
    bestlag = np.argmax(tpower)
    wspace = wtmp[bestlag,Cindx].reshape( (NX2, NX2))
    plt.imshow(wspace, aspect='auto')
    plt.axis("off")

#%% confirm that it worked by doing proper cropping
gdcrop = PixelDataset(sessid, num_lags=10, augment=None, cropidx=[xinds,yinds])

x,y = gdcrop[:]
y -= y.mean(axis=0)
stas = (x.reshape(len(gd), -1).T @ y).numpy()

NC = gdcrop.NC
sx,sy = U.get_subplot_dims(NC)
sx*=2
plt.figure(figsize=(10,20))
ws = 0
for cc in range(NC):
    plt.subplot(sx,sy,cc*2+1)
    wtmp = stas[:,cc].reshape((gdcrop.num_lags, -1))
    tpower = np.std(wtmp, axis=1)
    bestlag = np.argmax(tpower)
    wspace = wtmp[bestlag,:].reshape( (gdcrop.NY, gdcrop.NX))
    plt.imshow(wspace, aspect='auto')
    plt.axis("off")
    if cc in cids:
        plt.title(cc)
    plt.subplot(sx,sy,cc*2+2)
    if cc in cids:
        plt.plot(wtmp[:,np.argmax(wspace)], '-ob')
        plt.plot(wtmp[:,np.argmin(wspace)], '-or')
        ws += np.abs(wspace)
    else:        
        plt.plot(wtmp[:,np.argmax(wspace)], '-o', color=(.5, .5, .5))
        plt.plot(wtmp[:,np.argmin(wspace)], '-o', color=(.5, .5, .5))

#%% Main eyetracking functionss

def find_best_epoch(ckpt_folder):
    from os import listdir
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    try:
        ckpt_files = listdir(ckpt_folder)  # list of strings
        epochs = [int(filename[6:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
        out = max(epochs)
    except FileNotFoundError:
        out = None
    return out


def get_model(gd, save_dir='./checkpoints', version=1, continue_training=False, use_divnorm=False):
    '''
    get a shared generalized quadratic model given a PixelDataset (see Datasets.py)
    
    continue_training will load the best model and continue training from there. Useful
    for refining a model after stim correction

    version (default: 1) - set to a new number if you want to train a completely new model

    '''
    from V1FreeViewingCode.models.basic import sGQM
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TestTubeLogger
    from pathlib import Path

    save_dir = Path(save_dir)

    # get train/validation set
    n_val = np.floor(len(gd)/5).astype(int)
    n_train = (len(gd)-n_val).astype(int)

    gd_train, gd_val = random_split(gd, lengths=[n_train, n_val])

    # build dataloaders
    bs = 1000
    train_dl = DataLoader(gd_train, batch_size=bs)
    valid_dl = DataLoader(gd_val, batch_size=bs)

    D_in = gd.NX*gd.NY*gd.num_lags

    if use_divnorm:
        gqm0 = sDN(input_dim=D_in, n_hidden=gd.NC*2, output_dim=gd.NC,
        learning_rate=.001,betas=[.9,.999],
        weight_decay=1e-0,
        optimizer='AdamW')
    else:
        gqm0 = sGQM(input_dim=D_in, n_hidden=gd.NC*2, output_dim=gd.NC,
            learning_rate=.001,betas=[.9,.999],
            weight_decay=1e-0,
            normalization=2,
            relu = True,
            filternorm = 0,
            optimizer='AdamW',
            ei_split=gd.NC)

    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')


    logger = TestTubeLogger(
        save_dir=save_dir,
        name=gd.id,
        version=version  # fixed to one to ensure checkpoint load
    )

    ckpt_folder = save_dir / sessid / 'version_{}'.format(version) / 'checkpoints'

    best_epoch = find_best_epoch(ckpt_folder)

    if best_epoch is None:

        trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            deterministic=False,
            progress_bar_refresh_rate=20,
            max_epochs=1000,
            auto_lr_find=True)

        seed_everything(42)

        trainer.tune(gqm0, train_dl, valid_dl) # find learning rate
        trainer.fit(gqm0, train_dl, valid_dl)

    else:
        if use_divnorm:
            gqm0 = sDN.load_from_checkpoint(str(ckpt_folder / 'epoch={}.ckpt'.format(best_epoch)))
        else:
            gqm0 = sGQM.load_from_checkpoint(str(ckpt_folder / 'epoch={}.ckpt'.format(best_epoch)))

        if continue_training:
            trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            deterministic=False,
            progress_bar_refresh_rate=20,
            max_epochs=1000,
            auto_lr_find=False,
            resume_from_checkpoint=str(ckpt_folder / 'epoch={}.ckpt'.format(best_epoch))
            )
            
            seed_everything(42)
            
            trainer.fit(gqm0, train_dl, valid_dl)

    return gqm0

def get_null_adjusted_ll(model, xt, yt, bits=False):
    '''
    get null-adjusted log likelihood
    bits=True will return in units of bits/spike
    '''
    m0 = model.cpu()
    loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
    lnull = -loss(torch.ones(yt.shape)*yt.mean(axis=0), yt).detach().cpu().numpy().sum(axis=0)
    llneuron = -loss(m0(xt),yt).detach().cpu().numpy().sum(axis=0)
    rbar = yt.sum(axis=0).numpy()
    ll = (llneuron - lnull)/rbar
    if bits:
        ll/=np.log(2)
    return ll

from pytorch_lightning import LightningModule
from V1FreeViewingCode.models.basic import Poisson
import torch.nn as nn

class convGQM(Poisson):
    '''
    This model class converts a shared GQM model into a convolutional model (with convolutional output)

    '''
    def __init__(self, input_dim=(15, 8, 6),
        gqm=None,
        ksize=None,
        **kwargs):
        
        super().__init__()
        self.save_hyperparameters()
        
        
        # ksize = (input_dim[1], input_dim[2])
        padding = (input_dim[1]-ksize[0], input_dim[2]-ksize[1])
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
            self.conv1.weight.data[i,:,:,:] = gqm.linear1.weight.data[i,:].reshape( (input_dims[0], ksize[0], ksize[1]))

        self.conv1.bias = gqm.linear1.bias


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

def get_likelihood_surface(gd, cmod, Npos=20, radius=1, valid_eye_range=5.2):
    '''
    main eye-tracking loop
    '''
    from tqdm import tqdm # progress bar
    from copy import deepcopy

    assert gd.corrected is False, "cannot get an LL surface on an already corrected stimulus"

    loss = nn.PoissonNLLLoss(log_input=False, reduction='none')

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
            if len(inds) > 500:
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

def plot_example_LLsurfs(LLspace, idxs=np.arange(2,5), icol=5, softmax=10):

    nplot = len(idxs)

    plt.figure(figsize=(15,6))
    for i,j in zip(idxs, range(nplot)):
        plt.subplot(1,nplot,j+1)
        I = LLspace[icol,i,:,:]
        I = 1-ne.normalize_range(I) # flip sign and normalize between 0 and 1
        Ly,Lx = I.shape

        x,y = ne.radialcenter(I**softmax)
        print("(%0.2f, %0.2f)" %(x-Lx//2,y-Ly//2))
        # print("(%0.2f, %0.2f)" %(x,y))
        plt.imshow(I, aspect='equal')
        plt.axvline(Lx/2, color='w')
        plt.axhline(Ly/2, color='w')
        plt.plot(x,y, '+r')
        plt.title('(%d,%d)' %(icol,idxs[j]))


#%% Get Dataset for training 

# augment the data with additive guassian noise
# This does a trick where the dataloader will pretend there is 5x more data and each
#  time __getitem__ is called, it corrupts the design matrix with additive gaussian noise
aug = [
    {'type': 'gaussian',
    'scale': 1,
    'proportion': 1}
    ]

num_lags = 10 # keep at 10 if you want to match existing models
gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=aug, cids=cids)

gdcrop = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=aug, cids=cids, cropidx=[xinds, yinds])    

#%% train stimulus model
# version 1: full stimulus 30,35
# version 2: cropped [5,25], [5,25]
gqm0 = get_model(gdcrop, version=7, continue_training=False, use_divnorm=True)
print("Done getting shared-GQM")

#%%
gqm0.plot_filters(gdcrop, sort=True)
#%% see that the model cross-validates better than the null

# test set
gd_test = PixelDataset(sessid, num_lags=num_lags,
    train=False, cids=cids, augment=None, cropidx=gdcrop.cropidx)

# # get test set
xt,yt=gd_test[:]

l2 = get_null_adjusted_ll(gqm0, xt, yt,bits=False)

plt.plot(l2, '-o')
plt.axhline(0)
plt.ylabel("test LL (null-adjusted)")
plt.xlabel("Unit Id")

#%% test divisive normalization layer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from V1FreeViewingCode.models.basic import weightConstraint, sGQM, Poisson
from pytorch_lightning import LightningModule

class divNorm(nn.Module):
    def __init__(self, in_features):
        super(divNorm, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        self.linear.weight.data[:] = 1
        self.linear.bias.data[:]=1
        self.beta = Parameter(2*torch.ones(in_features)) # raise to the power beta
        self.relu = F.relu

    # def _reset_parameters(self):
    #     self.beta = 
    def reset_parameters(self) -> None:
        # if self.beta is not None:
        print("initialize beta")
        nn.init.uniform_(self.beta, 1.0, 2.0)
        nn.init.unitorm_(self.linear.weight, 0.0, 1.0)
        nn.init.unitorm_(self.linear.bias, 0.5, 1.0)

    def forward(self, x):
        sz = list(x.shape) # get dimensions of input
        pdims = np.append( np.setdiff1d(np.arange(0,len(sz)), 1),1) # permute dims (move C to edge)
        
        snew = [sz[i] for i in pdims] # reshape size after permute

        x = x.permute(list(pdims)) #[N, C, Y, X] --> [N, Y, X, C]; or [N,C] -> [N,C]
        x = x.reshape((-1, sz[1])) # [N, Y, X, C] --> [N*X*Y, C]; or [N,C] -> [N,C]
        x = self.relu(x) # rectify
        # self.beta.apply(self.posConstraint) # positive 
        # self.beta.clamp_(1.0)
        x = torch.pow(x, self.beta)
        self.linear.weight.data.clamp_(0.0) 
        xdiv = self.linear(x) # [N*X*Y, C] --> [N*X*Y, C] # in and out are same dimension
        x /= xdiv # divisive normalization

        x = self.relu(x)

        x = x.reshape(snew) # [N*X*Y, K] --> [N, Y, X, K]
        x = x.permute(list(np.argsort(pdims))) # [N, Y, X, K] --> [N, K, Y, X]
        return x

class sDN(sGQM):
    def __init__(self, divnorm=True, **kwargs):
        super(sDN,self).__init__(**kwargs)
        # sGQM.__init__(self,**kwargs)
        print("adding divnorm to sGQM")
        self.divnorm = divNorm(self.hparams["n_hidden"])
        self.save_hyperparameters()
        # if divnorm:
        #     
            

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.divnorm(x)
        x = self.spikeNL(self.readout(x))

        return x

D_in = gd.NX*gd.NY*gd.num_lags
gqm0 = sDN(input_dim=D_in, n_hidden=gd.NC*2, output_dim=gd.NC,
        learning_rate=.001,betas=[.9,.999],
        weight_decay=1e-0,
        optimizer='AdamW')

# gqm0.divnorm
      
#%%

x1 = gqm0.flatten(xt)
x1 = gqm0.linear1(x1)


n = gqm0.hparams["n_hidden"]
dv = divNorm(n)

x = x1
#%%
plt.figure(figsize=(10,5))
f = plt.plot(dv(x1).detach().numpy())

#%% make model convolutional

input_dims = (gd.num_lags, gd.NY, gd.NX)
cmod = convGQM(input_dim=input_dims, gqm=gqm0, ksize=(gdcrop.NY, gdcrop.NX))


#%% reload data
# reload data with no augmentation and wider valid eye-position range
gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=None, cids=cids, valid_eye_rad=8)

gdcrop = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=None, cids=cids, valid_eye_rad=8, cropidx=gdcrop.cropidx)    

iframer = 1
#%% test on a small amount of data (to unpack how it works)
iframer += 1000
print(iframer)
idx = range(1+iframer,500+iframer)
x,y = gd[idx] # preload some data to get dimensions
NC = gd.NC
xh = cmod(x) # predict rates
sz = list(xh.size())

n = np.asarray([cmod.conv1.weight[i,:].abs().max().detach().numpy() for i in range(cmod.conv1.weight.shape[0])])
cinds = np.argsort(n)[::-1][-len(n):]
cc = cinds[0]

xn = x.numpy()
w = cmod.conv1.weight.detach().cpu().numpy()
loss = nn.PoissonNLLLoss(log_input=False, reduction='none')
wtmp = w[cc,:,:,:].squeeze()
wtmp = (wtmp - np.min(wtmp)) / (np.max(wtmp) - np.min(wtmp))
tpower = np.std(wtmp.reshape(w.shape[1], -1), axis=1)
bestlag = np.argmax(tpower)

plt.figure(figsize=(10,3))
for iframe in range(num_lags):
    plt.subplot(2,num_lags,iframe+1)
    plt.imshow(xn[0,iframe,:,:],cmap='gray')
    plt.axis("off")
    plt.subplot(2,num_lags,iframe+num_lags+1)
    plt.imshow(wtmp[iframe,:,:],cmap='gray',vmin=0, vmax=1)
    plt.axis("off")

plt.figure(figsize=(10,3))
plt.subplot(1,3,1) # plot stimulus
plt.imshow(xn[0,bestlag,:,:], cmap='gray')
plt.axis("off")

plt.subplot(1,3,2) # plot filter
plt.imshow(wtmp[bestlag,:,:], vmin=0, vmax=1, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3) # plot convolutional output
xc = cmod.nl(cmod.conv1(x)).detach().numpy()
plt.imshow(xc[0,cc,:].squeeze(), cmap='gray')
plt.axis("off")

xh = cmod(x)
# reshape and get loss across neurons over space
sz = list(xh.size())

L = 0
for cc in range(NC):
    yc = y[:,cc][:,None].repeat((1, sz[1]*sz[2])).reshape(sz[0:3])
    L += loss(xh[:,:,:,cc], yc).sum(axis=0)
L = L.detach().cpu().numpy()


plt.figure()
plt.imshow(-L)
plt.colorbar()
plt.xlabel("azimuth")
plt.ylabel('elevation')
plt.title("likelihood surface")


xcrop,ycrop = gdcrop[idx] # preload some data to get dimensions
l0 = loss(gqm0(xcrop),ycrop).sum().detach().cpu().numpy()
l1 = np.min(L)
print("%02.2f, %02.2f" %(l0,l1))


# %% debug convolutional layer

xc = (cmod.conv1(x)).detach().numpy()
x1 = (gqm0.linear1(gqm0.flatten(xcrop))).detach().cpu().numpy()[:,cc]
dims = xc.shape[2:]
x2 = xc[:,cc,:,:].reshape( (1000, -1))

# for i in range(x2.shape[1]):
#     plt.plot(x1, x2[:,i,j], '.')

sse = np.sum((np.expand_dims(x1, axis=1) - x2)**2, axis=0).reshape(dims)
cy,cx = np.where(sse==np.min(sse.flatten()))
plt.imshow(sse)
plt.plot(cx, cy, 'or')

plt.figure()
plt.plot(x1, xc[:,cc,cy,cx], '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
# 
#%% get likelihood surfaces as a function of gaze position

# get Likelihood surfaces
LLspace, locs = get_likelihood_surface(gd, cmod, Npos=20, radius=1.0, valid_eye_range=5)

#%% plot a few LL surfaces

idxs = np.arange(2, 10)

plot_example_LLsurfs(LLspace, idxs=idxs, icol=3, softmax=100)
#%% get correction grid
sz = LLspace.shape
Npos = sz[0]
NY = sz[2]
NX = sz[3]
softmax = 200
plot=True

# global center
I = np.mean(np.mean(LLspace, axis=0), axis=0)
cy, cx = np.where(I == np.min(I.flatten()))

plt.imshow(-I)
plt.plot(cx, cy, 'or')

from neureye import normalize_range,radialcenter

plt.figure(figsize=(40,40))
for xx in range(Npos):
        for yy in range(Npos):
            if plot:
                ax = plt.subplot(Npos,Npos,yy*Npos+xx+1)
            I = deepcopy(LLspace[xx,yy,:,:])
            I = 1-normalize_range(I) # flip sign and normalize between 0 and 1
            
            if plot:
                plt.imshow(I)
                ax.set_xticks([])
                ax.set_yticks([])
            
            min1,min2 = radialcenter(I**softmax) # softmax center
            if not np.isnan(min1):
                centers5[xx,yy,0] = min1-cx
                centers5[xx,yy,1] = min2-cy # in pixels
            if plot:
                plt.plot(min1, min2, '.r')
                # plt.axvline(cx, color='k')
                
                # plt.axhline(cy, color='k')

# centers5 = np.zeros([Npos,Npos,2])

#%%
from importlib import reload
reload(ne)
centers5,LLspace3 = ne.get_centers_from_LLspace(LLspace, softmax=200, plot=True, interpolation_steps=2, crop_edge=0)


#%% plot
from copy import deepcopy
centers6 = deepcopy(centers5)
# NY = centers6.shape[0]
# NX = centers6.shape[1]
# centers6[:,:,0] = centers6[:,:,0] - centers6[NY//2,NX//2,0]
# centers6[:,:,1] = centers6[:,:,1] - centers6[NY//2,NX//2,1]

plt.figure()
f = plt.plot(centers6[:,:,0], centers6[:,:,1], '.')



#%%

 # get correction
xcorrec,ycorrec = ne.get_shifter_from_centers(centers6, locs*gd.ppd, maxshift=0.5, nearest=False)
xshift = xcorrec(gd_test.eyeX, gd_test.eyeY)
yshift = ycorrec(gd_test.eyeX, gd_test.eyeY)
ix = np.logical_or(np.isnan(xshift), np.isnan(yshift))
print("Found %d/%d nan indices" %(np.sum(ix), len(ix)))
#%% save correction grid


import scipy.io as sio
from pathlib import Path
fpath = Path(gd.dirname)
fname = fpath / (sessid + "_CorrGrid.mat")

sio.savemat(str(fname.resolve()), {'centers': centers6, 'locs': locs, 'LLspace': LLspace})

#%% check that correction did anything
gqm1 = get_model(gd_test, version=2) # no cropping
gd_test = PixelDataset(sessid, num_lags=num_lags,
    train=True, cids=cids, corrected=False, valid_eye_rad=3.5, cropidx=gdcrop.cropidx)

# # get test set
xt0,yt0=gd_test[:]

l0 = get_null_adjusted_ll(gqm1, xt0, yt0, bits=False)

# #%%
from importlib import reload
# reload(PixelDataset)
import V1FreeViewingCode.Analysis.notebooks.Datasets as data
reload(data)
# from data import PixelDataset
from V1FreeViewingCode.Analysis.notebooks.Datasets import PixelDataset

gd_testC = PixelDataset(sessid, num_lags=num_lags,
    train=True, cids=cids, corrected=True, valid_eye_rad=3.5, cropidx=gdcrop.cropidx)


xt,yt=gd_testC[:]

# (xt-xt0).mean()

l1 = get_null_adjusted_ll(gqm1, xt, yt, bits=False)

plt.figure()
plt.plot(l0, l1, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel("LL before")
plt.ylabel("LL corrected")

#%% loop over these steps 

# step 1: load data (corrected), fit model
gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=aug, cids=cids, corrected=True)
gqm0 = get_model(gd, continue_training=True)

# step 2

#%% train stimulus model




# %%

    

# %%

w2 = gqm0.readout.weight.detach().cpu().numpy()

plt.imshow(w2)
# plt.plot(w2)
plt.figure()
plt.plot(np.sum(w2, axis=0))

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