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

sessid = '20200304b'

#%% get STAs
gd = PixelDataset(sessid, num_lags=10, augment=None, include_eyepos=True)

y = gd[:]['robs']
x = gd[:]['stim']
# z = gd['']
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

cids = np.arange(0, gd.NC)
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
x0 = 7
y0 = 7

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


#%% divisive normalization
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from V1FreeViewingCode.models.basic import sGQM, Poisson
from pytorch_lightning import LightningModule

class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=6, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class weightConstraint(object):
    def __init__(self,minval=0.0):
        self.minval = minval
        # pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            # print("Entered")
            module.weight.data.clamp_(self.minval)
        # if hasattr(module,'bias'):
        #     # print("Entered")
        #     module.bias.data.clamp_(self.minval)

# class powNL(nn.Module):
#     '''
#     raise to a power that is a learned parameter

#     '''
#     def __init__(self, in_features, defpow=2):
#         super(powNL, self).__init__()
#         self.weight = Parameter(defpow*torch.ones(in_features)) # raise to the power beta

#     def forward(self, x):
#         x.pow_(self.weight)
#         return x

class posLinear(nn.Module):
    '''
    Linear layer with positive weights
    weights can take any value, but will only impact the model they are greater than a minimum value

    '''
    def __init__(self, in_features, out_features, minval=0.0, hard_constraint=False):
        super(posLinear, self).__init__()
        self.hard_constraint=hard_constraint
        self.weight = Parameter(torch.ones(in_features, out_features)) # raise to the power beta
        self.bias = Parameter(torch.zeros(out_features))
        if self.hard_constraint:
            self.posConstraint = weightConstraint(minval)
        else:
            self.register_buffer("minval", torch.tensor(minval))

    def forward(self, x):
        if self.hard_constraint:
            self.apply(self.posConstraint)
            x = x@self.weight + self.bias
        else:
            x = x@(self.weight.max(self.minval)) + self.bias

        return x
# TODO: F.linear instead of @?? might be cleaner
class powNL(nn.Module):
    '''
    raise to a power that is a learned parameter

    '''
    def __init__(self, in_features, defpow=1.5, minpow=1.0, maxpow=2.0):
        super(powNL, self).__init__()
        # self.weight = Parameter(defpow*torch.ones(in_features)) # raise to the power beta
        self.relu = F.relu
        # self.register_buffer("minpow", torch.tensor(minpow))
        # self.register_buffer("maxpow", torch.tensor(minpow))

        self.register_buffer("weight", defpow*torch.ones(in_features))

    def forward(self, x):
        # x = self.relu(x).pow_(self.weight.max(self.minpow).min(self.maxpow))
        x = self.relu(x).pow_(self.weight)
        return x

class divNorm(nn.Module):
    def __init__(self, in_features):
        super(divNorm, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        # TODO: come up with way to zero the diagonal?
        # mask out diagonals?
        # mask = (1-eye) ... register buffer
        # always mask before forward

        self.relu = F.relu
        self.posConstraint = weightConstraint(0.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        print("initialize weights custom")
        nn.init.uniform(self.linear.weight, 0.0, 1.0)
        nn.init.uniform(self.linear.bias, 0.5, 1.0)

    def forward(self, x):
        sz = list(x.shape) # get dimensions of input
        pdims = np.append( np.setdiff1d(np.arange(0,len(sz)), 1),1) # permute dims (move C to edge)
        snew = [sz[i] for i in pdims] # reshape size after permute

        x = x.permute(list(pdims)) #[N, C, Y, X] --> [N, Y, X, C]; or [N,C] -> [N,C]
        x = x.reshape((-1, sz[1])) # [N, Y, X, C] --> [N*X*Y, C]; or [N,C] -> [N,C]
        
        x = self.relu(x) # rectify

        # apply constraints
        # self.linear.apply(self.posConstraint) # > 0.0

        xdiv = self.relu(self.linear(x)) # [N*X*Y, C] --> [N*X*Y, C] # in and out are same dimension
        x = x / xdiv.clamp_(0.001)

        x = x.reshape(snew) # [N*X*Y, K] --> [N, Y, X, K]
        x = x.permute(list(np.argsort(pdims))) # [N, Y, X, K] --> [N, K, Y, X]
        return x

class divNormPow(nn.Module):
    def __init__(self, in_features, hard_constraint=False):
        super(divNormPow, self).__init__()
        self.hardconstraint = hard_constraint

        self.linear = posLinear(in_features=in_features, out_features=in_features, hard_constraint=hard_constraint)
        self.pow = powNL(1, defpow=1.5, minpow=1.0, maxpow=2.0) # one nonlinearity for all 
        self.relu = F.relu
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        print("initialize weights custom")
        # nn.init.uniform(self.pow.weight, 1.0, 2.0)
        nn.init.uniform(self.linear.weight, 0.0, 1.0)
        nn.init.uniform(self.linear.bias, 0.5, 1.0)

    def forward(self, x):
        sz = list(x.shape) # get dimensions of input
        pdims = np.append( np.setdiff1d(np.arange(0,len(sz)), 1),1) # permute dims (move C to edge)
        snew = [sz[i] for i in pdims] # reshape size after permute

        x = x.permute(list(pdims)) #[N, C, Y, X] --> [N, Y, X, C]; or [N,C] -> [N,C]
        x = x.reshape((-1, sz[1])) # [N, Y, X, C] --> [N*X*Y, C]; or [N,C] -> [N,C]
        
        # x = self.relu(x) # rectify

        # apply constraints (no hard constraints)
        # self.pow.apply(self.oneConstraint) # > 1.0
        # self.linear.apply(self.posConstraint) # > 0.0

        x = self.pow(x)
        xdiv = self.relu(self.linear(x)) # [N*X*Y, C] --> [N*X*Y, C] # in and out are same dimension
        x = x / xdiv.clamp_(0.001)

        # x = self.relu(x)

        x = x.reshape(snew) # [N*X*Y, K] --> [N, Y, X, K]
        x = x.permute(list(np.argsort(pdims))) # [N, Y, X, K] --> [N, K, Y, X]
        return x

class sDN(sGQM):
    def __init__(self, divnorm=True, **kwargs):
        super(sDN,self).__init__(**kwargs)
        print("adding divnorm to sGQM")
        self.divnorm = divNormPow(self.hparams["n_hidden"])
        self.save_hyperparameters()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.divnorm(x)
        x = self.spikeNL(self.readout(x))

        return x

class sQDN(sGQM):
    def __init__(self, divnorm=True, **kwargs):
        super(sQDN,self).__init__(**kwargs)
        print("adding divnorm to sGQM")
        self.norm = divNorm(self.hparams["n_hidden"])
        self.save_hyperparameters()


    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
    # #     # update params
    #     optimizer.step()
    #     self.divnorm.beta.data.clamp_(1.0)
    #     self.divnorm.linear.weight.data.clamp_(0.0)
    #     self.divnorm.linear.bias.data.clamp_(0.0)
    #     self.zero_grad()
    #         # self.beta.data.clamp
    #         # self.l2.apply(self.posconstraint)
        

D_in = gd.NX*gd.NY*gd.num_lags
gqm0 = sQDN(input_dim=D_in, n_hidden=gd.NC*2, output_dim=gd.NC,
        learning_rate=.001,betas=[.9,.999],
        weight_decay=1e-0,
        optimizer='AdamW')



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


def get_model(gd, save_dir='./checkpoints', version=1, continue_training=False, use_divnorm=0):
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

    if use_divnorm==2:
        gqm0 = sQDN(input_dim=D_in, n_hidden=gd.NC*2, output_dim=gd.NC,
            learning_rate=.001,betas=[.9,.999],
            weight_decay=1e-0,
            normalization=2,
            relu = True,
            filternorm = 0,
            optimizer='AdamW',
            ei_split=gd.NC)

    elif use_divnorm==1:
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
            filternorm = 1,
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
            auto_lr_find=False)

        seed_everything(42)

        # trainer.tune(gqm0, train_dl, valid_dl) # find learning rate
        trainer.fit(gqm0, train_dl, valid_dl)

    else:
        if use_divnorm==2:
            gqm0 = sQDN.load_from_checkpoint(str(ckpt_folder / 'epoch={}.ckpt'.format(best_epoch)))
        elif use_divnorm==1:
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

class convSDN(Poisson):
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

        self.relu = gqm.relu
        
        self.spikeNL = gqm.spikeNL

        self.hparams.relu = gqm.hparams.relu
        self.nl = gqm.divnorm
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
        x = self.nl(x)

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
    'scale': .2,
    'proportion': .5}
    ]

num_lags = 10 # keep at 10 if you want to match existing models

# stimlist defaults to grating/gabor
gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=aug, cids=cids)

gdcrop = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=aug, cids=cids, cropidx=[xinds, yinds])    

#%% train using natural images
natimg = PixelDataset(sessid, stims='GaborNatImg', num_lags=num_lags,
cids=cids, augment=aug, full_dataset=False) #, cropidx=[xinds,yinds])

gqm0 = get_model(natimg, version=52, continue_training=False, use_divnorm=1)
print("Done getting shared-GQM")
# gqm0 = get_model(gd, version=1, continue_training=False, use_divnorm=0)
# print("Done getting shared-GQM")

gqm0.plot_filters(gd, sort=True)
#%% train stimulus model
# version 1: full stimulus 30,35
# version 2: cropped [5,25], [5,25]
# version 30: div norm = 1
gqm0 = get_model(gd, version=1, continue_training=False, use_divnorm=1)
print("Done getting shared-GQM")
# gqm0 = get_model(gd, version=1, continue_training=False, use_divnorm=0)
# print("Done getting shared-GQM")

gqm0.plot_filters(gd, sort=True)
#%%
# version 27: divnorm=1, cropped
gqm0 = get_model(gdcrop, version=10, continue_training=False, use_divnorm=1)
gqm0.plot_filters(gdcrop, sort=True)
#%% see that the model cross-validates better than the null

rsvp = PixelDataset(sessid, stims='FixRsvp', num_lags=num_lags, cids=cids, full_dataset=True)#, cropidx=[xinds,yinds])
rsvpc = PixelDataset(sessid, stims='FixRsvp', num_lags=num_lags, cids=cids, full_dataset=True, cropidx=[xinds,yinds])
iframer = 1

#%% test on a small amount of data (to unpack how it works)
iframer += 1000
print(iframer)
idx = range(1+iframer,500+iframer)
x,y = rsvpc[idx] # preload some data to get dimensions
NC = rsvp.NC
xh = gqm0(x) # predict rates
sz = list(xh.size())

n = np.asarray([gqm0.linear1.weight[i,:].abs().max().detach().numpy() for i in range(gqm0.linear1.weight.shape[0])])
cinds = np.argsort(n)[::-1][-len(n):]
cc = cinds[0]

xn = x.numpy()
w = gqm0.linear1.weight.detach().cpu().numpy()
loss = nn.PoissonNLLLoss(log_input=False, reduction='none')
wtmp = w[cc,:].reshape((rsvp.num_lags, rsvpc.NY, rsvpc.NX))
wtmp = (wtmp - np.min(wtmp)) / (np.max(wtmp) - np.min(wtmp))
tpower = np.std(wtmp.reshape(w.shape[1], -1), axis=1)
bestlag = np.argmax(tpower)

plt.figure(figsize=(10,3))
for iframe in range(num_lags):
    plt.subplot(2,num_lags,iframe+1)
    plt.imshow(xn[0,iframe,:,:],cmap='gray', interpolation='none')
    plt.axis("off")
    plt.subplot(2,num_lags,iframe+num_lags+1)
    plt.imshow(wtmp[iframe,:,:],cmap='gray',vmin=0, vmax=1, interpolation='none')
    plt.axis("off")

#%%
# predict from the whole dataset
x,y = rsvpc[:] # preload some data to get dimensions
NC = rsvp.NC
xh = gqm0(x) # predict rates

eyeX = rsvp.eyeX[rsvp.valid]
eyeY = rsvp.eyeY[rsvp.valid]
deltaEye = np.hypot(eyeX, eyeY)

frameTimes = rsvp.frameTime[rsvp.valid]
trStarts = np.append(0, np.where(np.diff(frameTimes, axis=0)>1)[0]+1)
trStops = np.append(np.where(np.diff(frameTimes, axis=0)>1)[0], len(frameTimes))
nTrials = len(trStarts)
fixstart = np.zeros(nTrials)
fixstop = np.zeros(nTrials)
for iTrial in range(nTrials):
    tix = np.arange(trStarts[iTrial], trStops[iTrial])
    fixinds = np.where(deltaEye[tix]<1)[0]
    if len(fixinds) > 10:
        fixstart[iTrial] = int(fixinds[0] + trStarts[iTrial])
        fixstop[iTrial] = int(fixinds[-1] + trStarts[iTrial])

good = fixstart!=0
fixstart = fixstart[good]
fixstop = fixstop[good]
fixdur = fixstop-fixstart
ntime = int(np.max(fixdur))

nfix = len(fixstart)

yhat = np.zeros( (nfix, ntime, NC))
ytrue = np.zeros( (nfix, ntime, NC))

for ifix in range(nfix-1):
    iix = np.arange(fixstart[ifix], fixstart[ifix]+ntime)
    yhat[ifix,:,:]=xh[iix,:].detach().numpy()
    ytrue[ifix,:,:]=y[iix,:].detach().numpy()

    ii2 = np.arange(int(fixdur[ifix]), ntime)
    ytrue[ifix,ii2,:] = np.nan
    yhat[ifix,ii2,:] = np.nan
    

ind = np.argsort(fixdur)

#%%
cc += 1
if cc >= NC:
    cc = 0
# cc = 6
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
f = plt.imshow(ytrue[ind,:,cc], aspect='auto')
plt.subplot(1,2,2)
f = plt.imshow(yhat[ind,:,cc], aspect='auto')

plt.figure(figsize=(10,2))
plt.plot(np.nanmean(ytrue[:,:,cc], axis=0), color='k')
plt.plot(np.nanmean(yhat[:,:,cc], axis=0), color='r')
plt.plot(yhat[ind[-1],:,cc])
plt.xlim([0,150])
plt.axis("tight")
plt.title(cc)
#%%

# test set
gd_test = PixelDataset(sessid, num_lags=num_lags,
    train=False, cids=cids, augment=None, cropidx=gdcrop.cropidx)

# gd_test = PixelDataset(sessid, num_lags=num_lags,
#     train=False, cids=cids, augment=None)

# # get test set
xt,yt=gd_test[:]
# xt,yt=rsvp[:]

l2 = get_null_adjusted_ll(gqm0, xt, yt,bits=False)

plt.plot(l2, '-o')
plt.axhline(0)
plt.ylabel("test LL (null-adjusted)")
plt.xlabel("Unit Id")

#%% test divisive normalization layer

plt.imshow(gqm0.divnorm.linear.weight.detach())
plt.figure()
plt.plot(gqm0.divnorm.linear.bias.detach())

plt.figure()
plt.plot(gqm0.divnorm.pow.weight.detach())
      
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
# cmod = convGQM(input_dim=input_dims, gqm=gqm0, ksize=(gdcrop.NY, gdcrop.NX))
cmod = convSDN(input_dim=input_dims, gqm=gqm0, ksize=(gdcrop.NY, gdcrop.NX))


#%% reload data
# reload data with no augmentation and wider valid eye-position range
gd = PixelDataset(sessid, num_lags=num_lags,
    train=False, augment=None, cids=cids, valid_eye_rad=8)

gdcrop = PixelDataset(sessid, num_lags=num_lags,
    train=False, augment=None, cids=cids, valid_eye_rad=8, cropidx=gdcrop.cropidx)    

iframer = 1
#%% test on a small amount of data (to unpack how it works)
iframer += 10
print(iframer)
idx = range(1+iframer,5000+iframer)
x,y = rsvp[idx] # preload some data to get dimensions
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

#%%
iframer =0 #+= 200
print(iframer)
idx = range(1+iframer,5000+iframer)
x,y = gd[idx] # preload some data to get dimensions

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


xcrop,ycrop = gd[idx] # preload some data to get dimensions
l0 = loss(gqm0(xcrop),ycrop).sum().detach().cpu().numpy()
l1 = np.min(L)
print("%02.2f, %02.2f" %(l0,l1))

cy,cx = L.shape
plt.plot(cx//2-3, cy//2, '+r')
ny,nx = np.where(L==l1)
plt.plot(nx,ny, '.r')
ny,nx = np.where(L==l0)
plt.plot(nx,ny, '.b')
#%%
ny,nx = np.where(L==l0)

#%%
iframer = 0 #+= 10
print(iframer)
idx = range(1+iframer,8000+iframer)
xcrop,ycrop = rsvpc[idx] # preload some data to get dimensions
f = plt.imshow(gqm0(xcrop).detach().T)
plt.figure()
f = plt.imshow(ycrop.T)

y0 = gqm0(xcrop).detach().T
y1 = ycrop.T

#%%
plt.figure()
cc += 1
if cc >= gd.NC:
    cc = 0
a=plt.xcorr(y1[cc,:], y0[cc,:], usevlines=False, maxlags=100)
# plt.figure()
# plt.plot(a,b)
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

idxs = np.arange(2, 10, 3)

plot_example_LLsurfs(LLspace, idxs=idxs, icol=4, softmax=100)
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
from copy import deepcopy
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
centers5,LLspace3 = ne.get_centers_from_LLspace(LLspace, softmax=100, plot=True, interpolation_steps=2, crop_edge=0)


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
# gqm1 = get_model(gd_test, version=16) # no cropping
gd_test = PixelDataset(sessid, num_lags=num_lags,
    train=True, cids=cids, corrected=False, valid_eye_rad=3.5, cropidx=gdcrop.cropidx)

# # get test set
xt0,yt0=gd_test[:]

l0 = get_null_adjusted_ll(gqm0, xt0, yt0, bits=False)

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

l1 = get_null_adjusted_ll(gqm0, xt, yt, bits=False)

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