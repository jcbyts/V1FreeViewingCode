#%%
import warnings

from pytorch_lightning.core.lightning import LightningModule; warnings.simplefilter('ignore')

import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import V1FreeViewingCode.Analysis.notebooks.Utils as U
import V1FreeViewingCode.Analysis.notebooks.gratings as gt


from scipy.ndimage import gaussian_filter
from copy import deepcopy

# pytorch
import numpy as np
import torch
from torch import nn

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# pytorch data management
from torch.utils.data import TensorDataset, Dataset, DataLoader, dataloader, random_split

import NDN3.NDNutils as NDNutils

import importlib # importlib lets you reload modules. Really useful for developing without having to restart the kernel

# Import modeling from V1FreeViewingCode repo
import V1FreeViewingCode.Analysis.notebooks.Utils as U # utilities useful for organizing data (do I even use this)

# path to where checkpointing will occur
iteration = 1
save_dir='./checkpoints/huklab_testing{}'.format(iteration)

from pathlib import Path

# %%  Helper functions
import scipy.io as sio

def resampleAtTimes(oldtimes, X, newtimes, sm=10):
    from scipy.ndimage import convolve1d
    from scipy.interpolate import interp1d

    kern = np.hanning(sm)   # a Hanning window with width 50
    kern /= kern.sum()      # normalize the kernel weights to sum to 1
    
    fs = 1//np.median(np.diff(oldtimes)) # sampling rate
    
    # get velocity by smoothing the derivative
    x = convolve1d(X, kern, axis=0)
    
    # resample at new times
    f = interp1d(oldtimes,x,kind='linear', axis=0, fill_value='extrapolate')
    return f(newtimes)

def load_data(fname, binSize=10):

    matdat = sio.loadmat(fname)

    startTime = np.min(matdat['spikeTimes'])
    stopTime = np.max(matdat['spikeTimes'])
    newtimes = np.arange(startTime, stopTime, binSize/1000)

    # bin spikes
    Robs = gt.bin_spike_times(matdat['spikeTimes'], matdat['spikeIds'], cids=np.unique(matdat['spikeIds']), bin_size=binSize, start_time=startTime, stop_time=stopTime ).T
    NT = Robs.shape[0]
    treadSpd = resampleAtTimes(matdat['treadTime'].flatten(), matdat['treadSpeed'].flatten(), newtimes)
    sacBoxcar = resampleAtTimes(matdat['eyeTime'].flatten(), (matdat['eyeLabels'].flatten()==2).astype('float32'), newtimes)

    onsets = np.digitize(matdat['GratingOnsets'], newtimes)
    offsets = np.digitize(matdat['GratingOffsets'], newtimes)

    directions = np.unique(matdat['GratingDirections'])
    ND = len(directions)
    gratingDir = np.zeros( (NT, ND))
    gratingCon = np.zeros( (NT, 1) )
    Ntrials = len(onsets)
    for iTrial in range(Ntrials):
        thcol = np.where(directions==matdat['GratingDirections'][iTrial][0])[0]
        inds = np.arange(onsets[iTrial], offsets[iTrial], 1)
        gratingDir[inds, thcol] = 1
        gratingCon[inds] = 1

    return gratingDir, gratingCon, treadSpd, sacBoxcar, Robs

class gratingdataset(Dataset):
    
    def __init__(self, gratingDir, gratingCon, treadSpd, sacBoxcar, Robs,
        num_lags=12, sac_lags=40, sacon_backshift=10, sacoff_backshift=0):
        
        self.num_lags = num_lags
        self.sac_lags = sac_lags
        self.sacon_backshift = sacon_backshift

        # map numpy arrays into tensors
        self.direction = torch.tensor(gratingDir.astype('float32'))
        self.contrast = torch.tensor(gratingCon.astype('float32'))
        self.tread = torch.tensor(treadSpd.astype('float32'))

            # get saccade onsets and offsets
        sacon = np.append(0, np.diff(sacBoxcar, axis=0)==1).astype('float32')
        sacoff = np.append(0, np.diff(sacBoxcar, axis=0)==-1).astype('float32')
        sacon = np.expand_dims(sacon, axis=1)
        sacoff = np.expand_dims(sacoff, axis=1)

        self.sacon = torch.tensor(sacon.astype('float32'))
        self.sacoff = torch.tensor(sacoff.astype('float32'))
        self.robs = torch.tensor(Robs.astype('float32'))
        
        startind = max(self.num_lags, self.sac_lags+self.sacon_backshift)+1
        self.valid = np.arange(startind, len(self.contrast))

    def __getitem__(self, index):

        # indices = index - np.arange(0, self.num_lags)
        # print(isinstance(index, int))
        # print(isinstance(index, slice))
        
        indices = self.valid[index]
        
        if isinstance(index, slice):
            stim_inds = np.expand_dims(indices, axis=1) - range(0,self.num_lags) 
            sac_inds = np.expand_dims(indices, axis=1) - range(0, self.sac_lags) + self.sacon_backshift   
        else:
            stim_inds = indices - range(0,self.num_lags)
            sac_inds = indices - range(0, self.sac_lags) + self.sacon_backshift

        # return {'contrast': exindices}
        
        

        return {'contrast': self.contrast[stim_inds,:], 'direction': self.direction[stim_inds,:], 'robs': self.robs[indices,:]}
        
    def __len__(self):
        return len(self.valid)

#%% load a session

fname = "/home/jake/Data/Datasets/HuklabTreadmill/processed/gru_20210525.mat"

gratingDir, gratingCon, treadSpd, sacBoxcar, Robs = load_data(fname, binSize=10)


gd = gratingdataset(gratingDir, gratingCon, treadSpd, sacBoxcar, Robs,
    num_lags=15)

#%% Test STA

sample = gd[:]
stas = torch.einsum('nlw,nc->lwc', sample['direction'], sample['robs']-sample['robs'].mean(dim=0))
sta2 = stas.detach().cpu().numpy()

NC = sample['robs'].shape[1]
sx = np.ceil(np.sqrt(NC))
sy = np.round(np.sqrt(NC))
plt.figure(figsize=(sx*2, sy*2))
for cc in range(NC):
    plt.subplot(sx, sy, cc+1)
    plt.imshow(sta2[:,:,cc])


#%% build a simple model
from torch.nn import functional as F

from pytorch_lightning import LightningModule

class Poisson(LightningModule):
    def __init__(self,
        learning_rate=1e-3,
        batch_size=1000,
        data_dir='',
        optimizer='AdamW',
        weight_decay=1e-2,
        amsgrad=False,
        betas=[.9,.999],
        max_iter=10000,
        **kwargs):

        super().__init__()
        self.save_hyperparameters()

        self.loss = nn.PoissonNLLLoss(log_input=False)

    def training_step(self, batch, batch_idx):
        
        y = batch['robs']
        y_hat = self(batch)

        loss = self.loss(y_hat, y)
        regularizers = self.regularizer()

        self.log('train_loss', loss + regularizers)
        return {'loss': loss + regularizers}

    def validation_step(self, batch, batch_idx):

        y = batch['robs']
        y_hat = self(batch)

        loss = self.loss(y_hat, y)
        
        self.log('val_loss', loss)
        return {'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        # # logging
        # if(self.current_epoch==1):
        #     self.logger.experiment.add_text('core', str(dict(self.core.hparams)))
        #     self.logger.experiment.add_text('readout', str(dict(self.readout.hparams)))

        avg_val_loss = torch.tensor([x['loss'] for x in validation_step_outputs]).mean()

        self.log('val_loss', avg_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return


    def configure_optimizers(self):
        
        if self.hparams.optimizer=='LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(),
                lr=self.hparams.learning_rate,
                max_iter=10000) #, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100)
        elif self.hparams.optimizer=='AdamW':
            optimizer = torch.optim.AdamW(self.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                weight_decay=self.hparams.weight_decay,
                amsgrad=self.hparams.amsgrad)
        elif self.hparams.optimizer=='Adam':
            optimizer = torch.optim.Adam(self.parameters(),
            lr=self.hparams.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        

class baseGLM(Poisson):
    def __init__(self, input_dim=(12, 1),
        output_dim=128,
        d2t=0.1,
        **kwargs):

        super().__init__()
        self.save_hyperparameters()
        
        self.l0 = nn.Flatten()
        self.contrast = nn.Linear(np.prod(self.hparams.input_dim), self.hparams.output_dim ,bias=True)
        self.spikeNL = nn.Softplus() 

    def regularizer(self):
        d2tcon = self.contrast.weight.diff(axis=1).pow(2).sum()
        return self.hparams.d2t * d2tcon
        # self.contrast.weight.data.diff()

    def forward(self, sample):
        x = self.l0(sample['contrast'])
        x = self.spikeNL(self.contrast(x))
        return x

from torch.nn.parameter import Parameter

class stimGLM(Poisson):
    def __init__(self, input_dim=(12, 1),
        num_directions=12,
        output_dim=128,
        d2t=0.1,
        **kwargs):

        super().__init__()
        self.save_hyperparameters()
        
        self.directionTuning = Parameter(torch.Tensor( size = (self.hparams.num_directions,self.hparams.output_dim) ))
        self.directionKernel = Parameter(torch.Tensor( size = (self.hparams.input_dim[0], self.hparams.output_dim) ))
        self.bias = Parameter(torch.Tensor( size = (1, self.hparams.output_dim) ))
        self.spikeNL = nn.Softplus() 
        self.directionTuning.data = torch.randn(self.directionTuning.shape)
        self.directionKernel.data = torch.randn(self.directionKernel.shape)
        self.bias.data = torch.rand(self.bias.shape)
        
    def regularizer(self):
        d2tdir = self.directionKernel.diff(axis=0).pow(2).sum()
        l2dir = self.directionTuning.pow(2).sum()
        return self.hparams.d2t * d2tdir + self.hparams.d2t * l2dir
        # self.contrast.weight.data.diff()

    def forward(self, sample):
        x = torch.einsum('nld,lc->ndc', sample['direction'], self.directionKernel)
        x = torch.einsum('ndc,dc->nc', x, self.directionTuning)
        x = self.spikeNL(x + self.bias)
        return x



#%%
import V1FreeViewingCode.models.utils as ut

bglm = baseGLM(input_dim=(gd.num_lags,1), output_dim=NC,
    weight_decay=1e-3,
    learning_rate=1e-1,
    d2t=1e-5,
    optimizer='AdamW',
    amsgrad=True)

sglm = stimGLM(input_dim=(gd.num_lags,1), output_dim=NC,
    num_directions=12,
    weight_decay=1e-3,
    learning_rate=1e-3,
    d2t=1e-4,
    optimizer='AdamW',
    amsgrad=True)



#%%
for version in [14]: # fit a few runs
    
    #% get trainer and train/valid data loaders (data loaders are iteratable datasets)
    trainer, train_dl, valid_dl = ut.get_trainer(gd, version=version,
            save_dir=save_dir,
            name='test',
            auto_lr=False,
            num_workers=64,
            earlystopping=True,
            batchsize=1000)


#     trainpath = Path(save_dir) / idtag / "version_{}".format(version)
    trainer.fit(sglm, train_dl, valid_dl)
print("Done")

#%%
f = plt.plot(bglm.contrast.weight.detach().cpu().numpy().T)

plt.figure()
f = plt.plot(sglm.directionKernel.detach().cpu().numpy())

plt.figure()
f = plt.plot(sglm.directionTuning.detach().cpu().numpy())



# %%
