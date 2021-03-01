# %% import libraries
import warnings
from collections import OrderedDict, Iterable
import numpy as np
import torch
from torch import nn
import copy

import os
from argparse import ArgumentParser
# from warnings import warn
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

#%%
"""
optinal validation loss: pearson correlation
"""
class Corr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):

        x = output
        y = target

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost




#%% shifter for eye calibration
class EyeShifter(LightningModule):
    """
    Perceptron 
    """
    def __init__(self, input_channels, hidden_channels):
        super().__init__()            
        self.input_channels = input_channels
        self.output_channels = 2
        self.hidden_channels = hidden_channels

        self.save_hyperparameters()

        # initialize network
        self.features = nn.Sequential()

        # --- first layer
        layer = OrderedDict()
        
        layer["linear"] = nn.Linear(
                self.input_channels,
                self.hidden_channels,
                bias=False)

        layer["nonlin"] = nn.ReLU()

        self.features.add_module("layer0", nn.Sequential(layer))

        layer = OrderedDict()
        
        layer["linear"] = nn.Linear(
                self.hidden_channels,
                self.output_channels,
                bias=True)

        self.features.add_module("layer1", nn.Sequential(layer))
    
    def forward(self,x):

        return self.features(x).clamp(-1.0, 1.0)

# %% Cores

class Core(LightningModule):
    def initialize(self):
        raise NotImplementedError("Not initializing")

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: "gamma" in x or "skip" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


class Core2d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

    @staticmethod
    def init_conv_hermite(m):
        if isinstance(m, HermiteConv2D):
            nn.init.normal_(m.coeffs.data, std=0.1)

"""
Readout Base class
"""
class Readout(LightningModule):
    def initialize(self, *args, **kwargs):
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(
                lambda x: not x.startswith("_") and ("gamma" in x or "pool" in x or "positive" in x), dir(self)
        ):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"




"""
Main encoder class
"""
class Encoder(LightningModule):
    def __init__(self,
        core=Core(),
        readout=Readout(),
        output_nl=nn.Softplus(),
        shifter=None,
        loss=nn.PoissonNLLLoss(log_input=False),
        val_loss=None,
        detach_core=False,
        train_shifter=False,
        learning_rate=1e-3,
        batch_size=1000,
        num_workers=0,
        data_dir='',
        optimizer='AdamW',
        weight_decay=1e-2,
        amsgrad=False,
        betas=[.9,.999],
        max_iter=10000,
        **kwargs):

        super().__init__()
        self.core = core
        self.readout = readout
        self.detach_core = detach_core
        self.shifter=shifter
        self.train_shifter = train_shifter
        self.save_hyperparameters('learning_rate','batch_size',
            'num_workers', 'data_dir', 'optimizer', 'weight_decay', 'amsgrad', 'betas',
            'max_iter')          
        
        if val_loss is None:
            self.val_loss = loss
        else:
            self.val_loss = val_loss

        self.output_nl = output_nl
        self.loss = loss

        # # manually tracked hparams
        # self.hparams.update(core=str(type(self.core)))
        # self.hparams.update(readout=str(type(self.readout)))
        # self.hparams.update(shifter=str(type(self.shifter)))

    def forward(self, x, shifter=None):
        x = self.core(x)
        if self.detach_core:
            x = x.detach()
        if self.shifter is not None and shifter is not None:
            x = self.readout(x, shift=self.shifter(shifter))
        else:
            x = self.readout(x)

        return self.output_nl(x)

    def training_step(self, batch, batch_idx):
        x = batch['stim']
        y = batch['robs']
        if self.shifter is not None and batch['eyepos'] is not None and self.train_shifter:
            y_hat = self(x, shifter=batch['eyepos'])
        else:
            y_hat = self(x)

        loss = self.loss(y_hat, y)
        regularizers = int(not self.detach_core) * self.core.regularizer() + self.readout.regularizer()

        self.log('train_loss', loss + regularizers)
        return {'loss': loss + regularizers}

    def validation_step(self, batch, batch_idx):

        x = batch['stim']
        y = batch['robs']
        if self.shifter is not None and batch['eyepos'] is not None and self.train_shifter:
            y_hat = self(x, shifter=batch['eyepos'])
        else:
            y_hat = self(x)
        loss = self.val_loss(y_hat, y)
        self.log('val_loss', loss)
        return {'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        # logging
        if(self.current_epoch==1):
            self.logger.experiment.add_text('core', str(dict(self.core.hparams)))
            self.logger.experiment.add_text('readout', str(dict(self.readout.hparams)))

        avg_val_loss = torch.tensor([x['loss'] for x in validation_step_outputs]).mean()
        tqdm_dict = {'val_loss': avg_val_loss}

        return {
                'progress_bar': tqdm_dict,
                'log': {'val_loss': avg_val_loss},
        }


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
    
    def on_save_checkpoint(self, checkpoint):
        # track the core, readout, shifter class and state_dicts
        checkpoint['core_type'] = type(self.core)
        checkpoint['core_hparams'] = self.core.hparams
        checkpoint['core_state_dict'] = self.core.state_dict()

        checkpoint['readout_type'] = type(self.readout)
        checkpoint['readout_hparams'] = self.readout.hparams
        checkpoint['readout_state_dict'] = self.readout.state_dict() # TODO: is this necessary or included in self state_dict?

        checkpoint['shifter_type'] = type(self.shifter)
        if checkpoint['shifter_type']!=type(None):
            checkpoint['shifter_hparams'] = self.shifter.hparams
            checkpoint['shifter_state_dict'] = self.shifter.state_dict() # TODO: is this necessary or included in model state_dict?

    def on_load_checkpoint(self, checkpoint):
        # properly handle core, readout, shifter state_dicts
        self.core = checkpoint['core_type'](**checkpoint['core_hparams'])
        self.readout = checkpoint['readout_type'](**checkpoint['readout_hparams'])
        if checkpoint['shifter_type']!=type(None):
            self.shifter = checkpoint['shifter_type'](**checkpoint['shifter_hparams'])
            self.shifter.load_state_dict(checkpoint['shifter_state_dict'])
        self.core.load_state_dict(checkpoint['core_state_dict'])
        self.readout.load_state_dict(checkpoint['readout_state_dict'])


# %% specific cores
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t # for conv2 default

class posConv2D(nn.Conv2d):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):

        super(posConv2D, self).__init__(in_channels,
            out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)
        self.register_buffer("minval", torch.tensor(0.0))

    def forward(self, x):
        posweight = torch.maximum(self.weight, self.minval)
        return self._conv_forward(x, posweight)


class PosLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PosLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.register_buffer("minval", torch.tensor(0.0))

    def forward(self, x):
        pos_weight = torch.maximum(self.weight, self.minval)
        return F.linear(x, pos_weight, self.bias)



def adaptive_elu(x, xshift, yshift):
    return F.elu(x - xshift, inplace=True) + yshift


class AdaptiveELU(nn.Module):
    """
    ELU shifted by user specified values. This helps to ensure the output to stay positive.
    """

    def __init__(self, xshift, yshift, **kwargs):
        super(AdaptiveELU, self).__init__(**kwargs)

        self.xshift = xshift
        self.yshift = yshift

    def forward(self, x):
        return adaptive_elu(x, self.xshift, self.yshift)

class EiMask(nn.Module):
    """
        Apply signed mask. Should work regardless of whether it's a linear or convolution model.
        Uses einstein summation.
    """
    def __init__(self, ni, ne):
        super(EiMask,self).__init__()

        self.register_buffer("ei_mask", torch.cat((torch.ones((1,ne)), -torch.ones((1,ni))), axis=1).squeeze())
    
    def forward(self, x):
        out = torch.einsum("nc...,c->nc...", x, self.ei_mask)
        return out


#%%
# class EiMask(nn.Module): old EImask code
#     """
#         Apply signed mask (only works on linear output)
#     """
#     def __init__(self, ni, ne):
#         super(EiMask,self).__init__()

#         self.register_buffer("ei_mask", torch.cat((torch.ones((1,ne)), -torch.ones((1,ni))), axis=1))
    
#     def forward(self, x):
#         return x*self.ei_mask

"""
Below, make a share NIM core
"""
class NimCore(Core):
    def __init__(
        self,
        input_size,
        hidden_channels,
        layers=1,
        ei_split=0,
        weight_norm=True,
        weight_norm_dim=0,
        gamma_hidden=0,
        gamma_input=0.0,
        elu_xshift=0.0,
        elu_yshift=0.0,
        group_norm=False,
        group_norm_num=4,
        skip=0,
        final_nonlinearity=True,
        bias=True,
        input_regularizer="LaplaceL2",
        stack=None,
        laplace_padding=0,
        use_avg_reg=True,
    ):
        """
        Args:
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            layers:         number of layers
            gamma_hidden:   regularizer factor for group sparsity
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see neuralpredictors.regularizers)
            skip:           Adds a skip connection
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            elu_xshift, elu_yshift: final_nonlinearity(x) = Elu(x - elu_xshift) + elu_yshift
            bias:           Adds a bias layer.
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            normalize_laplace_regularizer: Boolean, if set to True, will use the LaplaceL2norm function from
                neuralpredictors.regularizers, which returns the regularizer as |laplace(filters)| / |filters|
            input_regularizer:  String that must match one of the regularizers in ..regularizers
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            Implemented as layers_to_stack = layers[stack:]. thus:
                                stack = -1 will only select the last layer as the readout layer.
                                stack of -2 will read out from the last two layers.
                                And stack of 1 will read out from layer 1 (0 indexed) until the last layer.
            use_avg_reg:    bool. Whether to use the averaged value of regularizer(s) or the summed.
            To enable learning batch_norms bias and scale independently, the arguments bias, batch_norm and batch_norm_scale
            work together: By default, all are true. In this case there won't be a bias learned in the convolutional layer, but
            batch_norm will learn both its bias and scale. If batch_norm is false, but bias true, a bias will be learned in the
            convolutional layer. If batch_norm and bias are true, but batch_norm_scale is false, batch_norm won't have learnable
            parameters and a BiasLayer will be added after the batch_norm layer.
        """

        super().__init__()

        # regularizer_config = (
        #     dict(padding=laplace_padding, kernel=input_kern)
        #     if input_regularizer == "GaussianLaplaceL2"
        #     else dict(padding=laplace_padding)
        # )
        # self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)

        self.save_hyperparameters()
        self.input_channels = np.prod(input_size)

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        # self.input_size = input_size
        # self.input_channels = np.prod(input_size)
        # self.hidden_channels = hidden_channels
        self.skip = skip
        self.use_avg_reg = use_avg_reg
        # self.ei_split = ei_split

        

        if use_avg_reg:
            warnings.warn("The averaged value of regularizer will be used.", UserWarning)

        
        # dictates how to concatenate outputs
        if stack is None:
            self.stack = list(range(1,self.hparams.layers+1))
        else:
            self.stack = [*range(self.hparams.layers)[stack:]] if isinstance(stack, int) else stack


        # initialize network
        self.features = nn.Sequential()

        # this core is fully-connected, flatten inputs
        # self,flatten = nn.Flatten()
        self.features.add_module("lsayer00", nn.Sequential(nn.Flatten()))

        # --- first layer
        layer = OrderedDict()
        if weight_norm:
            layer["linear"] = nn.utils.weight_norm(
                nn.Linear(
                    self.input_channels, hidden_channels,
                    bias=bias),
                    dim=weight_norm_dim,
                    name='weight')
        else:
            layer["linear"] = nn.Linear(
                self.input_channels,
                hidden_channels,
                bias=bias)

        if layers > 1 or final_nonlinearity:
            layer["nonlin"] = AdaptiveELU(elu_xshift, elu_yshift)

        if ei_split > 0 and layers > 1:
            ni = ei_split
            ne = hidden_channels - ni
            layer["eimask"] = EiMask(ni, ne)

        self.features.add_module("layer0", nn.Sequential(layer))

        # --- other layers
        for l in range(1, self.hparams.layers):
            layer = OrderedDict()

            if ei_split>0:
                if weight_norm:
                    layer["linear"] = nn.utils.weight_norm(
                        PosLinear(
                            hidden_channels, hidden_channels,
                            bias=bias),
                            dim=weight_norm_dim,
                            name='weight')
                else:
                    layer["linear"] = PosLinear(
                        hidden_channels,
                        hidden_channels,
                        bias=bias)
            else:
                if weight_norm:
                    layer["linear"] = nn.utils.weight_norm(
                        nn.Linear(
                            self.input_channels, hidden_channels,
                            bias=bias),
                            dim=weight_norm_dim,
                            name='weight')
                else:
                    layer["linear"] = nn.Linear(
                        self.input_channels,
                        hidden_channels,
                        bias=bias)


            if group_norm:
                layer["norm"] = nn.GroupNorm(group_norm_num, hidden_channels)

            if final_nonlinearity or l < self.layers - 1:
                layer["nonlin"] = AdaptiveELU(elu_xshift, elu_yshift)

            if ei_split>0: # and l < self.layers - 1:
                layer["eimask"] = EiMask(ni, ne)

            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        # self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.hparams.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.hparams.skip, l) :], dim=1))
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight, avg=self.use_avg_reg)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.hparams.layers):
            ret = ret + self.features[l].linear.weight.pow(2).sqrt().mean()
            # ret = ret + self.features[l].linear.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.hparams.layers - 1) if self.hparams.layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.hparams.gamma_hidden # + self.hparams.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return (len(self.features)-1) * self.hparams.hidden_channels
        
#%% readout
# ------------------ Base Classes -------------------------



class FullReadout(Readout):
    """
    This readout connects to all 
    """

    def __init__(self, input_size, output_size,
        bias=True,
        constrain_positive=False,
        weight_norm=False,
        weight_norm_dim=0,
        gamma_readout=0,
        **kwargs):
        super().__init__()

        self.gamma_readout = gamma_readout

        if constrain_positive:
            if weight_norm:
                self.features = nn.utils.weight_norm(
                    PosLinear(
                        input_size, output_size,
                        bias=bias),
                        dim=weight_norm_dim,
                        name='weight')
            else:
                self.features = PosLinear(
                    input_size, output_size,
                    bias=bias)
        else:
            if weight_norm:
                self.features = nn.utils.weight_norm(
                    nn.Linear(
                        input_size, output_size,
                        bias=bias),
                        dim=weight_norm_dim,
                        name='weight')
            else:
                self.features = nn.Linear(
                    input_size, output_size,
                    bias=bias)

    def forward(self, x):        
        return self.features(x)

    def feature_l1(self, average=True):
        """
        Returns l1 regularization term for features.
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self.features.weight.abs().mean()
        else:
            return self.features.weight.abs().sum()

    # def initialize(self):


    def regularizer(self):
        return self.feature_l1() * self.gamma_readout


#%% load data
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

#%%
sessid = '20200304b'
aug = [
    {'type': 'gaussian',
    'scale': .1,
    'proportion': 1}
    ]

num_lags = 10 # keep at 10 if you want to match existing models

# stimlist defaults to grating/gabor
gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=aug, stims='GaborNatImg',
    include_eyepos=True)


# %% Initialize NIM core
input_size = (gd.num_lags, gd.NY, gd.NX)
hidden_channels = 20
core = NimCore(input_size, hidden_channels,
    gamma_hidden=.2, gamma_input=0, skip=0,
    elu_yshift=1.0, group_norm=True,
    ei_split=10,
    layers=2,bias=True)
    
readout = FullReadout(core.outchannels, gd.NC,
    constrain_positive=True,
    gamma_readout=.01)

model = Encoder(core, readout)

# %% train the model
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pathlib import Path

save_dir='./checkpoints'
version=4
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
    progress_bar_refresh_rate=20,
    max_epochs=1000,
    auto_lr_find=False)

seed_everything(42)


trainer.fit(model, train_dl, valid_dl)

    
# %%

def plot_filters(model, gd, sort=False):
    import matplotlib.pyplot as plt  # plotting
    ei_mask = model.core.features.layer0.eimask.ei_mask.detach().cpu().numpy()
    w = model.core.features.layer0.linear.weight.detach().cpu().numpy()
    # w = model.features.weight.detach().cpu().numpy()
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
        wtmp = np.reshape(w[cc,:], (gd.num_lags, gd.NX*gd.NY))
        bestlag = np.argmax(np.std(wtmp, axis=1))
        plt.imshow(np.reshape(wtmp[bestlag,:], (gd.NY, gd.NX)), interpolation=None, )
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

# %% plot input filters

plot_filters(model, gd)
# %% plot readout weights

plt.imshow(model.readout.features.weight.detach().numpy())
# %%

w = model.core.features.layer1.linear.weight.detach().cpu().numpy()
plt.imshow(w)
# %% 2D conv core and gaussian readout

import regularizers

"""
STACKED 2D Conv from Sinz lab code
"""
class Stacked2dCore(Core2d):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        layers=3,
        gamma_hidden=0,
        gamma_input=0.0,
        gamma_center=.1, # regularize the first conv layer to be centered
        skip=0,
        final_nonlinearity=True,
        elu_xshift=0.0,
        elu_yshift=0.0,
        bias=True,
        momentum=0.1,
        pad_input=True,
        hidden_padding=None,
        batch_norm=True,
        batch_norm_scale=True,
        independent_bn_bias=True,
        hidden_dilation=1,
        laplace_padding=0,
        input_regularizer="LaplaceL2norm",
        stack=None,
        use_avg_reg=True,
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            gamma_hidden:   regularizer factor for group sparsity
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see neuralpredictors.regularizers)
            skip:           Adds a skip connection
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            elu_xshift, elu_yshift: final_nonlinearity(x) = Elu(x - elu_xshift) + elu_yshift
            bias:           Adds a bias layer.
            momentum:       BN momentum
            pad_input:      Boolean, if True, applies zero padding to all convolutions
            hidden_padding: int or list of int. Padding for hidden layers. Note that this will apply to all the layers
                            except the first (input) layer.
            batch_norm:     Boolean, if True appends a BN layer after each convolutional layer
            batch_norm_scale: If True, a scaling factor after BN will be learned.
            independent_bn_bias:    If False, will allow for scaling the batch norm, so that batchnorm
                                    and bias can both be true. Defaults to True.
            hidden_dilation:    If set to > 1, will apply dilated convs for all hidden layers
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            normalize_laplace_regularizer: Boolean, if set to True, will use the LaplaceL2norm function from
                neuralpredictors.regularizers, which returns the regularizer as |laplace(filters)| / |filters|
            input_regularizer:  String that must match one of the regularizers in ..regularizers
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            Implemented as layers_to_stack = layers[stack:]. thus:
                                stack = -1 will only select the last layer as the readout layer.
                                stack of -2 will read out from the last two layers.
                                And stack of 1 will read out from layer 1 (0 indexed) until the last layer.

            use_avg_reg:    bool. Whether to use the averaged value of regularizer(s) or the summed.

            To enable learning batch_norms bias and scale independently, the arguments bias, batch_norm and batch_norm_scale
            work together: By default, all are true. In this case there won't be a bias learned in the convolutional layer, but
            batch_norm will learn both its bias and scale. If batch_norm is false, but bias true, a bias will be learned in the
            convolutional layer. If batch_norm and bias are true, but batch_norm_scale is false, batch_norm won't have learnable
            parameters and a BiasLayer will be added after the batch_norm layer.
        """

        super().__init__()

        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kern)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.gamma_center = gamma_center
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.skip = skip
        self.use_avg_reg = use_avg_reg

        if use_avg_reg:
            warnings.warn("The averaged value of regularizer will be used.", UserWarning)

        self.features = nn.Sequential()
        if stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = [*range(self.layers)[stack:]] if isinstance(stack, int) else stack

        # --- first layer
        layer = OrderedDict()
        layer["conv"] = nn.Conv2d(
            input_channels,
            hidden_channels,
            input_kern,
            padding=input_kern // 2 if pad_input else 0,
            bias=bias and not batch_norm,
        )
        if batch_norm:
            if independent_bn_bias:
                layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
            else:
                layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum, affine=bias and batch_norm_scale)
                if bias:
                    if not batch_norm_scale:
                        layer["bias"] = Bias2DLayer(hidden_channels)
                elif batch_norm_scale:
                    layer["scale"] = Scale2DLayer(hidden_channels)

        if layers > 1 or final_nonlinearity:
            layer["nonlin"] = AdaptiveELU(elu_xshift, elu_yshift)
        self.features.add_module("layer0", nn.Sequential(layer))

        # --- other layers
        if not isinstance(hidden_kern, Iterable):
            hidden_kern = [hidden_kern] * (self.layers - 1)

        for l in range(1, self.layers):
            layer = OrderedDict()

            hidden_padding = ((hidden_kern[l - 1] - 1) * hidden_dilation + 1) // 2
            layer["conv"] = nn.Conv2d(
                hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                hidden_channels,
                hidden_kern[l - 1],
                padding=hidden_padding,
                bias=bias and not batch_norm,
                dilation=hidden_dilation,
            )
            if batch_norm:
                if independent_bn_bias:
                    layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
                else:
                    layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum, affine=bias and batch_norm_scale)
                    if bias:
                        if not batch_norm_scale:
                            layer["bias"] = Bias2DLayer(hidden_channels)
                    elif batch_norm_scale:
                        layer["scale"] = Scale2DLayer(hidden_channels)

            if final_nonlinearity or l < self.layers - 1:
                layer["nonlin"] = AdaptiveELU(elu_xshift, elu_yshift)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        # center regularization
        regw = 1 - regularizers.gaussian2d(input_kern,sigma=input_kern//2)
        self.register_buffer("center_reg_weights", torch.tensor(regw))

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l) :], dim=1))
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight, avg=self.use_avg_reg)

    def center_reg(self):
        ret = torch.einsum('ijxy,xy->ij', self.features[0].conv.weight, self.center_reg_weights).pow(2).mean()
        return ret

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace() + self.gamma_center * self.center_reg()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels


"""
Stacked 2D EI Core
"""
class Stacked2dEICore(Core2d):
    def __init__(
        self,
        input_channels=10,
        hidden_channels=10,
        input_kern=9,
        hidden_kern=9,
        layers=3,
        gamma_hidden=0,
        gamma_input=0.0,
        gamma_center=.1, # regularize the first conv layer to be centered
        prop_inh=.5,
        skip=0,
        activation="elu",
        final_nonlinearity=True,
        bias=True,
        momentum=0.1,
        pad_input=True,
        hidden_padding=None,
        group_norm=True,
        num_groups=2,
        weight_norm=True,
        hidden_dilation=1,
        laplace_padding=0,
        input_regularizer="LaplaceL2norm",
        stack=None,
        use_avg_reg=True,
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            gamma_hidden:   regularizer factor for group sparsity
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see neuralpredictors.regularizers)
            gamma_center:   regularizer to center the first layer kernels  in x,y
            prop_inh:       proportion of inhibition in the hidden layers
            skip:           Adds a skip connection
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            elu_xshift, elu_yshift: final_nonlinearity(x) = Elu(x - elu_xshift) + elu_yshift
            bias:           Adds a bias layer.
            momentum:       BN momentum
            pad_input:      Boolean, if True, applies zero padding to all convolutions
            hidden_padding: int or list of int. Padding for hidden layers. Note that this will apply to all the layers
                            except the first (input) layer.
            batch_norm:     Boolean, if True appends a BN layer after each convolutional layer
            batch_norm_scale: If True, a scaling factor after BN will be learned.
            independent_bn_bias:    If False, will allow for scaling the batch norm, so that batchnorm
                                    and bias can both be true. Defaults to True.
            hidden_dilation:    If set to > 1, will apply dilated convs for all hidden layers
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            normalize_laplace_regularizer: Boolean, if set to True, will use the LaplaceL2norm function from
                neuralpredictors.regularizers, which returns the regularizer as |laplace(filters)| / |filters|
            input_regularizer:  String that must match one of the regularizers in ..regularizers
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            Implemented as layers_to_stack = layers[stack:]. thus:
                                stack = -1 will only select the last layer as the readout layer.
                                stack of -2 will read out from the last two layers.
                                And stack of 1 will read out from layer 1 (0 indexed) until the last layer.

            use_avg_reg:    bool. Whether to use the averaged value of regularizer(s) or the summed.

            To enable learning batch_norms bias and scale independently, the arguments bias, batch_norm and batch_norm_scale
            work together: By default, all are true. In this case there won't be a bias learned in the convolutional layer, but
            batch_norm will learn both its bias and scale. If batch_norm is false, but bias true, a bias will be learned in the
            convolutional layer. If batch_norm and bias are true, but batch_norm_scale is false, batch_norm won't have learnable
            parameters and a BiasLayer will be added after the batch_norm layer.
        """

        super().__init__()
        
        self.save_hyperparameters()

        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kern)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.gamma_center = gamma_center
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.skip = skip
        self.use_avg_reg = use_avg_reg
        if activation == "elu":
            self.activation = AdaptiveELU(0.0, 1.0)

        if use_avg_reg:
            warnings.warn("The averaged value of regularizer will be used.", UserWarning)

        self.features = nn.Sequential()
        if stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = [*range(self.layers)[stack:]] if isinstance(stack, int) else stack

        ni = int(np.ceil(prop_inh*hidden_channels))
        ne = hidden_channels - ni
        self.num_inh = ni
        self.num_exc = ne

        # --- first layer
        layer = OrderedDict()
        if weight_norm:
            layer["conv"] = nn.utils.weight_norm(nn.Conv2d(
                input_channels,
                hidden_channels,
                input_kern,
                padding=input_kern // 2 if pad_input else 0,
                bias=bias and not group_norm),
                dim=0, name='weight')
        else:
            layer["conv"] = nn.Conv2d(
                input_channels,
                hidden_channels,
                input_kern,
                padding=input_kern // 2 if pad_input else 0,
                bias=bias and not group_norm,
            )

        if layers > 1 or final_nonlinearity:
            layer["nonlin"] = self.activation #AdaptiveELU(elu_xshift, elu_yshift)

        if group_norm:
            layer["norm"] = nn.GroupNorm(num_groups, hidden_channels)

        layer["eimask"] = EiMask(self.num_inh, self.num_exc)

        self.features.add_module("layer0", nn.Sequential(layer))

        # --- other layers
        if not isinstance(hidden_kern, Iterable):
            hidden_kern = [hidden_kern] * (self.layers - 1)

        for l in range(1, self.layers):
            layer = OrderedDict()

            hidden_padding = ((hidden_kern[l - 1] - 1) * hidden_dilation + 1) // 2
            if weight_norm:
                layer["conv"] = nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                    hidden_channels,
                    hidden_kern[l - 1],
                    padding=hidden_padding,
                    bias=bias,
                    dilation=hidden_dilation),
                    dim=0)

            else:
                layer["conv"] = nn.Conv2d(
                    hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                    hidden_channels,
                    hidden_kern[l - 1],
                    padding=hidden_padding,
                    bias=bias,
                    dilation=hidden_dilation,
                )

            if final_nonlinearity or l < self.layers - 1:
                layer["nonlin"] = self.activation #AdaptiveELU(elu_xshift, elu_yshift)

            if group_norm:
                layer["norm"] = nn.GroupNorm(num_groups, hidden_channels)

            layer["eimask"] = EiMask(self.num_inh, self.num_exc)

            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        # center regularization
        regw = 1 - regularizers.gaussian2d(input_kern,sigma=input_kern//4)
        self.register_buffer("center_reg_weights", torch.tensor(regw))

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l) :], dim=1))
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight, avg=self.use_avg_reg)

    def center_reg(self):
        ret = torch.einsum('ijxy,xy->ij', self.features[0].conv.weight, self.center_reg_weights).pow(2).mean()
        return ret

    def group_sparsity(self):
        ret = self._input_weights_regularizer(self.features[1].conv.weight, avg=self.use_avg_reg)

        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)
        # return ret

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace() + self.gamma_center * self.center_reg()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels

"""
GAUSSIAN READOUT CORE FROM SINZ LAB CODE
"""

class FullGaussian2d(Readout):
    """
    A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned.

    Args:
        in_shape (list, tuple): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]. Default: 0.1
        init_sigma (float): The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is
            'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the
            covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        gauss_type (str): Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        grid_mean_predictor (dict): Parameters for a predictor of the mean grid locations. Has to have a form like
                        {
                        'hidden_layers':0,
                        'hidden_features':20,
                        'final_tanh': False,
                        }
        shared_features (dict): Used when the feature vectors are shared (within readout between neurons) or between
                this readout and other readouts. Has to be a dictionary of the form
               {
                    'match_ids': (numpy.array),
                    'shared_features': torch.nn.Parameter or None
                }
                The match_ids are used to match things that should be shared within or across scans.
                If `shared_features` is None, this readout will create its own features. If it is set to
                a feature Parameter of another readout, it will replace the features of this readout. It will be
                access in increasing order of the sorted unique match_ids. For instance, if match_ids=[2,0,0,1],
                there should be 3 features in order [0,1,2]. When this readout creates features, it will do so in
                that order.
        shared_grid (dict): Like `shared_features`. Use dictionary like
               {
                    'match_ids': (numpy.array),
                    'shared_grid': torch.nn.Parameter or None
                }
                See documentation of `shared_features` for specification.

        source_grid (numpy.array):
                Source grid for the grid_mean_predictor.
                Needs to be of size neurons x grid_mean_predictor[input_dimensions]

    """

    def __init__(self, in_shape=[10,10,10],
                outdims=10,
                bias=True,
                init_mu_range=0.1,
                init_sigma=1,
                batch_sample=True,
                align_corners=True,
                gauss_type='uncorrelated',
                grid_mean_predictor=None,
                constrain_positive=False,
                shared_features=None,
                shared_grid=None,
                source_grid=None,
                **kwargs):

        super().__init__()

        # determines whether the Gaussian is isotropic or not
        self.save_hyperparameters()

        self.gauss_type = gauss_type

        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")

        # store statistics about the images and neurons
        self.in_shape = in_shape
        self.outdims = outdims

        # sample a different location per example
        self.batch_sample = batch_sample

        self.constrain_positive = constrain_positive

        # position grid shape
        self.grid_shape = (1, outdims, 1, 2)

        # the grid can be predicted from another grid
        self._predicted_grid = False
        self._shared_grid = False
        self._original_grid = not self._predicted_grid

        if grid_mean_predictor is None and shared_grid is None:
            self._mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron
        elif grid_mean_predictor is not None and shared_grid is not None:
            raise ConfigurationError('Shared grid_mean_predictor and shared_grid_mean cannot both be set')
        elif grid_mean_predictor is not None:
            self.init_grid_predictor(source_grid=source_grid, **grid_mean_predictor)
        elif shared_grid is not None:
            self.initialize_shared_grid(**(shared_grid or {}))

        if gauss_type == 'full':
            self.sigma_shape = (1, outdims, 2, 2)
        elif gauss_type == 'uncorrelated':
            self.sigma_shape = (1, outdims, 1, 2)
        elif gauss_type == 'isotropic':
            self.sigma_shape = (1, outdims, 1, 1)
        else:
            raise ValueError(f'gauss_type "{gauss_type}" not known')

        self.init_sigma = init_sigma
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))  # standard deviation for gaussian for each neuron

        self.initialize_features(**(shared_features or {}))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.register_buffer("regvalplaceholder", torch.zeros(1))
        self.init_mu_range = init_mu_range
        self.align_corners = align_corners
        self.initialize()

    @property
    def shared_features(self):
        return self._features

    @property
    def shared_grid(self):
        return self._mu

    @property
    def features(self):
        if self.constrain_positive:
            # feat = self._features.data.clamp(0.0)
            feat = F.relu(self._features)
        else:
            feat = self._features

        if self._shared_features:
            feat = self.scales * feat[..., self.feature_sharing_index]
        
        return feat

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    def feature_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if self._original_features:
            if average:
                return self._features.abs().mean()
            else:
                return self._features.abs().sum()
        else:
            return 0

    @property
    def mu(self):
        if self._predicted_grid:
            return self.mu_transform(self.source_grid.squeeze()).view(*self.grid_shape)
        elif self._shared_grid:
            if self._original_grid:
                return self._mu[:, self.grid_sharing_index, ...]
            else:
                return self.mu_transform(self._mu.squeeze())[self.grid_sharing_index].view(*self.grid_shape)
        else:
            return self._mu

    def sample_grid(self, batch_size, sample=None):
        """
        Returns the grid locations from the core by sampling from a Gaussian distribution
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """
        with torch.no_grad():
            self.mu.clamp_(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
            if self.gauss_type != 'full':
                self.sigma.clamp_(min=0)  # sigma/variance i    s always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]

        sample = self.training if sample is None else sample
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

        if self.gauss_type != 'full':
            return torch.clamp(
                norm * self.sigma + self.mu, min=-1, max=1
            )  # grid locations in feature space sampled randomly around the mean self.mu
        else:
            return torch.clamp(
                torch.einsum('ancd,bnid->bnic', self.sigma, norm) + self.mu, min=-1, max=1
            )  # grid locations in feature space sampled randomly around the mean self.mu

    def init_grid_predictor(self, source_grid, hidden_features=20, hidden_layers=0, final_tanh=False):
        self._original_grid = False
        layers = [
            nn.Linear(source_grid.shape[1], hidden_features if hidden_layers > 0 else 2)
        ]

        for i in range(hidden_layers):
            layers.extend([
                nn.ELU(),
                nn.Linear(hidden_features, hidden_features if i < hidden_layers - 1 else 2)
            ])

        if final_tanh:
            layers.append(
                nn.Tanh()
            )
        self.mu_transform = nn.Sequential(*layers)

        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer('source_grid', torch.from_numpy(source_grid.astype(np.float32)))
        self._predicted_grid = True

    def initialize(self):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """

        if not self._predicted_grid or self._original_grid:
            self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.gauss_type != 'full':
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)
        self._features.data.fill_(1 / self.in_shape[0])
        if self._shared_features:
            self.scales.data.fill_(1.)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, w, h = self.in_shape
        self._original_features = True
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (1, c, 1, n_match_ids), \
                    f'shared features need to have shape (1, {c}, 1, {n_match_ids})'
                self._features = shared_features
                self._original_features = False
            else:
                self._features = Parameter(
                    torch.Tensor(1, c, 1, n_match_ids))  # feature weights for each channel of the core
            self.scales = Parameter(torch.Tensor(1, 1, 1, self.outdims))  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer('feature_sharing_index', torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = Parameter(
                torch.Tensor(1, c, 1, self.outdims))  # feature weights for each channel of the core
            self._shared_features = False

    def initialize_shared_grid(self, match_ids=None, shared_grid=None):
        c, w, h = self.in_shape

        if match_ids is None:
            raise ConfigurationError('match_ids must be set for sharing grid')
        assert self.outdims == len(match_ids), 'There must be one match ID per output dimension'

        n_match_ids = len(np.unique(match_ids))
        if shared_grid is not None:
            assert shared_grid.shape == (1, n_match_ids, 1, 2), \
                f'shared grid needs to have shape (1, {n_match_ids}, 1, 2)'
            self._mu = shared_grid
            self._original_grid = False
            self.mu_transform = nn.Linear(2, 2)
            self.mu_transform.bias.data.fill_(0.)
            self.mu_transform.weight.data = torch.eye(2)
        else:
            self._mu = Parameter(torch.Tensor(1, n_match_ids, 1, 2))  # feature weights for each channel of the core
        _, sharing_idx = np.unique(match_ids, return_inverse=True)
        self.register_buffer('grid_sharing_index', torch.from_numpy(sharing_idx))
        self._shared_grid = True

    def forward(self, x, sample=None, shift=None, out_idx=None):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted

        Returns:
            y: neuronal activity
        """
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")
        feat = self.features
        feat = feat.reshape(1, c, self.outdims)
        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, 2)

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = grid[:, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y

    def regularizer(self):
        return self.regvalplaceholder

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.gauss_type + ' '
        r += self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        if self._shared_features:
            r += ", with {} features".format('original' if self._original_features else 'shared')

        if self._predicted_grid:
            r += ", with predicted grid"
        if self._shared_grid:
            r += ", with {} grid".format('original' if self._original_grid else 'shared')

        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r


# %% reload training set w/ natural images
num_lags = 14
sessid = '20200304b'
# load Gabors, Gratings, Natural Images
gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=None, stims='GaborNatImg', include_eyepos=True)

#%%

input_size = (gd.num_lags, gd.NY, gd.NX)
input_channels = gd.num_lags
hidden_channels = 10
input_kern = 15
hidden_kern = 21
core = Stacked2dEICore(input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        layers=2,
        gamma_hidden=1e-6,
        gamma_input=1e-5,
        gamma_center=.1,
        prop_inh=.5,
        skip=0,
        activation="elu",
        final_nonlinearity=True,
        bias=False,
        pad_input=True,
        hidden_padding=hidden_kern//2,
        group_norm=True,
        num_groups=5,
        weight_norm=False,
        hidden_dilation=1,
        laplace_padding=2,
        input_regularizer="LaplaceL2norm",
        stack=None,
        use_avg_reg=True)

# initialize input layer to be centered
regw = regularizers.gaussian2d(input_kern,sigma=input_kern//4)
core.features[0].conv.weight.data = torch.einsum('ijkm,km->ijkm', core.features[0].conv.weight.data, torch.tensor(regw))

# initialize hidden layer to be centered and positive
regw = regularizers.gaussian2d(hidden_kern,sigma=hidden_kern//4)
core.features[1].conv.weight.data = torch.einsum('ijkm,km->ijkm', core.features[1].conv.weight.data.abs(), torch.tensor(regw))
    
in_shape = [core.outchannels, gd.NY, gd.NX]
bias = True
readout = FullGaussian2d(in_shape, gd.NC, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
                 align_corners=False, gauss_type='uncorrelated', grid_mean_predictor=None,
                 constrain_positive=True,
                 shared_features=None, shared_grid=None, source_grid=None)

readout.bias.data = gd.y.mean(axis=0)

# initialize shifter
shifter = EyeShifter(2, 10)
shifter.features.layer1.linear.weight.data *= torch.tensor(0.01)

# combine core and readout into model
model = Encoder(core, readout, shifter=shifter, train_shifter=False,
    weight_decay=.01, optimizer='AdamW', learning_rate=.01,
    betas=[.9, .999], amsgrad=True)#, val_loss=Corr())

#%% reload data
sessid = '20200304b'
aug = [
    {'type': 'gaussian',
    'scale': .2,
    'proportion': 1}
    ]

# load Gabors, Gratings, Natural Images
gd = PixelDataset(sessid, num_lags=num_lags,
    train=True, augment=None, stims='GaborNatImg', include_eyepos=True)
# %%
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pathlib import Path

save_dir='./checkpoints'
version=5
save_dir = Path(save_dir)



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

#%%
ckpt_folder = save_dir / sessid / 'version_{}'.format(version) / 'checkpoints'

best_epoch = find_best_epoch(ckpt_folder)
print(best_epoch)

chkpath = str(ckpt_folder / 'epoch={}.ckpt'.format(best_epoch))
model2 = Encoder.load_from_checkpoint(chkpath, strict=True)
# chkpath = str(ckpt_folder / 'epoch={}.pt'.format(best_epoch))
# model = torch.load(chkpath)

#%%

checkpoint = torch.load(chkpath)
w = checkpoint['core_state_dict']['features.layer0.conv.weight'].detach().cpu().numpy()
plt.subplot(1,2,1)
plt.imshow(w[1,5,:,:])
model2.core.load_state_dict(checkpoint['core_state_dict'])
w = model.core.features.layer0.conv.weight.detach().cpu().numpy()
plt.subplot(1,2,2)
plt.imshow(w[1,5,:,:])


#%% Train
version = 7

# get train/validation set
n_val = np.floor(len(gd)/5).astype(int)
n_train = (len(gd)-n_val).astype(int)

gd_train, gd_val = random_split(gd, lengths=[n_train, n_val])

# build dataloaders
bs = 1000
train_dl = DataLoader(gd_train, batch_size=bs)
valid_dl = DataLoader(gd_val, batch_size=bs)

D_in = gd.NX*gd.NY*gd.num_lags
  
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
    auto_lr_find=True)

# seed_everything(42)

# trainer.tune(model, train_dl, valid_dl)
trainer.fit(model, train_dl, valid_dl)

#%% save model
torch.save(model, checkpoint_callback.best_model_path.replace('.ckpt', '.pt'))

#%%

ckpath = checkpoint_callback.best_model_path
checkpoint = torch.load(ckpath)

# core = checkpoint['core_type'](**checkpoint['core_hparams'])
core = Stacked2dEICore(**checkpoint['core_hparams'])
core.load_state_dict(checkpoint['core_state_dict'])



# %%
# core.eval()
# nn.utils.remove_weight_norm(core.features.layer0.conv)

w = model.core.features.layer1.conv.weight.detach().cpu().numpy()
# w = core.features.layer0.conv.weight.detach().cpu().numpy()
# w = checkpoint['core_state_dict']['features.layer0.conv.weight_v'].detach().cpu().numpy()

sx = w.shape[0]
sy = w.shape[1]
plt.figure(figsize=(10,10))
for i in range(sx):
    wtmp = copy.deepcopy(np.squeeze(w[i,:,:,:]))
    wtmp = (wtmp - np.min(wtmp)) / (np.max(wtmp) - np.min(wtmp))
    for j in range(sy):
        plt.subplot(sx,sy, i*sy + j + 1)
        plt.imshow(wtmp[j,:,:],interpolation=None, vmin=0, vmax = 1)
        plt.axis("off")

# %%
w = model.readout.features.detach().cpu().numpy().squeeze()
f =plt.plot(w)

# %%

def get_null_adjusted_ll(model, sample, yt, bits=False):
    '''
    get null-adjusted log likelihood
    bits=True will return in units of bits/spike
    '''
    m0 = model.cpu()
    loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
    lnull = -loss(torch.ones(yt.shape)*yt.mean(axis=0), yt).detach().cpu().numpy().sum(axis=0)
    yhat = m0(sample['stim'], shifter=sample['eyepos'])
    llneuron = -loss(yhat,yt).detach().cpu().numpy().sum(axis=0)
    rbar = yt.sum(axis=0).numpy()
    ll = (llneuron - lnull)/rbar
    if bits:
        ll/=np.log(2)
    return ll

 
# test set
gd_test = PixelDataset(sessid, num_lags=num_lags,
    train=False, augment=None, stims='GaborNatImg', include_eyepos=False)

# # get test set
sample=gd_test[:]

l2 = get_null_adjusted_ll(model, sample, sample['robs'],bits=False)

plt.plot(l2, '-o')
plt.axhline(0)
plt.ylabel("test LL (null-adjusted)")
plt.xlabel("Unit Id")
# %%

xy = model.readout.mu.detach().cpu().numpy().squeeze()
plt.plot(xy[:,0], xy[:,1], '.')
ix = np.where(l2>.05)[0]
plt.plot(xy[ix,0], xy[ix,1], '.')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Readout')


#%% detach core, add shifter, retrain readout
ckpt_folder = save_dir / sessid / 'version_{}'.format(version) / 'checkpoints'

best_epoch = find_best_epoch(ckpt_folder)
print(best_epoch)

chkpath = str(ckpt_folder / 'epoch={}.ckpt'.format(best_epoch))
model2 = Encoder.load_from_checkpoint(chkpath, strict=True)

# clone model
# model2 = Encoder(core, readout, shifter=shifter)
# model2.load_state_dict(model.state_dict())

#%% turn off training core and readout
for param in model2.core.parameters():
        param.requires_grad = True
for param in model2.readout.parameters():
        param.requires_grad = True
#%%
# model2.detach_core = True # no more training the core
model2.train_shifter = True
model2.shifter.features[0].linear.weight.data[:] *= torch.tensor(0.01)
# %%
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
    auto_lr_find=True)

trainer.tune(model2, train_dl, valid_dl)
trainer.fit(model2, train_dl, valid_dl)
# %%

xx,yy = np.meshgrid(np.linspace(-5, 5,100),np.linspace(-5, 5,100))
xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))
# ygrid = torch.tensor( xx.astype('float32').flatten()))
inputs = torch.cat( (xgrid,ygrid), dim=1)
# inputs = torch.cat( (xgrid.repeat((1, 10)),ygrid.repeat((1,10))), dim=1)
shifter = model2.shifter
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
gd_test = PixelDataset(sessid, num_lags=num_lags,
    train=False, augment=None, stims='GaborNatImg', include_eyepos=True)

sample=gd_test[:]

model2.training = False
l3 = get_null_adjusted_ll(model2, sample, sample['robs'],bits=False)

plt.figure()
plt.plot(l2, l3, 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')
# plt.axhline(0)
plt.ylabel("test LL with shifter")
plt.xlabel("test LL no shifter")
# %%

xy = model2.readout.mu.detach().cpu().numpy().squeeze()
plt.plot(xy[:,0], xy[:,1], '.')
ix = np.where(l2>.05)[0]
plt.plot(xy[ix,0], xy[ix,1], '.')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Readout')
# %%

core = model2.core
readout = model2.readout
# readout.training = False
mu = readout.mu.detach().cpu()

n = 10
xx,yy = np.meshgrid(np.linspace(-.1, .1,n),np.linspace(-.1, .1,n))

L = np.zeros(xx.shape)

#%%
ind += 10
sample = gd_test[np.arange(ind, ind + 50)]
x = core(sample['stim'])

for ii in range(n):
    for jj in range(n):
        readout.mu[:,:,:,0] = mu[:,:,:,0] + xx[ii,jj]
        readout.mu[:,:,:,1] = mu[:,:,:,1] + yy[ii,jj]

        yhat = model2.output_nl(readout(x))
        L[ii,jj] = -model2.loss(yhat, sample['robs']).detach().cpu().numpy()


plt.imshow(L)
plt.colorbar()
# %%
