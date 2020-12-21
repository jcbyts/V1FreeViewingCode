import warnings
import numpy as np
from collections import OrderedDict, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

# import regularizers
import V1FreeViewingCode.models.regularizers as regularizers
from V1FreeViewingCode.models.layers import AdaptiveELU, PosLinear, posConv2D, EiMask, powNL, divNorm, ShapeLinear

""" 
Base classes:
"""
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
    """
    2D convolutional core base class
    """
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
============================================================================
Specific Cores
============================================================================
1. Shared NIM
2. Shared GQM
3. Shared DivNorm
4. Stacked 2D convolutional core
5. Stacked 2D NIM EI
"""

"""
Shared NIM core
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
        input_regularizer="LaplaceL2norm",
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

        input_kern = input_size[-2:]
        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kern)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)
        
        self.save_hyperparameters()
        self.input_channels = np.prod(input_size)

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_size = input_size
        # self.input_channels = np.prod(input_size)
        # self.hidden_channels = hidden_channels
        self.skip = skip
        self.use_avg_reg = use_avg_reg
        # self.ei_split = ei_split

        if use_avg_reg:
            warnings.warn("The averaged value of regularizer will be used.", UserWarning)

        
        # dictates how to concatenate outputs
        if stack is None:
            self.stack = list(range(self.hparams.layers))
        else:
            self.stack = [*range(self.hparams.layers)[stack:]] if isinstance(stack, int) else stack


        # initialize network
        self.features = nn.Sequential()

        # --- first layer
        layer = OrderedDict()
        if weight_norm:
            layer["conv"] = nn.utils.weight_norm( # call Shape linear looks like conv for regularization
                    ShapeLinear(input_size, hidden_channels,
                    bias=bias),
                    dim=weight_norm_dim,
                    name='weight')
        else:
            layer["conv"] = ShapeLinear(input_size,
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
        return self.group_sparsity() * self.hparams.gamma_hidden + self.hparams.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return (len(self.features)) * self.hparams.hidden_channels
    
    def plot_filters(self, sort=False):
        import numpy as np
        import matplotlib.pyplot as plt  # plotting
        ei_mask = self.features.layer0.eimask.ei_mask.detach().cpu().numpy()
        w = self.features.layer0.conv.weight.detach().cpu().numpy()
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



"""
STACKED 2D Convolutional Core (based on Sinz lab code)
"""
class Stacked2dCore(Core2d):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        layers=1,
        gamma_hidden=0,
        gamma_input=0.0,
        gamma_center=.1, # regularize the first conv layer to be centered
        skip=0,
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
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            Implemented as layers_to_stack = layers[stack:]. thus:
                                stack = -1 will only select the last layer as the readout layer.
                                stack of -2 will read out from the last two layers.
                                And stack of 1 will read out from layer 1 (0 indexed) until the last layer.

            use_avg_reg:    bool. Whether to use the averaged value of regularizer(s) or the summed.
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

        if stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = [*range(self.layers)[stack:]] if isinstance(stack, int) else stack

        # center regularization
        regw = 1 - regularizers.gaussian2d(input_kern,sigma=input_kern//2)
        self.register_buffer("center_reg_weights", torch.tensor(regw))

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
        if self.stack:
            ret = len(self.stack) * self.hidden_channels
        else:
            ret = len(self.features) * self.hidden_channels
        return ret
    
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

"""
Inherit Stacked 2D Core to build EI convNIM
"""
class Stacked2dEICore(Stacked2dCore):
    def __init__(
        self,
        input_channels=10,
        hidden_channels=10,
        input_kern=9,
        hidden_kern=9,
        prop_inh=.5,
        activation="elu",
        final_nonlinearity=True,
        bias=False,
        pad_input=True,
        hidden_padding=None,
        group_norm=True,
        num_groups=2,
        weight_norm=True,
        hidden_dilation=1,
        **kwargs):

        self.save_hyperparameters()

        super().__init__(input_channels,hidden_channels,input_kern,hidden_kern,**kwargs)

        self.features = nn.Sequential()

        # specify EI split
        ni = int(np.ceil(prop_inh*hidden_channels))
        ne = hidden_channels - ni
        self.num_inh = ni
        self.num_exc = ne

        if activation=="elu":
            self.activation = AdaptiveELU(0.0,1.0)
        else:
            self.activation = nn.ReLU()

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

        if self.layers > 1 or final_nonlinearity:
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
                    hidden_channels if not self.skip > 1 else min(self.skip, l) * hidden_channels,
                    hidden_channels,
                    hidden_kern[l - 1],
                    padding=hidden_padding,
                    bias=bias,
                    dilation=hidden_dilation),
                    dim=0)

            else:
                layer["conv"] = nn.Conv2d(
                    hidden_channels if not self.skip > 1 else min(self.skip, l) * hidden_channels,
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

"""
Stacked 2D convGQM
"""
class Stacked2dGqmCore(Stacked2dCore):
    def __init__(
        self,
        input_channels=10,
        hidden_channels=10,
        input_kern=9,
        hidden_kern=9,
        prop_lin=.5,
        final_nonlinearity=True,
        bias=False,
        pad_input=True,
        hidden_padding=None,
        group_norm=True,
        num_groups=2,
        weight_norm=True,
        hidden_dilation=1,
        **kwargs):

        self.save_hyperparameters()

        super().__init__(input_channels,hidden_channels,input_kern,hidden_kern,**kwargs)

        self.features = nn.Sequential()

        # specify EI split
        nlin = int(np.ceil(prop_lin*hidden_channels))
        nquad = hidden_channels - nlin
        self.num_lin = nlin
        self.num_quad = nquad

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

        if self.layers > 1 or final_nonlinearity:
            layer["nonlin"] = powNL( np.concatenate( (np.ones(nlin), 2*np.ones(nquad))), rectified=False)

        if group_norm:
            layer["norm"] = nn.GroupNorm(num_groups, hidden_channels)

        self.features.add_module("layer0", nn.Sequential(layer))

        # --- other layers
        if not isinstance(hidden_kern, Iterable):
            hidden_kern = [hidden_kern] * (self.layers - 1)

        for l in range(1, self.layers):
            layer = OrderedDict()

            hidden_padding = ((hidden_kern[l - 1] - 1) * hidden_dilation + 1) // 2
            if weight_norm:
                layer["conv"] = nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels if not self.skip > 1 else min(self.skip, l) * hidden_channels,
                    hidden_channels,
                    hidden_kern[l - 1],
                    padding=hidden_padding,
                    bias=bias,
                    dilation=hidden_dilation),
                    dim=0)

            else:
                layer["conv"] = nn.Conv2d(
                    hidden_channels if not self.skip > 1 else min(self.skip, l) * hidden_channels,
                    hidden_channels,
                    hidden_kern[l - 1],
                    padding=hidden_padding,
                    bias=bias,
                    dilation=hidden_dilation,
                )

            if final_nonlinearity or l < self.layers - 1:
                layer["nonlin"] = powNL( np.concatenate( (np.ones(nlin), 2*np.ones(nquad))), rectified=False)

            if group_norm:
                layer["norm"] = nn.GroupNorm(num_groups, hidden_channels)

            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        self.apply(self.init_conv)