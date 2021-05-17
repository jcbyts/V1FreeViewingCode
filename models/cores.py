import warnings
import numpy as np
from collections import OrderedDict, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

# import regularizers
import V1FreeViewingCode.models.regularizers as regularizers
from V1FreeViewingCode.models.layers import AdaptiveELU, PosLinear, posConv2D, posConv3d, EiMask, powNL, divNorm, ShapeLinear

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
        if type(sort) is np.ndarray:
            cinds = sort
        elif sort:
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

class Core3d(Core):
    '''
    3D convolutional core base class: use for temporal convolutions
    '''
    def initialize(self, cuda=False):
        self.apply(self.init_conv)

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

"""
============================================================================
Specific Cores
============================================================================
1. GLM (use identity readout)
2. GQM (uses identity readout)
3. Shared NIM
4. Stacked 2D convolutional core
5. Stacked 2D NIM EI
6. Stacked 2D Divisive Normalization model
7. 3D temporal convolution
"""

"""
Shared NIM core
"""

class GLM(Core):
    def __init__(
        self,
        input_size,
        output_channels,
        weight_norm=True,
        bias=True,
        activation="softplus",
        input_regularizer="RegMats",
        input_reg_types=["d2x", "d2t"],
        input_reg_amt=[.005,.001]
    ):

        super().__init__()

        self.save_hyperparameters()
        
        self.input_channels = input_size[0]

        regularizer_config = {'dims': input_size,
                            'type': input_reg_types, 'amount': input_reg_amt}
        self._input_weights_regularizer = regularizers.__dict__["RegMats"](**regularizer_config)

        self.features = nn.Sequential()

        # --- first layer
        layer = OrderedDict()
        if weight_norm:
            layer["conv"] = nn.utils.weight_norm( # call Shape linear looks like conv for regularization
                    ShapeLinear(input_size, output_channels,
                    bias=bias),
                    dim=0,
                    name='weight')
        else:
            layer["conv"] = ShapeLinear(input_size,
                output_channels,
                bias=bias)

        if activation=="softplus":
            layer["nonlin"] = nn.Softplus()
        elif activation=="relu":
            layer["nonlin"] = nn.ReLU()

        self.features.add_module("layer0", nn.Sequential(layer))

    def forward(self, input_):
        ret = self.features(input_)

        return ret
    
    def input_reg(self):
        return self._input_weights_regularizer(self.features[0].conv.weight)

    def regularizer(self):
        return self.input_reg()


"""
NIM old
"""
class NimCore(Core):
    def __init__(
        self,
        input_size,
        num_channels,
        ei_split=None,
        weight_norm=None,
        gamma_group=0, # group sparsity penalty
        act_funcs=None,
        divnorm=None,
        group_norm=None,
        group_norm_num=None,
        bias=None,
        weight_regularizer="RegMats",
        weight_reg_types=None,
        weight_reg_amt=None,
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

        
        
        self.save_hyperparameters()
        self.input_channels = np.prod(input_size)
        self.num_channels = num_channels
        self.layers = len(num_channels)
        print("%d layers" %self.layers)
        self.gamma_group = gamma_group
        self.input_size = input_size
        self.ei_split = ei_split
        self.act_funcs = act_funcs
        self.divnorm = divnorm
        self.group_norm = group_norm
        self.weight_norm = weight_norm
        self.bias = bias

        self.use_avg_reg = use_avg_reg

        # check arguments and set them up
        # setup regularization (using reg mats)
        if weight_reg_types is None:
            weight_reg_types = []
            reg_amt = []
            for l in range(self.layers):
                reg_amt.append([.25,.25,.5])
                weight_reg_types.append(["d2xt", "local","center"])

        self._weights_regularizer = nn.ModuleList() # modules list is a list of pytorch modules
        
        # first layer regularizer
        regularizer_config = {'dims': input_size,
                                'type': weight_reg_types[0], 'amount': weight_reg_amt[0]}

        self._weights_regularizer.append(regularizers.__dict__[weight_regularizer](**regularizer_config))

        # for l in range(1,self.layers):
        #     regularizer_config = {'dims': [num_channels[l-1]],
        #                     'type': weight_reg_types[l], 'amount': weight_reg_amt[l]}
        #     self._weights_regularizer.append(regularizers.__dict__[weight_regularizer](**regularizer_config))
        
        if self.ei_split is None:
            self.ei_split = [0 for i in range(self.layers)]
        
        if self.divnorm is None:
            self.divnorm = [0 for i in range(self.layers)]

        if self.weight_norm is None:
            self.weight_norm = [0 for i in range(self.layers)]
        
        if self.group_norm is None:
            self.group_norm = [0 for i in range(self.layers)]

        if self.act_funcs is None:
            self.act_funcs = ["relu" for i in range(self.layers)]
        
        if self.bias is None:
            self.bias = [0 for i in range(self.layers)]
        
        # dictates how to concatenate outputs
        if stack is None:
            self.stack = list(range(self.layers)) # output all layers --> full scaffold
        else:
            self.stack = [*range(self.layers)[stack:]] if isinstance(stack, int) else stack

        # initialize network
        self.features = nn.Sequential()

        # --- first layer
        layer = OrderedDict()
        if self.weight_norm[0]:
            # TODO: i've been calling all first operations "conv" for plotting, but this should be changed
            layer["conv"] = nn.utils.weight_norm( # call Shape linear looks like conv for regularization
                    ShapeLinear(input_size, num_channels[0],
                    bias=self.bias[0]),
                    dim=0,
                    name='weight')
        else:
            layer["conv"] = ShapeLinear(input_size,
                num_channels[0],
                bias=self.bias[0])

        # specify Exc / Inh split
        l = 0
        ninh = int(np.ceil(self.ei_split[l]*self.num_channels[l]))
        nexc = self.num_channels[l] - ninh

        # add activation
        if self.act_funcs[l]=="elu":
            layer["nonlin"] = AdaptiveELU(0.0,1.0)
        elif self.act_funcs[l]=="relu":
            layer["nonlin"] = nn.ReLU()
        elif self.act_funcs[l]=="pow":
            layer["nonlin"] = powNL(1.5)
        elif self.act_funcs[l]=="pow2":
            layer["nonlin"] = powNL(2)

        if ninh > 0:
            layer["eimask"] = EiMask(ninh, nexc)

        if self.divnorm[l]:
            layer["norm"] = divNorm(self.num_channels[l])
        elif self.group_norm[l]:
            layer["norm"] = nn.GroupNorm(2, self.num_channels[l])

        self.features.add_module("layer0", nn.Sequential(layer))

        # --- other layers
        for l in range(1, self.layers):
            layer = OrderedDict()

            if self.ei_split[l-1]>0: # previous layer has inhibitory subunitss
                Linear = PosLinear
            else:
                Linear = nn.Linear

            if self.weight_norm[l]:
                layer["conv"] = nn.utils.weight_norm(Linear(
                    self.num_channels[l-1],
                    self.num_channels[l],
                    bias=self.bias[l]),
                    dim=0)

            else:
                layer["conv"] = Linear(
                    self.num_channels[l-1],
                    self.num_channels[l],
                    bias=self.bias[l])

            # specify Exc / Inh split
            ninh = int(np.ceil(self.ei_split[l]*self.num_channels[l]))
            nexc = self.num_channels[l] - ninh

            # add activation
            if self.act_funcs[l]=="elu":
                layer["nonlin"] = AdaptiveELU(0.0,1.0)
            elif self.act_funcs[l]=="relu":
                layer["nonlin"] = nn.ReLU()
            elif self.act_funcs[l]=="pow":
                layer["nonlin"] = powNL(1.5)
            elif self.act_funcs[l]=="pow2":
                layer["nonlin"] = powNL(2)

            if ninh > 0:
                layer["eimask"] = EiMask(ninh, nexc)

            if self.divnorm[l]:
                layer["norm"] = divNorm(self.num_channels[l])
            elif self.group_norm[l]:
                layer["norm"] = nn.GroupNorm(2, self.num_channels[l])

            self.features.add_module("layer{}".format(l), nn.Sequential(layer))


    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            input_ = feat(input_)
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def weight_reg(self):
        ret = 0
        for l in [0]:
            ret = ret + self._weights_regularizer[l](self.features[l].conv.weight)
        return ret

    def group_sparsity(self):
        ret = 0
        for l in range(0, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum().sqrt()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.weight_reg() + self.gamma_group * self.group_sparsity()

    @property
    def outchannels(self):
        ret = 0
        if self.stack:
            for l in self.stack:
                ret += self.features[l].conv.weight.shape[0]
        else:
            for l in range(self.layers):
                ret += self.features[l].conv.weight.shape[0]
        return ret
    
    def plot_filters(self, sort=False, cmaps=None):
        import numpy as np
        import matplotlib.pyplot as plt  # plotting
        
        w = self.features.layer0.conv.weight.detach().cpu().numpy()
        sz = w.shape

        if sz[3]==1:
            is1D = True
        else:
            is1D = False

        if cmaps is None:
            cmaps = [plt.cm.coolwarm, plt.cm.RdBu]
        # w = model.features.weight.detach().cpu().numpy()
        w = w.reshape(sz[0], sz[1], sz[2]*sz[3])
        nfilt = w.shape[0]

        if self.ei_split[0] > 0:
            ei_mask = self.features.layer0.eimask.ei_mask.detach().cpu().numpy()
        else:
            ei_mask = np.ones(nfilt)

        if type(sort) is np.ndarray:
            cinds = sort
        elif sort:
            n = np.asarray([w[i,:].abs().max().detach().numpy() for i in range(nfilt)])
            cinds = np.argsort(n)[::-1][-len(n):]
        else:
            cinds = np.arange(0, nfilt)

        if is1D:
            sx = np.ceil(np.sqrt(nfilt))
            sy = np.round(np.sqrt(nfilt))    
        else:
            sx = np.ceil(np.sqrt(nfilt*2))
            sy = np.round(np.sqrt(nfilt*2))
            mod2 = sy % 2
            sy += mod2
            sx -= mod2
        

        plt.figure(figsize=(sy*5,sx*5))
        for cc,jj in zip(cinds, range(nfilt)):
            wtmp = np.squeeze(w[cc,:])
            if is1D:
                plt.subplot(sx, sy, jj+1)
                if ei_mask[cc]>0:
                    plt.imshow(wtmp, cmap=cmaps[0])
                else:
                    plt.imshow(wtmp, cmap=cmaps[1])
            else:
                plt.subplot(sx,sy,jj*2+1)
                
                bestlag = np.argmax(np.std(wtmp, axis=1))
                plt.imshow(np.reshape(wtmp[bestlag,:], (sz[2], sz[3])), interpolation=None, cmap=cmaps[0])
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
2D Convolutional Core With Linear Orthogonal Filters
"""
class LinearOrthBasis2D(Core2d):
    def __init__(
        self,
        input_channels,
        num_channels,
        kernel_size=15,
        regularizer="RegMats",
        reg_types=["d2xt", "local","center"],
        reg_amt=[.25,.25,.25],
        weight_norm=True,
        group_norm=True,
        num_groups=4,
        gamma_orth=1,
        pad_input=True,
        bias=False):

        super().__init__()

        regularizer_config = {'dims': [input_channels, kernel_size, kernel_size],
                            'type': reg_types, 'amount': reg_amt}
        self._weights_regularizer = regularizers.__dict__[regularizer](**regularizer_config)
        self.save_hyperparameters()
        self.gamma_orth = gamma_orth
        self.num_channels = num_channels
        self.flatten = nn.Flatten()
        self.features = nn.Sequential()
        self.register_buffer('I', torch.eye(num_channels))
        # --- first layer
        layer = OrderedDict()
        if weight_norm:
            layer["conv"] = nn.utils.weight_norm(nn.Conv2d(
                input_channels,
                num_channels,
                kernel_size,
                padding=kernel_size // 2 if pad_input else 0,
                bias=bias and not group_norm),
                dim=0, name='weight')
        else:
            layer["conv"] = nn.Conv2d(
                input_channels,
                num_channels,
                kernel_size,
                padding=kernel_size // 2 if pad_input else 0,
                bias=bias and not group_norm,
            )

        if group_norm:
            layer["norm"] = nn.GroupNorm(num_groups, num_channels)

        self.features.add_module("layer0", nn.Sequential(layer))
    
    def forward(self, input_):
        return self.features(input_)

    def weight_reg(self):
        return self._weights_regularizer(self.features[0].conv.weight)

    def orth_reg(self):
        w = self.features[0].conv.weight
        w = self.flatten(w)
        w = w.T/w.norm(dim=1)
        offdiag = self.I - w.T@w

        return offdiag.pow(2).sum().sqrt()

    def regularizer(self):
        return self.weight_reg() + self.gamma_orth * self.orth_reg()

    @property
    def outchannels(self):
        return self.num_channels



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
        input_regularizer="RegMats",
        hidden_regularizer="RegMats",
        input_reg_types=["d2xt", "local","center"],
        input_reg_amt=[.25,.25,.5],
        hidden_reg_types=["d2x", "local", "center"],
        hidden_reg_amt=[.33,.33,.33],
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

        # regularizer_config = (
        #     dict(padding=laplace_padding, kernel=input_kern)
        #     if input_regularizer == "GaussianLaplaceL2"
        #     else dict(padding=laplace_padding)
        # )
        # self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)
        regularizer_config = {'dims': [input_channels, input_kern,input_kern],
                            'type': input_reg_types, 'amount': input_reg_amt}
        self._input_weights_regularizer = regularizers.__dict__["RegMats"](**regularizer_config)

        regularizer_config = {'dims': [hidden_channels, hidden_kern,hidden_kern],
                            'type': hidden_reg_types, 'amount': hidden_reg_amt}
        self._hidden_weights_regularizer = regularizers.__dict__["RegMats"](**regularizer_config)

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

        # # center regularization
        # regw = 1 - regularizers.gaussian2d(input_kern,sigma=input_kern//2)
        # self.register_buffer("center_reg_weights", torch.tensor(regw))

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l) :], dim=1))
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def input_reg(self):
        return self._input_weights_regularizer(self.features[0].conv.weight)

    def hidden_reg(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self._hidden_weights_regularizer(self.features[l].conv.weight)
        return self.gamma_hidden * self.group_sparsity() + ret
    # def center_reg(self):
    #     ret = torch.einsum('ijxy,xy->ij', self.features[0].conv.weight, self.center_reg_weights).pow(2).mean()
    #     return ret

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.hidden_reg() + self.gamma_input * self.input_reg()

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
        if type(sort) is np.ndarray:
            cinds = sort
        elif sort:
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
        activation="relu",
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

class Stacked2dDivNorm(Stacked2dCore):
    def __init__(
        self,
        input_channels=10,
        hidden_channels=10,
        input_kern=9,
        hidden_kern=9,
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
        from V1FreeViewingCode.models.layers import AdaptiveELU
        
        if activation=="elu":
            self.activation = AdaptiveELU(0.0,1.0)
        elif activation=="relu":
            self.activation = nn.ReLU()
        elif activation=="pow":
            self.activation = powNL(1.5)

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
            layer["nonlin"] = self.activation
            layer["norm"] = divNorm(hidden_channels)

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

        # specify Lin / Quad split
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





"""
3D Convolutional Core
"""
class Stacked3dCore(Core3d):
    def __init__(
        self,
        num_channels,
        kernel_size,
        input_channels=1,
        act_funcs=None,
        ei_split=None,
        divnorm=None,
        weight_norm=None,
        group_norm=None,
        bias=None,
        regularizer="RegMats",
        reg_types=None,
        reg_amt=None,
        gamma_group=.5,
        pad_input=0,
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
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            Implemented as layers_to_stack = layers[stack:]. thus:
                                stack = -1 will only select the last layer as the readout layer.
                                stack of -2 will read out from the last two layers.
                                And stack of 1 will read out from layer 1 (0 indexed) until the last layer.

            use_avg_reg:    bool. Whether to use the averaged value of regularizer(s) or the summed.
        """

        super().__init__()

        self.layers = len(num_channels)
        self.input_channels = input_channels # number of input channels
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.use_avg_reg = use_avg_reg
        self.ei_split = ei_split
        self.divnorm = divnorm
        self.act_funcs = act_funcs
        self.weight_norm = weight_norm
        self.group_norm = group_norm
        self.bias = bias
        self.gamma_group = gamma_group

        # setup regularization (using reg mats)
        if reg_types is None:
            reg_types = []
            reg_amt = []
            for l in range(self.layers):
                reg_amt.append([.25,.25,.5])
                reg_types.append(["d2xt", "local","center"])

        self._weights_regularizer = nn.ModuleList()
        for l in range(self.layers):
            regularizer_config = {'dims': list(kernel_size[l]),
                                'type': reg_types[l], 'amount': reg_amt[l]}
            self._weights_regularizer.append(regularizers.__dict__[regularizer](**regularizer_config))
        
        if self.ei_split is None:
            self.ei_split = [0 for i in range(self.layers)]
        
        if self.divnorm is None:
            self.divnorm = [0 for i in range(self.layers)]

        if self.weight_norm is None:
            self.weight_norm = [0 for i in range(self.layers)]
        
        if self.group_norm is None:
            self.group_norm = [0 for i in range(self.layers)]

        if self.act_funcs is None:
            self.act_funcs = ["relu" for i in range(self.layers)]
        
        if self.bias is None:
            self.bias = [0 for i in range(self.layers)]

        if use_avg_reg:
            warnings.warn("The averaged value of regularizer will be used.", UserWarning)

        if stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = [*range(self.layers)[stack:]] if isinstance(stack, int) else stack

        # --- BUILD NETWORK
        self.features = nn.Sequential()

        # --- first layer
        layer = OrderedDict()

        # specify Exc / Inh split
        ninh = int(np.ceil(self.ei_split[0]*self.num_channels[0]))
        nexc = self.num_channels[0] - ninh

        if self.weight_norm[0]:
            layer["conv"] = nn.utils.weight_norm(nn.Conv3d(
                input_channels,
                self.num_channels[0],
                self.kernel_size[0],
                padding=tuple(np.asarray(self.kernel_size[0])//2),
                bias=self.bias[0] and not self.group_norm[0]),
                dim=0, name='weight')
        else:
            layer["conv"] = nn.Conv3d(
                input_channels,
                self.num_channels[0],
                self.kernel_size[0],
                padding=tuple(np.asarray(self.kernel_size[0])//2),
                bias=self.bias[0] and not self.group_norm[0])

        # add activation
        if self.act_funcs[0]=="elu":
            layer["nonlin"] = AdaptiveELU(0.0,1.0)
        elif self.act_funcs[0]=="relu":
            layer["nonlin"] = nn.ReLU()
        elif self.act_funcs[0]=="pow":
            layer["nonlin"] = powNL(1.5)

        if ninh > 0:
            layer["eimask"] = EiMask(ninh, nexc)

        if self.divnorm[0]:
            layer["norm"] = divNorm(self.num_channels[0])
        elif self.group_norm[0]:
            layer["norm"] = nn.GroupNorm(2, self.num_channels[0])

        self.features.add_module("layer0", nn.Sequential(layer))

        # --- other layers
        for l in range(1, self.layers):
            layer = OrderedDict()

            if self.ei_split[l-1]>0: # previous layer has inhibitory subunitss
                conv3d = posConv3d
            else:
                conv3d = nn.Conv3d

            if self.weight_norm[l]:
                layer["conv"] = nn.utils.weight_norm(conv3d(
                    self.num_channels[l-1],
                    self.num_channels[l],
                    self.kernel_size[l],
                    padding=tuple(np.asarray(self.kernel_size[l])//2),
                    bias=self.bias[l]),
                    dim=0)

            else:
                layer["conv"] = conv3d(
                    self.num_channels[l-1],
                    self.num_channels[l],
                    self.kernel_size[l],
                    padding=tuple(np.asarray(self.kernel_size[l])//2),
                    bias=self.bias[l])

            # specify Exc / Inh split
            ninh = int(np.ceil(self.ei_split[l]*self.num_channels[l]))
            nexc = self.num_channels[l] - ninh

            # add activation
            if self.act_funcs[l]=="elu":
                layer["nonlin"] = AdaptiveELU(0.0,1.0)
            elif self.act_funcs[l]=="relu":
                layer["nonlin"] = nn.ReLU()
            elif self.act_funcs[l]=="pow":
                layer["nonlin"] = powNL(1.5)

            if ninh > 0:
                layer["eimask"] = EiMask(ninh, nexc)

            if self.divnorm[l]:
                layer["norm"] = divNorm(self.num_channels[l])
            elif self.group_norm[l]:
                layer["norm"] = nn.GroupNorm(2, self.num_channels[l])

            self.features.add_module("layer{}".format(l), nn.Sequential(layer))
        
        self.apply(self.init_conv)


    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            input_ = feat(input_)
            ret.append(input_[:,:,-1,:,:].squeeze()) # only save the last timepoint

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def weight_reg(self):
        ret = 0
        for l in range(self.layers):
            nch = self.features[l].conv.weight.shape[1]
            for ch in range(nch): # loop over input channels
                ret = ret + self._weights_regularizer[l](self.features[l].conv.weight[:,ch,:,:])
        return ret

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.weight_reg() + self.gamma_group * self.group_sparsity()

    @property
    def outchannels(self):
        ret = 0
        if self.stack:
            for l in self.stack:
                ret += self.features[l].conv.weight.shape[0]
        else:
            for l in range(self.layers):
                ret += self.features[l].conv.weight.shape[0]
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
        if type(sort) is np.ndarray:
            cinds = sort
        elif sort:
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
