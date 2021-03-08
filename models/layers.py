import math
import torch
from torch import nn
from typing import Tuple, Union

# from pytorch_lightning import LightningModule
from torch.nn import functional as F

from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t, _size_3_t # for conv2,conv3 default
from torch.nn.modules.utils import _triple # for posconv3

"""
1. posConv2D:   2D convolutional with positive weights
2. PosLinear:   Linear layer with positive weights
3. AdaptiveELU: adaptive exponential linear unit
4. EiMask:      apply excitation/inhibition mask on outputs

"""
class posConv2D(nn.Conv2d):
    """
    2D convolutional layer that is constrained to have positive weights
    """
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

class posConv3d(nn.Conv3d):
    """
    3D convolutional layer that is constrained to have positive weights
    """
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):

        super(posConv3d, self).__init__(in_channels,
            out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)
        self.register_buffer("minval", torch.tensor(0.0))

    def forward(self, input: Tensor) -> Tensor:
        posweight = torch.maximum(self.weight, self.minval)
        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            posweight, self.bias, self.stride, _triple(0),
                            self.dilation, self.groups)
        return F.conv3d(input, posweight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class PosLinear(nn.Linear):
    """
    Linear layer with constrained positive weights
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PosLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.register_buffer("minval", torch.tensor(0.0))

    def forward(self, x):
        pos_weight = torch.maximum(self.weight, self.minval)
        return F.linear(x, pos_weight, self.bias)


from torch.nn import init
from torch.nn.parameter import Parameter

class ShapeLinear(nn.Module):
    """
    Linear layer with true shape to weights
    
    Input is flattened and then it works like a normal linear layer. This only helps to do regularization.

    """
    def __init__(self, in_features, out_features:int, bias: bool = True, positive: bool = False):
        super(ShapeLinear, self).__init__()

        self.in_features = in_features
        # self.flatten = nn.Flatten()
        
        self.shape = tuple([out_features] + list(in_features))
        self.positive_constraint = positive

        self.weight = Parameter(torch.Tensor( size =self.shape ))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_buffer('bias', torch.zeros(out_features))

        if self.positive_constraint:
            self.register_buffer("minval", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)          

    def forward(self, x):
        w = self.weight
        if self.positive_constraint:
            w = torch.maximum(w, self.minval)
        x = torch.einsum('ncwh,kcwh->nk', x, w)
        return x + self.bias
        # if self.bias:
        # x = x + self.bias
        



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
        """
        ni: number of inhibitory units
        ne: number of excitatory units
        """
        super(EiMask,self).__init__()

        self.register_buffer("ei_mask", torch.cat((torch.ones((1,ne)), -torch.ones((1,ni))), axis=1).squeeze())
    
    def forward(self, x):
        out = torch.einsum("nc...,c->nc...", x, self.ei_mask)
        return out

class powNL(nn.Module):
    '''
    rectified powerlaw nonlinearity
    '''

    def __init__(self, power=1.5, rectified:bool=True):
        super(powNL, self).__init__()
        self.relu = F.relu
        self.rectified = rectified
        self.register_buffer("weight", torch.tensor(power.astype('float32')))

    def forward(self, x):
        if self.rectified:
            x = self.relu(x)
        if len(x.shape)==4: # 2D convolutional
            x = x.permute((0,2,3,1)).pow_(self.weight).permute((0,3,1,2))
        else:    
            x = x.pow_(self.weight)
        return x

class divNorm(nn.Module):
    """
    Divisive Normalization layer

    """
    def __init__(self, in_features):
        super(divNorm, self).__init__()
        
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features, in_features))
        self.bias = Parameter(torch.Tensor(in_features))

        self.relu = F.relu
        self.reset_parameters()

    def reset_parameters(self) -> None:
        print("divNorm: initialize weights custom")
        nn.init.uniform(self.weight, 0.0, 1.0)
        nn.init.uniform(self.bias, 0.0, 1.0)

    def forward(self, x):

        posweight = self.relu(self.weight)
        x = self.relu(x)
        xdiv = torch.einsum('nc...,ck->nk...', x, posweight)
        if len(x.shape)==4:
            xdiv = xdiv + self.bias[None,:,None,None] # is convolutional
        else:
            xdiv = xdiv + self.bias[None,:]

        x = x / xdiv.clamp_(0.001) # divide

        return x

class divNormPow(nn.Module):
    def __init__(self, in_features, power):
        super(divNormPow, self).__init__()

        if len(power)==in_features:
            self.pow = powNL
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


class Chomp(nn.Module):
    '''
    Chomp layer is for "chomping" off the result of padding on the output of a convolutional layer
    '''
    def __init__(self, chomp_sizes: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]], nb_dims):
        super(Chomp, self).__init__()
        if isinstance(chomp_sizes, int):
            self.chomp_sizes = [chomp_sizes] * nb_dims
        else:
            self.chomp_sizes = chomp_sizes

        assert len(self.chomp_sizes) == nb_dims, "must enter a chomp size for each dim"
        self.nb_dims = nb_dims

    def forward(self, x):
        input_size = x.size()

        if self.nb_dims == 1:
            slice_d = slice(0, input_size[2] - self.chomp_sizes[0], 1)
            return x[:, :, slice_d].contiguous()

        elif self.nb_dims == 2:
            if self.chomp_sizes[0] % 2 == 0:
                slice_h = slice(self.chomp_sizes[0] // 2, input_size[2] - self.chomp_sizes[0] // 2, 1)
            else:
                slice_h = slice(0, input_size[2] - self.chomp_sizes[0], 1)
            if self.chomp_sizes[2] % 2 == 0:
                slice_w = slice(self.chomp_sizes[1] // 2, input_size[3] - self.chomp_sizes[1] // 2, 1)
            else:
                slice_w = slice(0, input_size[3] - self.chomp_sizes[1], 1)
            return x[:, :, slice_h, slice_w].contiguous()

        elif self.nb_dims == 3:
            slice_d = slice(0, input_size[2] - self.chomp_sizes[0], 1)
            if self.chomp_sizes[1] % 2 == 0:
                slice_h = slice(self.chomp_sizes[1] // 2, input_size[3] - self.chomp_sizes[1] // 2, 1)
            else:
                slice_h = slice(0, input_size[3] - self.chomp_sizes[1], 1)
            if self.chomp_sizes[2] % 2 == 0:
                slice_w = slice(self.chomp_sizes[2] // 2, input_size[4] - self.chomp_sizes[2] // 2, 1)
            else:
                slice_w = slice(0, input_size[4] - self.chomp_sizes[2], 1)
            return x[:, :, slice_d, slice_h, slice_w].contiguous()

        else:
            raise RuntimeError("Invalid number of dims")