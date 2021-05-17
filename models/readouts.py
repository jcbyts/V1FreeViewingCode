import torch
from torch import nn
from collections import OrderedDict
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from .layers import PosLinear
from torch.nn.parameter import Parameter

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
============================================================================
Specific Readouts
============================================================================
1. Identity - this readout does nothing (depends on the core to the output dimensions)
2. Full - channel x width x height
3. Gaussian 2D
"""

class IdentityReadout(Readout):
    """
    This readout does nothing. It has no parameters. It is not trained. No output nonlinearity. It assumes the Core does everything.
    """
    def __init__(self, input_size, bias=True):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.register_buffer("regvalplaceholder", torch.zeros(1))

        if bias:
            bias = Parameter(torch.Tensor(input_size))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)
    
    def forward(self,x):
        return x + self.bias.clamp(min=0.0)
    
    def regularizer(self):
        return self.regvalplaceholder.sum()

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

        self.save_hyperparameters()
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

    def forward(self, x, shift=None):        
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


"""
GAUSSIAN READOUT CORE FROM SINZ LAB CODE
"""

class Point2DGaussian(Readout):
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
        shifter (dict): Parameters for a predictor of shfiting grid locations. Has to have a form like
                        {
                        'hidden_layers':1,
                        'hidden_features':20,
                        'final_tanh': False,
                        }
    """

    def __init__(self, in_shape=[10,10,10],
                outdims=10,
                bias=True,
                init_mu_range=0.1,
                init_sigma=1,
                gamma_l1=0.001,
                gamma_l2=0.1,
                batch_sample=True,
                align_corners=True,
                gauss_type='uncorrelated',
                shifter=None,
                constrain_positive=False,
                **kwargs):

        super().__init__()

        # pytorch lightning helper to save all hyperparamters
        self.save_hyperparameters()

        # determines whether the Gaussian is isotropic or not
        self.gauss_type = gauss_type

        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")

        # store statistics about the images and neurons
        self.in_shape = in_shape
        self.outdims = outdims

        # sample a different location per example
        self.batch_sample = batch_sample

        # constrain feature vector to be positive
        self.constrain_positive = constrain_positive

        # position grid shape
        self.grid_shape = (1, outdims, 1, 2)

        # initialize means
        self._mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron

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

        self.initialize_features()

        if shifter:
            self.shifter = nn.Sequential()
            layer = OrderedDict()
            if shifter["hidden_layers"]==0:
                layer["linear"] = nn.Linear(2, 2, bias=True)
                if shifter["final_tanh"]:
                    layer["activation"] = nn.Tanh()
            else:
                layer["linear"] = nn.Linear(2, shifter["hidden_features"], bias=False)
                if "activation" in shifter.keys():
                    if shifter["activation"]=="relu":
                        layer["activation"] = nn.ReLU()
                    elif shifter["activation"]=="softplus":
                        if "lengthscale" in shifter.keys():
                            layer["activation"] = nn.Softplus(beta=shifter["lengthscale"])
                        else:
                            layer["activation"] = nn.Softplus()
                else:
                        layer["activation"] = nn.ReLU()
            
            self.shifter.add_module("layer0", nn.Sequential(layer))

            for l in range(1,shifter['hidden_layers']+1):
                layer = OrderedDict()
                if l == shifter['hidden_layers']: # is final layer
                    layer["linear"] = nn.Linear(shifter["hidden_features"],2,bias=True)
                    if shifter["final_tanh"]:
                        layer["activation"] = nn.Tanh()
                else:
                    layer["linear"] = nn.Linear(shifter["hidden_features"],shifter["hidden_features"],bias=True)
                    if "activation" in shifter.keys():
                        if shifter["activation"]=="relu":
                            layer["activation"] = nn.ReLU()
                        elif shifter["activation"]=="softplus":
                            if "lengthscale" in shifter.keys():
                                layer["activation"] = nn.Softplus(beta=shifter["lengthscale"])
                            else:
                                layer["activation"] = nn.Softplus()
                    else:
                        layer["activation"] = nn.ReLU()

                self.shifter.add_module("layer{}".format(l), nn.Sequential(layer))
        else:
            self.shifter = None

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.register_buffer("regvalplaceholder", torch.zeros((1,2)))
        self.init_mu_range = init_mu_range
        self.align_corners = align_corners
        self.initialize()

    @property
    def features(self):
        if self.constrain_positive:
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
            self.mu.clamp(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
            if self.gauss_type != 'full':
                self.sigma.clamp(min=0)  # sigma/variance i    s always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]

        sample = self.training if sample is None else sample
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

        if self.gauss_type != 'full':
            return (norm * self.sigma + self.mu).clamp(-1,1) # grid locations in feature space sampled randomly around the mean self.mu
        else:
            return (torch.einsum('ancd,bnid->bnic', self.sigma, norm) + self.mu).clamp_(-1,1) # grid locations in feature space sampled randomly around the mean self.mu


    def initialize(self):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """

        self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.gauss_type != 'full':
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)

        self._features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def initialize_features(self, match_ids=None):
        import numpy as np
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, w, h = self.in_shape
        self._original_features = True
        if match_ids is not None:
            raise ValueError(f'match_ids to combine across session "{match_ids}" is not implemented yet')
        else:
            self._features = Parameter(torch.Tensor(1, c, 1, self.outdims))  # feature weights for each channel of the core
            self._shared_features = False
    

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
            # sample the grid_locations separately per sample per batch
            grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all sample in the batch
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
            # shifter is run outside the readout forward
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y

    def regularizer(self):
        if self.shifter is None:
            out = 0
        else:
            out = self.shifter(self.regvalplaceholder).abs().sum()*10
        # enforce the shifter to have 0 shift at 0,0 in
        feat = self.features
        out = out + self.hparams.gamma_l2 * feat.pow(2).mean().sqrt() + self.hparams.gamma_l1 * feat.abs().mean()
        return out

    def __repr__(self):
        """
        returns a string with setup of this model
        """
        c, w, h = self.in_shape
        r = self.gauss_type + ' '
        r += self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        if self.shifter is not None:
            r += " with shifter"

        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r