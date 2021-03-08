import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# def laplace():
#     return np.array([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]).astype(np.float32)[None, None, ...]


def laplace():
    """
    Returns a 3x3 laplace filter.

    """
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)[None, None, ...]


def laplace3d():
    l = np.zeros((3, 3, 3))
    l[1, 1, 1] = -6.0
    l[1, 1, 2] = 1.0
    l[1, 1, 0] = 1.0
    l[1, 0, 1] = 1.0
    l[1, 2, 1] = 1.0
    l[0, 1, 1] = 1.0
    l[2, 1, 1] = 1.0
    return l.astype(np.float32)[None, None, ...]


def gaussian2d(size, sigma=5, gamma=1, theta=0, center=(0, 0), normalize=True):
    """
    Returns a 2D Gaussian filter.

    Args:
        size (tuple of int, or int): Image height and width.
        sigma (float): std deviation of the Gaussian along x-axis. Default is 5..
        gamma (float): ratio between std devidation along x-axis and y-axis. Default is 1.
        theta (float): Orientation of the Gaussian (in ratian). Default is 0.
        center (tuple): The position of the filter. Default is center (0, 0).
        normalize (bool): Whether to normalize the entries. This is computed by
            subtracting the minimum value and then dividing by the max. Default is True.

    Returns:
        2D Numpy array: A 2D Gaussian filter.

    """

    sigma_x = sigma
    sigma_y = sigma / gamma

    xmax, ymax = (size, size) if isinstance(size, int) else size
    xmax, ymax = (xmax - 1) / 2, (ymax - 1) / 2
    xmin, ymin = -xmax, -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # shift the position
    y -= center[0]
    x -= center[1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gaussian = np.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))

    if normalize:
        gaussian -= gaussian.min()
        gaussian /= gaussian.max()

    return gaussian.astype(np.float32)


class Laplace(nn.Module):
    """
    Laplace filter for a stack of data. Utilized as the input weight regularizer.
    """

    def __init__(self, padding=None):
        """
        Laplace filter for a stack of data.

        Args:
            padding (int): Controls the amount of zero-padding for the convolution operation
                            Default is half of the kernel size (recommended)

        Attributes:
            filter (2D Numpy array): 3x3 Laplace filter.
            padding_size (int): Number of zeros added to each side of the input image
                before convolution operation.
        """
        super().__init__()
        self.register_buffer("filter", torch.from_numpy(laplace()))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def forward(self, x):
        return F.conv2d(x, self.filter, bias=None, padding=self.padding_size)


class LaplaceL2(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer. Unnormalized, not recommended to use.
        Use LaplaceL2norm instead.

        Args:
            padding (int): Controls the amount of zero-padding for the convolution operation.

        Attributes:
            laplace (Laplace): Laplace convolution object. The output is the result of
                convolving an input image with laplace filter.

    """

    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace(padding=padding)
        warnings.warn("LaplaceL2 Regularizer is deprecated. Use LaplaceL2norm instead.")

    def forward(self, x, avg=True):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1, k2)).pow(2)) / 2


class LaplaceL2norm(nn.Module):
    """
    Normalized Laplace regularizer for a 2D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1, k2)).pow(2)) / agg_fn(x.view(oc * ic, 1, k1, k2).pow(2))


class Laplace3d(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self, padding=None):
        super().__init__()
        self.register_buffer("filter", torch.from_numpy(laplace3d()))

    def forward(self, x):
        return F.conv3d(x, self.filter, bias=None)


class LaplaceL23d(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace3d()

    def forward(self, x):
        oc, ic, k1, k2, k3 = x.size()
        return self.laplace(x.view(oc * ic, 1, k1, k2, k3)).pow(2).mean() / 2


class FlatLaplaceL23d(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, x):
        oc, ic, k1, k2, k3 = x.size()
        assert k1 == 1, "time dimension must be one"
        return self.laplace(x.view(oc * ic, 1, k2, k3)).pow(2).mean() / 2


class LaplaceL1(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self, padding=0):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, avg=True):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1, k2)).abs())


class GaussianLaplaceL2Adaptive(nn.Module):
    """
    Laplace regularizer, with a Gaussian mask, for a 2D convolutional layer.
        Is flexible across kernel sizes, so that the regularizer can be applied to more
        than one layer, with changing kernel sizes
    """

    def __init__(self, padding=None, sigma=None):
        """
        Args:
            padding (int): Controls the amount of zero-padding for the convolution operation.
            sigma (float): std deviation of the Gaussian along x-axis. Default is calculated
                as the 1/4th of the minimum dimenison (height vs width) of the input.
        """
        super().__init__()
        self.laplace = Laplace(padding=padding)
        self.sigma = sigma

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        sigma = self.sigma if self.sigma else min(k1, k2) / 4

        out = self.laplace(x.view(ic * oc, 1, k1, k2))
        out = out * (1 - torch.from_numpy(gaussian2d(size=(k1, k2), sigma=sigma)).expand(1, 1, k1, k2).to(x.device))

        return agg_fn(out.pow(2)) / agg_fn(x.view(oc * ic, 1, k1, k2).pow(2))


class GaussianLaplaceL2(nn.Module):
    """
    Laplace regularizer, with a Gaussian mask, for a single 2D convolutional layer.

    """

    def __init__(self, kernel, padding=None):
        """
        Args:
            kernel: Size of the convolutional kernel of the filter that is getting regularized
            padding (int): Controls the amount of zero-padding for the convolution operation.
        """
        super().__init__()

        self.laplace = Laplace(padding=padding)
        self.kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        sigma = min(*self.kernel) / 4
        self.gaussian2d = torch.from_numpy(gaussian2d(size=(*self.kernel,), sigma=sigma))

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        out = self.laplace(x.view(oc * ic, 1, k1, k2))
        out = out * (1 - self.gaussian2d.expand(1, 1, k1, k2).to(x.device))

        return agg_fn(out.pow(2)) / agg_fn(x.view(oc * ic, 1, k1, k2).pow(2))

class RegMats(nn.Module):
    """
        Build regularizaiton matrices 
    """

    def __init__(self, dims=[10,10,10], type=['d2xt'], amount=[1]):
        """
        Args:
            kernel: Size of the convolutional kernel of the filter that is getting regularized
            padding (int): Controls the amount of zero-padding for the convolution operation.
        """
        super().__init__()
        assert len(type) == len(amount)
        amount = np.asarray(amount)
        newdims = [dims[c] for c in (0,2,1)] # reorder because build_reg_mats expects lags at the end
        reg_mat = 0
        for rr in range(len(type)):
            reg_mat += amount[rr]*build_reg_mats(newdims, type[rr])

        self.register_buffer("reg_mat", torch.tensor(reg_mat.astype('float32')))
        self.flatten = nn.Flatten()

    def forward(self, x, avg=None):
        x = self.flatten(x.permute((0,2,3,1))) # reorder because regmats have lags at end TODO: fix this
        pen = x@self.reg_mat@x.T

        return pen.trace()


import numpy as np
import scipy.sparse as sp


def build_reg_mats(input_dims, reg_type):
    """Build regularization matrices in default tf Graph

    Args:
        input_dims  <list> [nlags ny nx]
        reg_type (str): see `_allowed_reg_types` for options
    """
    if (reg_type == 'd2t') or (reg_type == 'd2x') or (reg_type == 'd2xt'):
        reg_mat = create_tikhonov_matrix(input_dims, reg_type)
        reg_mat = reg_mat.T@reg_mat

    elif (reg_type == 'max') or (reg_type == 'max_filt') or (reg_type == 'max_space'):
        reg_mat = create_maxpenalty_matrix(input_dims, reg_type)

    elif reg_type == 'center':
        reg_mat = create_maxpenalty_matrix(input_dims, reg_type)
        
    elif reg_type == 'local':
        reg_mat = create_localpenalty_matrix(input_dims, separable=False)

    elif reg_type == 'glocal':
        reg_mat = create_localpenalty_matrix(input_dims, separable=False, spatial_global=True)
    else:
        reg_mat = 0.0

    return reg_mat

            
def create_tikhonov_matrix(stim_dims, reg_type, boundary_conditions=None):
    """
    Usage: Tmat = create_Tikhonov_matrix(stim_dims, reg_type, boundary_cond)

    Creates a matrix specifying a an L2-regularization operator of the form
    ||T*k||^2, where T is a matrix and k is a vector of parameters. Currently 
    only supports second derivative/Laplacian operations

    Args:
        stim_dims (list of ints): dimensions associated with the target 
            stimulus, in the form [num_lags, num_x_pix, num_y_pix]
        reg_type (str): specify form of the regularization matrix
            'd2xt' | 'd2x' | 'd2t'
        boundary_conditions (None): is a list corresponding to all dimensions
            [i.e. [False,True,True]: will be free if false, otherwise true)
            [default is [True,True,True]
            would ideally be a dictionary with each reg
            type listed; currently unused

    Returns:
        scipy array: matrix specifying the desired Tikhonov operator

    Notes:
        The method of computing sparse differencing matrices used here is 
        adapted from Bryan C. Smith's and Andrew V. Knyazev's function 
        "laplacian", available here: 
        http://www.mathworks.com/matlabcentral/fileexchange/27279-laplacian-in-1d-2d-or-3d
        Written in Matlab by James McFarland, adapted into python by Dan Butts

        Currently, the no-boundary condition case for all but temporal dimension alone is untested and possibly wrong
        due to the fact that it *seems* that the indexing in python flips the first and second dimensions and a
        transpose is thus necessary at the early stage. Not a problem (it seems) because boundary conditions are
        currently applied by default, which makes the edges zero....
    """

    if boundary_conditions is None:
        boundary_conditions = [True]*3
    else:
        if not isinstance(boundary_conditions, list):
            boundary_conditions = [boundary_conditions]*3

    # first dimension is assumed to represent time lags
    nLags = stim_dims[0]

    # additional dimensions are spatial (Nx and Ny)
    nPix = stim_dims[1] * stim_dims[2]
    allowed_reg_types = ['d2xt', 'd2x', 'd2t']

    # assert (ischar(reg_type) && ismember(reg_type, allowed_reg_types),
    # 'not an allowed regularization type');

    # has_split = ~isempty(stim_params.split_pts);
    et = np.ones([1, nLags], dtype=np.float32)
    ex = np.ones([1, stim_dims[1]], dtype=np.float32)
    ey = np.ones([1, stim_dims[2]], dtype=np.float32)

    # Boundary conditions (currently implemented clumsily)
    # if isinf(stim_params.boundary_conds(1)) # if temporal dim has free boundary
    if not boundary_conditions[0]:
        et[0, [0, -1]] = 0  # constrain temporal boundary to zero: all else are free
    #else:
    #    print('t-bound')
    # if isinf(stim_params.boundary_conds(2)) # if first spatial dim has free boundary
    if not boundary_conditions[1]:
        ex[0, [0, -1]] = 0
    #else:
    #    print('x-bound')
    # if isinf(stim_params.boundary_conds(3)); # if second spatial dim has free boundary
    if not boundary_conditions[2]:
        ey[0, [0, -1]] = 0
    #else:
    #    print('y-bound')

    if nPix == 1:  # for 0-spatial-dimensional stimuli can only do temporal

        assert reg_type == 'd2t', 'Using stimuli with no spatial dimensions: ' + reg_type + ' not possible.'

        Tmat = sp.spdiags(np.concatenate((et, -2 * et, et), axis=0), [-1, 0, 1], nLags, nLags)
        # Python makes them transposed relative to matlab -- making the following necessary in
        # order for the no-boundary conditions to work correctly
        Tmat = np.transpose(Tmat)
        # if stim_params.boundary_conds(1) == -1 # if periodic boundary cond
        #    Tmat(end, 1) = 1;
        #    Tmat(1, end) = 1;

    elif stim_dims[2] == 1:  # for 1 - spatial dimensional stimuli
        if reg_type == 'd2t':
            assert nLags > 1, 'No d2t regularization possible with no lags.'

            D1t = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((et, -2 * et, et), axis=0), [-1, 0, 1], nLags, nLags)))

            Ix = sp.eye(stim_dims[1])
            Tmat = sp.kron(Ix, D1t)

        elif reg_type == 'd2x':
            It = sp.eye(nLags)

            D1x = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ex, -2 * ex, ex), axis=0), [-1, 0, 1], stim_dims[1], stim_dims[1])))

            Tmat = sp.kron(D1x, It)

        elif reg_type == 'd2xt':

            D1t = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((et, -2 * et, et), axis=0), [-1, 0, 1], nLags, nLags)))

            D1x = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ex, -2 * ex, ex), axis=0), [-1, 0, 1], stim_dims[1], stim_dims[1])))

            It = sp.eye(nLags)
            Ix = sp.eye(stim_dims[1])
            Tmat = sp.kron(Ix, D1t) + sp.kron(D1x, It)
        else:
            print('Unsupported reg type (1):', reg_type)
            Tmat = None

    else:  # for stimuli with 2-spatial dimensions
        if reg_type == 'd2t':
            assert nLags > 1, 'No d2t regularization possible with no lags.'

            D1t = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((et, -2 * et, et), axis=0), [-1, 0, 1], nLags, nLags)))

            Ix = sp.eye(stim_dims[1])
            Iy = sp.eye(stim_dims[2])
            Tmat = sp.kron(Iy, sp.kron(Ix, D1t))

        elif reg_type == 'd2x':
            It = sp.eye(nLags)
            Ix = sp.eye(stim_dims[1])
            Iy = sp.eye(stim_dims[2])

            D1x = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ex, -2 * ex, ex), axis=0), [-1, 0, 1], stim_dims[1], stim_dims[1])))

            D1y = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ey, -2 * ey, ey), axis=0), [-1, 0, 1], stim_dims[2], stim_dims[2])))

            Tmat = sp.kron(Iy, sp.kron(D1x, It)) + sp.kron(D1y, sp.kron(Ix, It))

        elif reg_type == 'd2xt':
            It = sp.eye(nLags)
            Ix = sp.eye(stim_dims[1])
            Iy = sp.eye(stim_dims[2])

            D1t = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((et, -2 * et, et), axis=0), [-1, 0, 1], nLags, nLags)))

            D1x = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ex, -2 * ex, ex), axis=0), [-1, 0, 1], stim_dims[1], stim_dims[1])))

            D1y = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags(np.concatenate((ey, -2 * ey, ey), axis=0), [-1, 0, 1], stim_dims[2], stim_dims[2])))

            Tmat = sp.kron(D1y, sp.kron(Ix, It)) + sp.kron(Iy, sp.kron(D1x, It)) + sp.kron(Iy, sp.kron(Ix, D1t))

        else:
            print('Unsupported reg type (2):', reg_type)
            Tmat = None

    Tmat = Tmat.toarray()  # make dense matrix before sending home

    return Tmat


def create_maxpenalty_matrix(input_dims, reg_type):
    """
    Usage: Tmat = create_maxpenalty_matrix(input_dims, reg_type)

    Creates a matrix specifying a an L2-regularization operator of the form
    ||T*k||^2, where T is a matrix and k is a vector of parameters. Currently 
    only supports second derivative/Laplacian operations

    Args:
        input_dims (list of ints): dimensions associated with the target input, 
            in the form [num_lags, num_x_pix, num_y_pix]
        reg_type (str): specify form of the regularization matrix
            'max' | 'max_filt' | 'max_space'

    Returns:
        numpy array: matrix specifying the desired Tikhonov operator

    Notes:
        Adapted from create_Tikhonov_matrix function above.
        
    """

    allowed_reg_types = ['max', 'max_filt', 'max_space', 'center']
    # assert (ischar(reg_type) && ismember(reg_type, allowed_reg_types), 'not an allowed regularization type');

    # first dimension is assumed to represent filters
    num_filt = input_dims[0]

    # additional dimensions are spatial (Nx and Ny)
    num_pix = input_dims[1] * input_dims[2]
    dims_prod = num_filt * num_pix

    rmat = np.zeros([dims_prod, dims_prod], dtype=np.float32)
    if reg_type == 'max':
        # Simply subtract the diagonal from all-ones
        rmat = np.ones([dims_prod, dims_prod], dtype=np.float32) - np.eye(dims_prod, dtype=np.float32)

    elif reg_type == 'max_filt':
        ek = np.ones([num_filt, num_filt], dtype=np.float32) - np.eye(num_filt, dtype=np.float32)
        rmat = np.kron(np.eye(num_pix), ek)

    elif reg_type == 'max_space':
        ex = np.ones([num_pix, num_pix]) - np.eye(num_pix)
        rmat = np.kron(ex, np.eye(num_filt, dtype=np.float32))

    elif reg_type == 'center':
        for i in range(dims_prod):
            pos_x = (i % (input_dims[0] * input_dims[1])) // input_dims[0]
            pos_y = i // (input_dims[0] * input_dims[1])

            center_x = (input_dims[1] - 1) / 2
            center_y = (input_dims[2] - 1) / 2

            alpha = np.square(pos_x - center_x) + np.square(pos_y - center_y)

            rmat[i, i] = 0.01*alpha

    else:
        print('Havent made this type of reg yet. What you are getting wont work.')

    return rmat


def create_localpenalty_matrix(input_dims, separable=True, spatial_global=False):
    """
    Usage: Tmat = create_maxpenalty_matrix(input_dims, reg_type)

    Creates a matrix specifying a an L2-regularization operator of the form
    ||T*k||^2, where T is a matrix and k is a vector of parameters. Currently
    only supports second derivative/Laplacian operations

    Args:
        input_dims (list of ints): dimensions associated with the target input,
            in the form [num_lags, num_x_pix, num_y_pix]
        reg_type (str): specify form of the regularization matrix
            'max' | 'max_filt' | 'max_space'

    Returns:
        numpy array: matrix specifying the desired Tikhonov operator

    Notes:
        Adapted from create_Tikhonov_matrix function above.

    """

    # assert (ischar(reg_type) && ismember(reg_type, allowed_reg_types), 'not an allowed regularization type');

    # first dimension is assumed to represent filters
    num_filt = input_dims[0]

    # additional dimensions are spatial (Nx and Ny)
    num_pix = input_dims[1] * input_dims[2]
    mat_seed = np.zeros([num_pix, num_pix], dtype=np.float32)

    for ii in range(num_pix):
        #pos1_x = (ii % (input_dims[0] * input_dims[1])) // input_dims[0]  # for non-separable
        pos1_x = ii % input_dims[1]
        pos1_y = ii // input_dims[1]
        for jj in range(num_pix):
            pos2_x = jj % input_dims[1]
            pos2_y = jj // input_dims[1]

            alpha = np.square(pos1_x - pos2_x) + np.square(pos1_y - pos2_y)

            mat_seed[ii, jj] = alpha / (np.square(input_dims[1]/2)+np.square(input_dims[2]/2))

    if separable:
        rmat = mat_seed
    else:
        #rmat = np.kron(mat_seed, np.eye(num_filt, dtype=np.float32))
        if spatial_global is False:
            #rmat = np.kron(np.eye(num_filt, dtype=np.float32), mat_seed)
            rmat = np.kron(mat_seed, np.eye(num_filt, dtype=np.float32))
        else:
            rmat = np.kron(mat_seed, np.ones([num_filt, num_filt], dtype=np.float32))

    return rmat
