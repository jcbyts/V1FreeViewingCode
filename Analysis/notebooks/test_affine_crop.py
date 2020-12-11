
#%%

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')
import NDN3.NDNutils as NDNutils
import NDN3.Utils.DanUtils as DU
# %%

def get_crop_indices(dims, outsize=None, translation=np.array((0,0)), theta=0, scale=1.0):
    ''' get_crop_indices
        Returns indices for sampling with an affine transform
        Inputs:
            dims <array> input dimensions
            outsize <array> output dimensions (optional)
            translation <array> x,y translation (0,0) is no shift
            theta <float> rotation angle (in degrees)
        Returns:
            indices <list> len=4 indices for interpolation
            wts <list> len=4 interpolation weights

        Example:

    '''
    sx = outsize[0]
    if len(outsize)==2:
        sy = outsize[1]
    else:
        sy = outsize[0]

    W = dims[0]
    H = dims[1]

    sxsy = np.array([scale*sx.astype("float32")/W, scale*sy.astype("float32")/H])
    S = np.array([[sxsy[0], 0.], [0., sxsy[1]]])
    th = theta/180*np.pi
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    tx = (translation[0]) / ((W)/2.)
    ty = (translation[1]) / ((H)/2.)

    # M = RSH; where R = rotation, S = shearing, H = scaling; Matrix multiplcation does not commute
    M0 = R @ S
    M = np.array([[M0[0,0], M0[0,1], tx], [M0[1,0], M0[1,1], ty]])

    # create normalized 2D grid
    x = np.linspace(-1, 1, sx)
    y = np.linspace(-1, 1, sy)
    x_t, y_t = np.meshgrid(x, y)

    # reshape to (xt, yt, 1) - augments the dimensions by one for translation
    ones = np.ones(np.prod(x_t.shape))
    sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

    # transform the sampling grid i.e. batch multiply
    grids = M @ sampling_grid
    grids = grids.reshape(2, sy, sx)
    grids = np.moveaxis(grids, 0, -1)
    x_s = grids[:, :, 0].squeeze()
    y_s = grids[:, :, 1].squeeze()

    # rescale x and y to [0, W/H]
    x = ((x_s + 1.) * W) * 0.5
    y = ((y_s + 1.) * H) * 0.5

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64)
    y1 = y0 + 1

    # make sure it's inside img range [0, H] or [0, W]
    x0 = np.clip(x0, 0, W-1)
    x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1)
    y1 = np.clip(y1, 0, H-1)

    # calculate weights for combining indices to interpolate
    w = []
    w.append( ((x1-x) * (y1-y)).flatten())
    w.append(((x1-x) * (y-y0)).flatten())
    w.append(((x-x0) * (y1-y)).flatten())
    w.append(((x-x0) * (y-y0)).flatten())

    # indices into 4 directions
    ix = []
    ix.append(y0.flatten()*W + x0.flatten())
    ix.append(y1.flatten()*W + x0.flatten())
    ix.append(y0.flatten()*W + x1.flatten())
    ix.append(y1.flatten()*W + x1.flatten())

    return ix, w



def roi_crop(frame,size,translation,theta=0,sxsy=0):
    # roi_crop implements a crop with bilinear interpolation
    # Inputs:
    #   frame [h,w,c] image
    #   size [1x1] or [1x2] x and y size of output (integer)
    #   translation [1 x 2]  x and y translation (float)
    #   theta (optional) angle
    #   
    #   M  [sx 0 tx; 0 sy ty] crop matrix (affine transform matrix)
    # Output:
    #   out []
    # written by jly 2019

    if len(frame.shape)==3:
        H, W, C = frame.shape
    else:
        H, W = frame.shape
        C = 0
    
    sx = size[0]
    if len(size)==2:
        sy = size[1]
    else:
        sy = size[0]
    
    # build M
    if sxsy==0:
        sxsy = np.array([sx.astype("float32")/W, sy.astype("float32")/H])

    S = np.array([[sxsy[0], 0.], [0., sxsy[1]]])
    th = theta/180*np.pi
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    
    tx = (translation[0] - (W)/2.) / ((W)/2.)
    ty = (translation[1] - (H)/2.) / ((H)/2.)
    
    # M = RSH; where R = rotation, S = shearing, H = scaling; Matrix multiplcation does not commute
    M0 = R @ S
    M = np.array([[M0[0,0], M0[0,1], tx], [M0[1,0], M0[1,1], ty]])
    
    # create normalized 2D grid
    x = np.linspace(-1, 1, sx)
    y = np.linspace(-1, 1, sy)

    x_t, y_t = np.meshgrid(x, y)

    # reshape to (xt, yt, 1) - augments the dimensions by one for translation
    ones = np.ones(np.prod(x_t.shape))
    sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

    # transform the sampling grid i.e. batch multiply
    grids = M @ sampling_grid
    grids = grids.reshape(2, sy, sx)
    grids = np.moveaxis(grids, 0, -1)

    x_s = grids[:, :, 0].squeeze()
    y_s = grids[:, :, 1].squeeze()

    # rescale x and y to [0, W/H]
    x = ((x_s + 1.) * W) * 0.5
    y = ((y_s + 1.) * H) * 0.5

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64)
    y1 = y0 + 1

    # make sure it's inside img range [0, H] or [0, W]
    x0 = np.clip(x0, 0, W-1)
    x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1)
    y1 = np.clip(y1, 0, H-1)
    
    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    # look up pixel values at corner coords
    if C==0:
        Ia = frame[y0, x0]
        Ib = frame[y1, x0]
        Ic = frame[y0, x1]
        Id = frame[y1, x1]
        
        out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    else:
        Ia = frame[y0, x0, :]
        Ib = frame[y1, x0, :]
        Ic = frame[y0, x1, :]
        Id = frame[y1, x1, :]
        out = Ia
        for i in range(C):
            out[:,:,i] = wa*Ia[:,:,i] + wb*Ib[:,:,i] + wc*Ic[:,:,i] + wd*Id[:,:,i]
            
    
    return out
# %%

I = DU.gabor_sized(30, 90)
plt.imshow(I)
plt.title("Default image")

outsize = np.array((20,20))
translation = np.array((0,0))
theta=10

# # %% cropping an image

# for i in np.arange(-10,10,2):
#     plt.figure()
#     Ic = roi_crop(I, np.array( (50,50)), np.array( (30,30)),theta=i)
#     plt.imshow(Ic)
# %% cropping /rotating a vector (flattened image)

I = DU.gabor_sized(30,90)

X = I.flatten() # flattened image
frame = I.copy()
dims = I.shape
outsize=np.array((60,60))
translation=np.array((0,0))
scale = .5
theta=90

ix,w = get_crop_indices(dims, outsize=outsize, translation=translation, theta=theta, scale=scale)

I2 = 0
for i in range(len(w)):
    I2 = I2 + w[i]*X[ix[i]]

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(I)
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(np.reshape(I2, outsize))
plt.title("Cropped/Rotated")

#%%
plt.figure(figsize=(10,4))
for ith in range(nsteps):
    plt.subplot(1,nsteps,ith+1)
    theta=rotate[ith]

    ix,w = ne.get_crop_indices(dims, outsize=outsize, translation=translation, theta=theta, scale=scale)

    I2 = 0
    for i in range(len(w)):
        I2 = I2 + w[i]*X[ix[i]]

        plt.imshow(np.reshape(I2, outsize))
        plt.axis("off")
        plt.title("%02.1f" %rotate[ith])



# %%
