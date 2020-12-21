import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py

def get_stim_list(id):
    stim_list = {
            '20191231': 'logan_20191231_Gabor_-20_-10_50_60_2_2_0_19_0.mat',
            '20200304': 'logan_20200304_-20_-10_50_60_0_19_0_1.hdf5',
            '20200306': 'logan_20200306_Gabor_-20_-10_40_60_2_2_0_9_0.mat'
        }
    return stim_list[id]

class PixelDataset(Dataset):
    """
    PixelDataset is a pytorch Dataset for loading stimulus movies and spikes
    Arguments:
    id:             <string>    id of the session (must exist in get_stim_list, e.g,. '20200304)
    num_lags:       <int>       number of time lags
    stims:          <list>      list of strings corresponding to requested stimuli
                                    "Gabor"     - gabor droplet noise
                                    "Grating"   - full-field gratings
                                    "BackImage" - static natural image 
                                    "FixRsvpStim" - rapidly flashed filtered natural images
    stimset:        <string>    "Train" or "Test" set
    downsample_s    <int>       spatial downsample factor (this slows things down, because smoothing before subsample)
    downsample_t    <int>       temporal downsample factor (this slows things down because of smoothing operation)
    valid_eye_rad   <float>     valid region on screen (centered on 0,0) in degrees of visual angle
    fixations_only  <bool>      whether to only include fixations
    dirname         <string>    full path to where the data are stored
    cids            <list>      list of cells to include (file is loaded in its entirety and then sampled because slicing hdf5 has to be simple)
    cropidx                     index of the form [(x0,x1),(y0,y1)] or None type
    include_eyepos  <bool>      flag to include the eye position info in __get_item__ output

    """
    def __init__(self,id,
        num_lags:int=1,
        stimset="Train",
        stims=["Gabor"],
        downsample_s: int=1,
        downsample_t: int=2,
        smooth_spikes=False,
        valid_eye_rad=5.2,
        fixations_only=True,
        dirname='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/',
        cids=None,
        cropidx=None,
        include_eyepos=False,
        shifter=None,
        preload=False):
        
        # load data
        self.dirname = dirname
        self.id = id
        self.cropidx = cropidx
        self.fname = get_stim_list(id)
        self.sdnorm = 15 # scale stimuli (puts model in better range??)
        self.stimset = stimset
        self.fhandle = h5py.File(self.dirname + self.fname, "r")
        self.isopen = True
        self.num_lags = num_lags
        self.downsample_s = downsample_s
        self.downsample_t = downsample_t
        self.fixations_only = fixations_only
        self.include_eyepos = include_eyepos
        self.valid_eye_rad = valid_eye_rad
        self.shifter=shifter

        # sanity check stimuli (all requested stimuli must be keys in the file)
        newstims = []
        for s in range(len(stims)):
            if stims[s] in self.fhandle.keys():
                newstims.append(stims[s])
        print("Found requested stimuli %s" %newstims)
        self.stims = newstims

        # useful info to pull from meta data
        sz = self.fhandle[self.stims[0]]['Test']['Stim'].attrs['size']
        ppd = self.fhandle[self.stims[0]]['Test']['Stim'].attrs['ppd'][0]
        self.centerpix = self.fhandle[self.stims[0]]['Test']['Stim'].attrs['center'][:]
        self.rect = self.fhandle[self.stims[0]]['Test']['Stim'].attrs['rect'][:]
        self.ppd = ppd
        self.NY = int(sz[0]//self.downsample_s)
        self.NX = int(sz[1]//self.downsample_s)

        # get valid indices
        self.valid = [self.get_valid_indices(stim) for stim in self.stims]
        self.lens = [len(v) for v in self.valid]
        indices = [[i] * v for i, v in enumerate(self.lens)]
        self.stim_indices = np.asarray(sum(indices, []))
        self.indices_hack = np.arange(0,len(self.stim_indices)) # stupid conversion from slice/int/range to numpy array
        
        # setup cropping
        if cropidx:
            self.cropidx = cropidx
            self.NX = cropidx[0][1] - cropidx[0][0]
            self.NY = cropidx[1][1] - cropidx[1][0]
        else:
            self.cropidx = None

        # spike meta data / specify clusters
        self.cluster_ids = self.fhandle[self.stims[0]]['Test']['Robs'].attrs['cids']
        cgs = self.fhandle['Neurons']['cgs'][:][0]
        
        self.NC = len(self.cluster_ids)
        if cids:
            self.cids = cids
            self.NC = len(cids)
            self.cluster_ids = self.cluster_ids[cids]
        else:
            self.cids = list(range(0,self.NC-1))
        
        self.single_unit = [int(cgs[c])==2 for c in self.cids]
        
        if preload: # preload data if it will fit in memory
            self.preload=False
            print("Preload True. Loading ")
            n = len(self)
            self.x = torch.ones((n,self.num_lags,self.NY, self.NX))
            self.y = torch.ones((n,self.NC))
            self.eyepos = torch.ones( (n,2))
            chunk_size = 10000
            nsteps = n//chunk_size+1
            for i in range(nsteps):
                print("%d/%d" %(i+1,nsteps))
                inds = np.arange(i*chunk_size, np.minimum(i*chunk_size + chunk_size, n))
                sample = self.__getitem__(inds)
                self.x[inds,:,:,:] = sample['stim'].detach().clone()
                self.y[inds,:] = sample['robs'].detach().clone()
                self.eyepos[inds,0] = sample['eyepos'][:,0].detach().clone()
                self.eyepos[inds,1] = sample['eyepos'][:,1].detach().clone()
            print("Done")
        self.preload = preload

    def __getitem__(self, index):
        """
            This is a required Dataset method
        """            
        
        if self.preload:
            return {'stim': self.x[index,:,:,:], 'robs': self.y[index,:], 'eyepos': self.eyepos[index,:]}
        else:            
            # index into valid stimulus indices (this is part of handling multiple stimulus sets)
            uinds, uinverse = np.unique(self.stim_indices[index], return_inverse=True)
            indices = self.indices_hack[index] # this is now a numpy array

            # loop over stimuli included in this index
            for ss in range(len(uinds)):
                stim_start = np.where(self.stim_indices==uinds[ss])[0][0]
                # stim_inds = np.where(uinverse==ss)[0] - stim_start
                stim_inds = indices - stim_start
                stim_inds = stim_inds[uinverse==ss]
                valid_inds = self.valid[uinds[ss]][stim_inds]
                file_inds = np.expand_dims(valid_inds, axis=1) - range(0,self.num_lags*self.downsample_t)

                ufinds, ufinverse = np.unique(file_inds.flatten(), return_inverse=True)
                if self.cropidx and not self.shifter:
                    I = self.fhandle[self.stims[ss]][self.stimset]["Stim"][self.cropidx[1][0]:self.cropidx[1][1],self.cropidx[0][0]:self.cropidx[0][1],ufinds]
                else:
                    I = self.fhandle[self.stims[ss]][self.stimset]["Stim"][:,:,ufinds]

                if self.shifter:
                    eyepos = self.fhandle[self.stims[ss]][self.stimset]["eyeAtFrame"][1:3,ufinds].T
                    eyepos[:,0] -= self.centerpix[0]
                    eyepos[:,1] -= self.centerpix[1]
                    eyepos/= self.ppd
                    I = self.shift_stim(I, eyepos)
                    if self.cropidx:
                        I = I[self.cropidx[1][0]:self.cropidx[1][1],self.cropidx[0][0]:self.cropidx[0][1],:]

                R = self.fhandle[self.stims[ss]][self.stimset]["Robs"][:,valid_inds]
                if self.include_eyepos:
                    eyepos = self.fhandle[self.stims[ss]][self.stimset]["eyeAtFrame"][1:3,valid_inds].T
                    eyepos[:,0] -= self.centerpix[0]
                    eyepos[:,1] -= self.centerpix[1]
                    eyepos/= self.ppd

                sz = I.shape
                I = I[:,:,ufinverse].reshape(sz[0],sz[1],-1, self.num_lags*self.downsample_t).transpose((2,3,0,1))
                R = R.T
                if self.NC != R.shape[1]:
                    R = R[:,np.asarray(self.cids)]

                # concatentate if necessary
                if ss ==0:
                    S = torch.tensor(self.transform_stim(I))
                    Robs = torch.tensor(R.astype('float32'))
                    if self.include_eyepos:
                        ep = torch.tensor(eyepos.astype('float32'))
                    else:
                        ep = None
                else:
                    S = torch.cat( (S, self.transform_stim(I)), dim=0)
                    Robs = torch.cat( (Robs, torch.tensor(R.astype('float32'))), dim=0)
                    if self.include_eyepos:
                        ep = torch.cat( (ep, torch.tensor(eyepos.astype('float32'))), dim=0)

            return {'stim': S, 'robs': Robs, 'eyepos': ep}

    def __len__(self):
        return sum(self.lens)

    def get_valid_indices(self, stim):
        # get blocks (start, stop) of valid samples
        blocks = self.fhandle[stim][self.stimset]['blocks'][:,:]
        valid = []
        for bb in range(blocks.shape[1]):
            valid.append(np.arange(blocks[0,bb]+self.num_lags*self.downsample_t,
                blocks[1,bb])) # offset start by num_lags
        
        valid = np.concatenate(valid).astype(int)

        if self.fixations_only:
            fixations = np.where(self.fhandle[stim][self.stimset]['labels'][:]==1)[1]
            valid = np.intersect1d(valid, fixations)
        
        if self.valid_eye_rad:
            xy = self.fhandle[stim][self.stimset]['eyeAtFrame'][1:3,:].T
            xy[:,0] -= self.centerpix[0]
            xy[:,1] = self.centerpix[1] - xy[:,1] # y pixels run down (flip when converting to degrees)
            eyeCentered = np.hypot(xy[:,0],xy[:,1])/self.ppd < self.valid_eye_rad
            valid = np.intersect1d(valid, np.where(eyeCentered)[0])

        return valid

    def transform_stim(self, s):
        # stim comes in N,Lags,Y,X
        s = s.astype('float32')/self.sdnorm

        if self.downsample_t>1 or self.downsample_s>1:
            from scipy.ndimage import gaussian_filter
            sig = [0, self.downsample_t-1, self.downsample_s-1, self.downsample_s-1] # smoothing before downsample
            s = gaussian_filter(s, sig)
            s = s[:,::self.downsample_t,::self.downsample_s,::self.downsample_s]

        if s.shape[0]==1:
            s=s[0,:,:,:] # return single item

        return s

    def shift_stim(self, im, eyepos):
        """
        apply shifter to translate stimulus as a function of the eye position
        """
        import torch.nn.functional as F
        import torch
        affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        sz = im.shape
        eyepos = torch.tensor(eyepos.astype('float32'))
        im = torch.tensor(im[:,None,:,:].astype('float32'))
        im = im.permute((3,1,0,2))

        shift = self.shifter(eyepos).detach()
        aff = torch.tensor([[1,0,0],[0,1,0]])

        affine_trans = shift[:,:,None]+aff[None,:,:]
        affine_trans[:,0,0] = 1
        affine_trans[:,0,1] = 0
        affine_trans[:,1,0] = 0
        affine_trans[:,1,1] = 1

        n = im.shape[0]
        grid = F.affine_grid(affine_trans, torch.Size((n, 1, sz[0], sz[1])), align_corners=True)

        im2 = F.grid_sample(im, grid)
        im2 = im2[:,0,:,:].permute((1,2,0)).detach().cpu().numpy()

        return im2

#% testing code
# gd = PixelDataset('20200304', stims=["Gabor"],
#     stimset="Train",
#     cropidx=None,
#     num_lags=10,
#     downsample_t=1,
#     downsample_s=1,
#     include_eyepos=True)


# #%%
# # 
# #     

# # self.fhandle[self.stim][self.stimset]['Stim']
# #%%
# from scipy.ndimage import gaussian_filter
# import numpy as np
# import matplotlib.pyplot as plt

# sample = gd[:1000]
# print(sample['stim'].shape)
# sample['robs'].shape
# # np.where(np.asarray(gd.single_unit))

# # from sys import getsizeof
# # getsizeof(sample['stim'])

# a = sample['eyepos']


# #%%

# f = plt.plot(a.detach().cpu().numpy())
# # plt.imshow( sample['stim'][1,0,:,:])

# #%%

# stas = torch.einsum('nlwh,nc->lwhc', sample['stim'], sample['robs']-sample['robs'].mean(dim=0))
# sta = stas.detach().cpu().numpy()
# cc = 0
# #%%
# NC = sta.shape[3]
# if cc >= NC:
#     cc = 0
# # cc = 7
# print(cc)
# w = sta[:,:,:,cc]
# w = (w - np.min(w)) / (np.max(w)-np.min(w))
# # w = w[::2,:,:]
# plt.figure(figsize=(10,3))
# for i in range(w.shape[0]):
#     plt.subplot(1,w.shape[0],i+1)
#     plt.imshow(w[i,:,:], vmin=0, vmax=1, interpolation=None)
#     plt.axis("off")
# cc +=1


# get_stim_list('20200304')

# #%%
# import matplotlib.pyplot as plt

# sample = gd[:10]

# plt.imshow(sample['stim'][:,:,2])
# #%%
# I = sample['stim']
# n = I.shape[2]
# n = 1000
# inds = np.arange(0,n-1)

# vind = np.arange(10,20)
# lags = np.expand_dims(vind, axis=1) - range(10)
# ulags, uinverse = np.unique(lags, return_inverse=True)
# # s = gd.fhandle["Gabor"]["Test"]["Stim"][:,:,lags]
# # s.shape
# ulags[uinverse].reshape(-1, 10)

# uinds, uinverse = np.unique(gd.stim_indices[inds], return_inverse=True)

# for ss in range(len(uinds)):
#     stim_start = np.where(gd.stim_indices==uinds[ss])[0][0]
#     stim_inds = np.where(uinverse==ss)[0] - stim_start

#     file_inds = np.expand_dims(gd.valid[uinds[ss]][stim_inds], axis=1) - range(gd.num_lags)

#     ufinds, ufinverse = np.unique(file_inds.flatten(), return_inverse=True)
#     I = gd.fhandle[gd.stims[ss]][gd.stimset]["Stim"][:,:,ufinds]

#     sz = I.shape
#     I = I[:,:,ufinverse].reshape(sz[0],sz[1],-1, gd.num_lags).transpose((2,3,0,1))

#     # transform
#     if ss ==0:
#         S = torch.tensor(I.astype('float32')/gd.sdnorm)
#     else:
#         S = torch.cat( (S, torch.tensor(I.astype('float32')/gd.sdnorm)), dim=0)

# print(S.shape)

# #%%

# plt.imshow(I[100,:,:,45])


# #%%
# num_lags = 10

# valid = np.where(gd.fhandle['Gabor']['Test']['valinds'][:].flatten())[0]
# plt.plot(valid, '.')



# #%%
# def crop_indx( Loriginal, xrange, yrange):
#     # brain-dead way to crop things with space indexed by one dim
#     # Note I'm calling x the horizontal dimension (as plotted by python and y the vertical direction)
#     # Also assuming everything square
#     indxs = []
#     for nn in range(len(yrange)):
#         indxs = np.concatenate((indxs, np.add(xrange,yrange[nn]*Loriginal)))
#     return indxs.astype('int')