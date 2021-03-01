import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import V1FreeViewingCode.Analysis.notebooks.gratings as gt
import V1FreeViewingCode.Analysis.notebooks.neureye as ne

class GratingDataset(Dataset):
    def __init__(self,id,num_lags=1,train=True,augment=0):
        '''
        Grating dataset dataloader
        pass in dataset ID and number of lags to time-embed (0 = no time embedding)

        '''
        # load data
        stim, _, _, Robs, _, basis, opts, sacbc, valid, eyepos = gt.load_and_setup([id])
        self.x = torch.from_numpy(stim.astype('float32'))
        self.y = torch.from_numpy(Robs.astype('float32'))
        self.valid = np.where(valid)[0]
        self.basis = basis
        self.opts = opts
        self.num_lags = num_lags
        self.NX = int(opts['NX'])
        self.NY = int(opts['NY'])
        self.NC = Robs.shape[1]
        self.augment = augment
        self.valid = self.valid[self.valid > self.num_lags]
        if train:
            self.valid = np.setdiff1d(self.valid, opts['Ti'])
        else:
            self.valid = opts['Ti']

        self.n_samples_true = len(self.valid)
        
        if self.augment==0:
            self.n_samples = self.n_samples_true
        else:
            self.n_samples = self.n_samples_true*5
        
     

    def __getitem__(self, index):
        if self.augment>0:
            index = index % self.n_samples_true

        vind = self.valid[index]
        
        if type(vind)==np.int64:
            lags = vind - range(self.num_lags)
        else:
            lags = np.expand_dims(vind, axis=1) - range(self.num_lags)
        
        # .permute((0,2,1)).reshape((-1,self.num_lags*self.NX*self.NY))

        # flatten and permute
        x = self.x[lags,:].reshape((-1, self.num_lags, self.NX, self.NY))
        if self.augment>0:
            x += torch.randn(x.size()) * .01 + 0

        return x, self.y[vind]

    def __len__(self):
        return self.n_samples

def get_stim_list():
    stim_list = {
            '20201231':
                ['logan_20191231_Gabor_-20_-10_50_60_2_2_0_19_0.mat',
                'logan_20191231_Grating_-20_-10_50_60_2_2_0_19_0.mat',
                'logan_20191231_FixRsvpStim_-20_-10_50_60_2_2_0_19_0.mat',
                'logan_20191231_BackImage_-20_-10_50_60_2_2_0_19_0.mat'],
            '20200304':
                ['logan_20200304_Gabor_-20_-10_50_60_2_2_0_19_0.mat',
                'logan_20200304_Grating_-20_-10_50_60_2_2_0_19_0.mat',
                'logan_20200304_FixRsvpStim_-20_-10_50_60_2_2_0_19_0.mat',
                'logan_20200304_BackImage_-20_-10_50_60_2_2_0_19_0.mat'],
                # ['logan_20200304_Gabor_-20_-10_40_60_2_2_0_9_0.mat',
                # 'logan_20200304_Grating_-20_-10_40_60_2_2_0_9_0.mat',
                # 'logan_20200304_BackImage_-20_-10_40_60_2_2_0_9_0.mat',
                # 'logan_20200304_FixRsvpStim_-20_-10_40_60_2_2_0_9_0.mat'],
            '20200304b':
                ['logan_20200304_Gabor_-20_-10_50_60_2_2_0_19_0_1.mat',
                'logan_20200304_Grating_-20_-10_50_60_2_2_0_19_0_1.mat',
                'logan_20200304_FixRsvpStim_-20_-10_50_60_2_2_0_19_0_1.mat',
                'logan_20200304_BackImage_-20_-10_50_60_2_2_0_19_0_1.mat'],
            '20200306':
                ['logan_20200306_Gabor_-20_-10_40_60_2_2_0_9_0.mat',
                'logan_20200306_Grating_-20_-10_40_60_2_2_0_9_0.mat',
                'logan_20200306_FixRsvpStim_-20_-10_40_60_2_2_0_9_0.mat',
                'logan_20200306_BackImage_-20_-10_40_60_2_2_0_9_0.mat']
        }
    return stim_list
class PixelDataset(Dataset):
    def __init__(self,id,
        num_lags=1,
        train=True,
        augment=None,
        stims=None,
        smooth_spikes=False,
        corrected=False,
        valid_eye_rad=5.2,
        fixations_only=True,
        dirname='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/',
        cids=None,
        cropidx=None,
        include_eyepos=False,
        full_dataset=False):
        '''
        Grating dataset dataloader
        pass in dataset ID and number of lags to time-embed (0 = no time embedding)

        '''
        # load data
        self.dirname = dirname
        self.id = id
        self.corrected = corrected
        self.cropidx = cropidx
        stim_list = get_stim_list()

        if stims=='NatImg':
            flist = stim_list[id][4] # only take gabor and grating file
        elif stims=='FixRsvp':
            flist = stim_list[id][2]
        elif stims=='GaborNatImg':
            flist = [stim_list[id][i] for i in (0,1,3)]
        else:
            flist = stim_list[id][:2] # only take gabor and grating file
        
        # Load Data
        Stim,Robs,valdat,labels,NX,NY,dt,eyeAtFrame,frameTime = ne.load_stim_files(dirname=dirname, flist=flist, corrected=False)

        # print(np.nanstd(Stim))
        Stim /= 15 # normalize stimulus by global standard deviation (across all conditions)

        # get valid indices
        if fixations_only:
            valdata = np.intersect1d(np.where(valdat[:,0] == 1)[0], np.where(labels[:,0] == 1)[0]) # fixations / valid
        else:
            valdata = np.where(valdat[:,0] == 1)[0]

         # degrees -- use this when looking at eye-calibration (see below)
        ppd = 37.50476617061

        eyeX = (eyeAtFrame[:,0]-640)/ppd
        eyeY = (eyeAtFrame[:,1]-380)/ppd

        eyeCentered = np.hypot(eyeX, eyeY) < valid_eye_rad
        # eyeCentered = np.logical_and(eyeX < 0, eyeCentered)
        valdata = np.intersect1d(valdata, np.where(eyeCentered)[0])

        if full_dataset: # no augmenting when loading a full dataset
            augment = None
            train = False

        augment = augment_sanity(augment)

        if cids is not None:
            self.cids = cids
            Robs = Robs[:,cids]
        else:
            self.cids = np.arange(0,Robs.shape[1])

        self.num_lags = num_lags
        self.NX = int(NX)
        self.NY = int(NY)
        self.NC = Robs.shape[1]
        self.augment = augment
        self.valid = valdata

        if train and smooth_spikes:
            from scipy.ndimage import gaussian_filter1d    
            Robs = gaussian_filter1d(Robs, 1, axis=0)

        self.valid = self.valid[self.valid > self.num_lags]

        self.include_eye = include_eyepos
        self.eyeX = eyeX
        self.eyeY = eyeY
        self.labels = labels
        self.frameTime = frameTime
        self.ppd = ppd

        # if eye correction requested
        if self.corrected:
            import scipy.io as sio
            from pathlib import Path 
            fpath = Path(self.dirname)
            fname = fpath / (self.id + "_CorrGrid.mat")
            if fname.exists():
                matdat = sio.loadmat(str(fname.resolve()))
                from copy import deepcopy
                # do correction
                centers = matdat['centers']
                locs = matdat['locs'].flatten()

                # get correction
                xcorrec,ycorrec = ne.get_shifter_from_centers(centers, locs*self.ppd, maxshift=2, nearest=False)

                xshift = xcorrec(self.eyeX*self.ppd, self.eyeY*self.ppd)
                yshift = ycorrec(self.eyeX*self.ppd, self.eyeY*self.ppd)

                ix = np.logical_or(np.isnan(xshift), np.isnan(yshift))

                print("Found %d/%d nan indices" %(np.sum(ix), len(ix)))

                xshift[ix] = 0.0
                yshift[ix] = 0.0

                # NEAREST option shifts shifts in integer numbers of pixels
                Stim = ne.shift_stim(Stim, xshift, yshift, [self.NX,self.NY], nearest=False)
                
            else:
                print("no correction grid. loading default")
                self.corrected = False

        # crop stimulus to smaller region?
        if not self.cropidx is None:
            xinds = cropidx[0]
            yinds = cropidx[1]
            NX = xinds[-1]-xinds[0]+1
            NY = yinds[-1]-yinds[0]+1
            cidx = crop_indx(self.NX, xinds, yinds)
            self.x = torch.from_numpy(Stim[:,cidx].astype('float32'))
            self.NX = NX
            self.NY = NY
        else:    
            self.x = torch.from_numpy(Stim.astype('float32'))

        self.y = torch.from_numpy(Robs.astype('float32'))
        self.dt = dt
        
        # test inds
        NT = len(self.valid)
        if full_dataset:
            test_inds = np.arange(0,NT)
        else:
            Ntest = NT//10 # 1/10th of the data used for testing
            np.random.seed(seed=665) # same seed each time
            test_inds = np.random.choice(self.valid, Ntest, replace=False)

        if train:
            self.valid = np.setdiff1d(self.valid, test_inds)
        else:
            self.valid = test_inds

        self.n_samples_true = len(self.valid)
        
        if self.augment is None:
            self.n_samples = self.n_samples_true
        else:
            self.n_samples = self.n_samples_true*5

    def __getitem__(self, index):
        # TODO: handle slice case

        if self.augment is not None: # TODO: use transforms (pytorch built-in)
            index = index % self.n_samples_true

        vind = self.valid[index]
        
        if type(vind)==np.int64:
            issingular = True
            lags = vind - range(self.num_lags)
        else:
            issingular = False
            lags = np.expand_dims(vind, axis=1) - range(self.num_lags)
        

        # flatten and permute
        if issingular:
            x = self.x[lags,:].reshape((self.num_lags, self.NY, self.NX))
        else:
            x = self.x[lags,:].reshape((-1, self.num_lags, self.NY, self.NX))

        # augmentation step
        if self.augment is not None:
            x = self.augment_batch(x)

        sample = {'stim': x, 'robs': self.y[vind], 'eyepos':None}

        if self.include_eye:
            ex = torch.tensor(self.eyeX[vind,None].astype('float32'))
            ey = torch.tensor(self.eyeY[vind,None].astype('float32'))

            if issingular:
                sample['eyepos'] = torch.cat( (ex,ey))
            else:
                sample['eyepos'] = torch.cat( (ex,ey), dim=1)

        return sample

    def __len__(self):
        return self.n_samples

    def augment_batch(self, x):
        sz = x.size()
        sz = list(sz)
        n_samples = x.shape[0]
        inds = np.arange(0, n_samples)
        n_aug = len(self.augment)

        for aa in range(n_aug):
            if self.augment[aa]['proportion']==1:
                ainds = inds
                n = n_samples
            else:
                n = int(self.augment[aa]['proportion']*n_samples)
                ainds = np.random.choice(inds, n, replace=False)
            sz[0] = int(n)

            if self.augment[aa]['type'] == 'gaussian':
                noise = torch.randn(sz)*self.augment[aa]['scale']
                x[ainds,:] += noise
            elif self.augment[aa]['type'] == 'dropout':
                noise = torch.rand(sz)>self.augment[aa]['scale']
                x[ainds,:] *= noise

        return x

def augment_sanity(augment):

        if augment is None:
            return augment

        gaussian = {'scale': 1,
            'proportion': .5}
        dropout = {'scale': .1,
            'proportion': .5}

        if type(augment) is not list:
                augment = [augment]

        for nn in range(len(augment)):
            if augment[nn]['type']=='gaussian':
                for a in gaussian.keys():
                    augment[nn].setdefault(a, gaussian[a])
            elif augment[nn]['type']=='dropout':
                for a in dropout.keys():
                    augment[nn].setdefault(a, dropout[a])
            else:
                print("Augmentation type not supported. You will encounter errors")                                   


        return augment            


def crop_indx( Loriginal, xrange, yrange):
    # brain-dead way to crop things with space indexed by one dim
    # Note I'm calling x the horizontal dimension (as plotted by python and y the vertical direction)
    # Also assuming everything square
    indxs = []
    for nn in range(len(yrange)):
        indxs = np.concatenate((indxs, np.add(xrange,yrange[nn]*Loriginal)))
    return indxs.astype('int')