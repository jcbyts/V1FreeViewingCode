import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import V1FreeViewingCode.Analysis.notebooks.gratings as gt

class GratingDataset(Dataset):
    def __init__(self,id,num_lags=1,train=True):
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
        self.NX = opts['NX']
        self.NY = opts['NY']
        self.NC = Robs.shape[1]
        self.valid = self.valid[self.valid > self.num_lags]
        if train:
            self.valid = np.setdiff1d(self.valid, opts['Ti'])
        else:
            self.valid = opts['Ti']
            
        self.n_samples = len(self.valid)
     

    def __getitem__(self, index):
        vind = self.valid[index]
        n = len(vind)
        print(vind)
        lags = np.arange(-self.num_lags,0)+1
        vinds = np.arange(vind-self.num_lags, vind)
        return self.x[vinds].flatten(), self.y[vind]

    def __len__(self):
        return self.n_samples