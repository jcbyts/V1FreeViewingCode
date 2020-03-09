#%% set paths
from __future__ import print_function
from __future__ import division

import sys
import Utils as U
sys.path.insert(0, '/home/jcbyts/Repos/')
import NDN3.NDNutils as NDNutils
which_gpu = NDNutils.assign_gpu()
#NDNutils.setup_no_gpu()


import os
import h5py
import numpy as np
import tensorflow as tf
import scipy.io as sio           # importing matlab data
import pandas as pd              # for handling csv meta data
import matplotlib.pyplot as plt  # plotting
from copy import deepcopy

import time
import random
import NDN3.NDN as NDN
#import TNDN as TNDN
import NDN3.Utils.DanUtils as DU
import NDN3.Utils.NDNplot as NDNplot
output_dir = '/home/jcbyts/PyPlay/tensorboard' + str(which_gpu)
print(output_dir)


# %%
datadir = "/home/jcbyts/Data/MitchellV1FreeViewing/grating_subspace/"
meta_file = "/home/jcbyts/Repos/V1FreeViewingCode/Data/datasets.csv"

df = pd.read_csv(meta_file)

# %%
sessionid = 60
fname = df.Tag[sessionid] + "_gratingsubspace.mat"

matdat = sio.loadmat(datadir+fname, squeeze_me=True)
stim = matdat['stim'].copy()
RobsAll = matdat['Robs'].copy()

oris = np.unique(matdat['oris'])
spatfreqs = np.unique(matdat['spatfreq'])
spatfreqs = spatfreqs[1:]
NX = len(oris)
NY = len(spatfreqs)
# %%
plt.figure(figsize=(14,4))
plt.imshow(RobsAll[0:100,:])

# %%
Nspks = np.sum(RobsAll,axis=0)
valcell = np.where(Nspks > 500)[0]
NC = len(valcell)
Robs = RobsAll[:,valcell]
print(NC, 'selected')

NT = Robs.shape[0]


# %% set optimization parmeters
adam_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')

early_stopping = 100

adam_params['batch_size'] = 1000 #NT // 50
adam_params['display'] = 30
adam_params['epochs_training'] = early_stopping * 100
adam_params['run_diagnostics'] = False

adam_params['epsilon'] = 1e-8
adam_params['early_stop'] = early_stopping
adam_params['early_stop_mode'] = 1
#adam_params['data_pipe_type'] = 'iterator'
adam_params['data_pipe_type'] = 'data_as_var'
adam_params['learning_rate'] = 1e-3
#adam_params['epochs_summary'] = 5
for d in adam_params:
    print("%20s:\t %s" %(d, adam_params[d]))

lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_params['maxiter'] = 1000
print('\nLBFGS:')
for d in lbfgs_params:
    print("%20s:\t %s" %(d, lbfgs_params[d]))

adam_loc = adam_params.copy()
#adam_loc['batch_size'] = len(valtot)//200
adam_loc['early_stop'] = 100
adam_locX = adam_loc.copy()
adam_locX['early_stop_mode'] = 2


# %% create time-embedded design matrix
num_lags = 10
Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1 )#[valdata,:]

# %% get STA
# Robs = NDNutils.shift_mat_zpad(RobsAll[:,valcell], -1, dim=0)
# STAs
stas = np.reshape(np.matmul(np.transpose(Xstim),Robs), [NX*NY,num_lags, NC])/NT
np.mean(np.max(abs(stas), axis=0))

# %% plot STA
sx, sy = U.get_subplot_dims(NC)
DU.subplot_setup(sx,sy)
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    plt.plot(np.transpose(stas[:,:,cc]))
    # bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    # plt.imshow(np.reshape(stas[:,bestlag,cc], [NY, NX]), cmap='gray')  #RdBu_r 
               #vmin=-np.max(abs(stas[:,:,cc])), vmax=np.max(abs(stas[:,:,cc])))
    plt.title(str(bestlag)+' cell'+str(cc),)
    plt.title(str(bestlag)+' cell'+str(cc),)
plt.show()

# %%
import importlib
importlib.reload(U)

# %%
