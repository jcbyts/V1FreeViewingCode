#%% set paths
from __future__ import print_function
from __future__ import division

import sys
import Utils as U
import gratings as gt
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
import importlib # for reimporting modules
output_dir = '/home/jcbyts/PyPlay/tensorboard' + str(which_gpu)
print(output_dir)

# %% load session
sessionid = 1
matdat = gt.load_data(sessionid)

#%% plot basis
m = 4
n = 5
xax = np.linspace(0, 360, 100)
yax = np.linspace(0, 15, 100)
xx = np.meshgrid(xax, yax)

D = gt.polar_basis(xx[0].flatten(),xx[1].flatten(),m,n, endpoints=[.5, 15])
# D = B*C

plt.figure(figsize=(10,10))
for i in range(m*(n-1)):
    plt.subplot(m,n,i+1)
    plt.imshow(np.reshape(D[:,i], (100,100)), extent=[0,360,15,0], aspect='auto')



#%%
for i in range(m*(n-1)):
    a=np.reshape(D[:,i], (100,100))
    # a = a/np.max(a)
    plt.contour(xx[0], xx[1],a, levels=[.5, 1.0])
# plt.semilogy()

#%%
plt.contourf(xx[0], xx[1], np.reshape(D[:,1], (100, 100)), 20, cmap='RdGy')
plt.colorbar()
#%%

th = matdat['grating']['ori'].copy()
sf = matdat['grating']['cpd'].copy()
ft = matdat['grating']['frameTime'].copy()
bs = np.median(np.diff(ft))

plt.plot(th,sf,'.')
#%%
NC = len(matdat['spikes']['cids'])
NT = len(ft)
RobsAll = np.zeros((NT,NC))
for i in range(NC):
    cc = matdat['spikes']['cids'][i]
    st = matdat['spikes']['st'][matdat['spikes']['clu']==cc]
    RobsAll[:,i] = U.bin_at_frames(st,ft,10).flatten()

# bin stimulus on basis
NX = 4
NY = 5
stim = gt.polar_basis(th,sf,NX,NY)
# stim = gt.unit_basis(th,sf)
NY = NY - 1

# zscore
# mu = np.mean(stim)
# sd = np.std(stim)
# stim = (stim - mu)/sd
# %%
plt.figure(figsize=(8,4))
plt.imshow(RobsAll[0:100,:], aspect='auto')

plt.figure(figsize=(8,4))
plt.imshow(stim[0:100,:], aspect='auto')

# %% find spiking neurons
Nspks = np.sum(RobsAll,axis=0)
valcell = np.where(Nspks > 500)[0]
NC = len(valcell)
Robs = RobsAll[:,valcell]
print(NC, 'selected')

NT = Robs.shape[0]
print("Found %d/%d units that had > 500 spikes" %(NC, RobsAll.shape[1]))

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
stas = Xstim.T @ Robs
stas = stas.reshape([NX*NY,num_lags,NC])
# stas = np.reshape(np.matmul(np.transpose(Xstim),Robs), [NX*NY,num_lags, NC])/NT
np.mean(np.max(abs(stas), axis=0))

# %% plot STA
import seaborn as sns
sx, sy = U.get_subplot_dims(NC)
plt.figure(figsize=(8.5,10))
# fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
# fig.set_size_inches(16, row_height*num_rows)
for cc in range(NC):
    ax = plt.subplot(sx,sy,cc+1)
    plt.plot(np.transpose(stas[:,:,cc]))
    plt.axis("off")
    # bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    # plt.imshow(np.reshape(stas[:,bestlag,cc], [NY, NX]), cmap='gray')  #RdBu_r 
               #vmin=-np.max(abs(stas[:,:,cc])), vmax=np.max(abs(stas[:,:,cc])))
    # plt.title(str(bestlag)+' cell'+str(cc),)
    plt.title('cell'+str(cc),)
plt.show()


#%%
cc = 4
plt.imshow(stas[:,:,cc], aspect='auto')

w = stas[:,:,cc]
wB = D@w


plt.figure(figsize=(10,4))
for i in range(num_lags):
    plt.subplot(1,num_lags,i+1)
    plt.imshow(np.reshape(wB[:,i], (100, 100)))
# %% shared GLM
XTreg = None
L1reg = None
valdata = np.arange(0,NT,1)

Ui, Xi = NDNutils.generate_xv_folds(NT, num_blocks=2)

glm_par = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY, num_lags], layer_sizes=[NC], layer_types=['normal'],
    normalization=[1], act_funcs=['softplus'],
    reg_list={'d2xt':[XTreg], 'l1':[L1reg]})

glm0 = NDN.NDN( glm_par, noise_dist='poisson' )
    
_ = glm0.train(input_data=Xstim[valdata,:], output_data=Robs[valdata,:], train_indxs=Ui, test_indxs=Xi,
                   learning_alg='adam', opt_params=adam_params )

LLs0 = glm0.eval_models(input_data=Xstim[valdata,:], output_data=Robs[valdata,:],
                        data_indxs=Xi, nulladjusted=False)
print(np.mean(LLs0))

LLs0n = glm0.eval_models(input_data=Xstim[valdata,:], output_data=Robs[valdata,:],
                        data_indxs=Xi, nulladjusted=True)



# %%
DU.plot_3dfilters(glm0)
# importlib.reload(U)
# 
#%%
glm0.
# %% saccade kernels
num_saclags = 40
back_shifts = 10

sac_on = U.bin_at_frames(slist[:,0], matdat['frameTimes'],maxbsize=0.1)
sac_off = U.bin_at_frames(slist[:,1], matdat['frameTimes'], maxbsize=0.1)
Xsac1 = NDNutils.create_time_embedding(NDNutils.shift_mat_zpad(sac_on,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
Xsac2 = NDNutils.create_time_embedding(NDNutils.shift_mat_zpad(sac_off,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
TSreg = 0.001




# %%
sacta = Xsac2.T @ Robs
h = plt.plot(sacta)
plt.show

# %%
