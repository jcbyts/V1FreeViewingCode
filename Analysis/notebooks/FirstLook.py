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

from copy import deepcopy
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

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



n = 4
# plt.figure()
# yax = np.linspace(0, 15, 1000)
# D = gt.tent_basis_log(yax, n)
# plt.plot(yax,D)
# plt.xlabel('Spatial Frequency')
# plt.ylabel

plt.figure()
xax = np.linspace(0, 180, 100)
D = gt.tent_basis_circ(xax, n)
plt.plot(xax, D)
plt.xlabel('Orientation')
plt.ylabel('Weight')

#%% get basis
m = 6
n = 7
xax = np.linspace(0, 180, 100)
yax = np.linspace(0, 15, 100)
xx = np.meshgrid(xax, yax)

D = gt.polar_basis_tent(xx[0].flatten(),xx[1].flatten(),m,n, endpoints=[.25, 30])

plt.figure(figsize=(10,10))
for i in range(m*(n-1)):
    a=np.reshape(D[:,i], (100,100))
    # plt.contour(xx[0], xx[1],a, levels=[.5,1.0])
    plt.contourf(xx[0], xx[1],a, levels=np.arange(.5, 1.15, .1), cmap='Blues', alpha=.5)
# plt.semilogy()
plt.xlabel('Orientation')
plt.ylabel('Spatial Frequency')

#%% get relevant variables

th = matdat['grating']['ori'].copy() # orientation
sf = matdat['grating']['cpd'].copy() # spatial frequency
ft = matdat['grating']['frameTime'].copy() # frame time
bs = np.median(np.diff(ft)) # bin size

plt.plot(th,sf,'.')
plt.xlabel('orientation')
plt.ylabel('spatial frequency')
#%% bin spike counts / stimulus
NC = len(matdat['spikes']['cids'])
NT = len(ft)
RobsAll = np.zeros((NT,NC))
for i in range(NC):
    cc = matdat['spikes']['cids'][i]
    st = matdat['spikes']['st'][matdat['spikes']['clu']==cc]
    RobsAll[:,i] = U.bin_at_frames(st,ft,10).flatten()

# bin stimulus on basis
NX = m
NY = n
stim = gt.polar_basis_tent(th,sf,NX,NY)
# stim = gt.polar_basis(th,sf,NX,NY)
# stim = gt.unit_basis(th,sf)
# NY = NY - 1

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
stas = Xstim.T @ Robs / np.sum(Robs)
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


#%% plot stas
fig, ax = plt.subplots(nrows=NC, ncols=num_lags, sharex=True, sharey=True, figsize=(8, NC*1))
for cc in range(NC):
    w = stas[:,:,cc]
    wB = D@w
    vmin = np.min(wB)
    vmax = np.max(wB)
    for i in range(num_lags):
        ax[cc,i].imshow(np.reshape(wB[:,i], (100, 100)), vmin=vmin, vmax=vmax, cmap='Blues')
        ax[cc,i].axis("off")
        if cc==0:
            ax[cc,i].set_title('lag %d' %i)

plt.tight_layout()
fig.text(0.5, 0.04, 'Orientation', ha='center')
fig.text(0.04, 0.5, 'Spatial Frequency', va='center', rotation='vertical')
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



#%%
plt.figure()
DU.plot_3dfilters(glm0)

plt.figure()
filters = DU.compute_spatiotemporal_filters(glm0)
basis={'B':D,'nx':100,'ny':100}
gt.plot_3dfilters(filters=filters, basis=basis)

#%% low rank?
F = np.reshape(filters, (np.prod(filters.shape[:3]), filters.shape[3]))
u,s,v = np.linalg.svd(F, full_matrices=False)

gt.plot_3dfilters(filters=u[:,:2], basis=basis)

# %% saccade kernels
num_saclags = 40
back_shifts = 10

slist = matdat['slist']
sac_on = U.bin_at_frames(slist[:,0], ft,maxbsize=0.1)
sac_off = U.bin_at_frames(slist[:,1], ft, maxbsize=0.1)
Xsac1 = NDNutils.create_time_embedding(NDNutils.shift_mat_zpad(sac_on,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
Xsac2 = NDNutils.create_time_embedding(NDNutils.shift_mat_zpad(sac_off,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
TSreg = 0.001




# %%
sacta = Xsac1.T @ Robs
sacta /= np.mean(sacta, axis=0)
xax = np.arange(-back_shifts, num_saclags-back_shifts, 1)
h = plt.plot(xax,sacta)
plt.show

# %% make main models

#%% 

NC = Robs.shape[1]
stim_only = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY, num_lags], layer_sizes=[NC], layer_types=['normal'],
    normalization=[0], act_funcs=['softplus'],
    reg_list={'d2xt':[XTreg], 'l1':[L1reg]})

stim_full = NDNutils.ffnetwork_params(
    input_dims=[1,NX,NY,num_lags], xstim_n=[0],
    layer_sizes=[NC], layer_types=['normal'], normalization=[0],
    act_funcs=['lin']
)
sac_on_full = NDNutils.ffnetwork_params(
    input_dims=[1,1,1,num_saclags], xstim_n=[1],
    layer_sizes=[NC], layer_types=['normal'], normalization=[1],
    act_funcs=['lin']
)

comb_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus']
)

stimfull = NDN.NDN([stim_only], noise_dist='poisson')
sacglmfull = NDN.NDN([stim_full, sac_on_full, comb_par], ffnet_out=2,
    noise_dist='poisson'
)

#%%
_ = stimfull.train(
    input_data=[Xstim], output_data=[Robs],
    train_indxs=Ui, test_indxs=Xi,
    learning_alg='adam', opt_params=adam_params
)

# FFmodS.networks[0].layers[0].weights = deepcopy(FFmod0.networks[0].layers[0].weights)
# v2f0 = FFmodS.fit_variables(layers_to_skip=[[0],[],[0]], fit_biases=False)
# v2f0[2][0]['biases']=True
# v2f = deepcopy(v2f0)
# v2f[0][0]['weights']=True

# #%%
# u,s,v = np.linalg.svd((stimfull.networks[0].layers[0].weights), full_matrices=False)
# F = np.reshape(u, filters.shape)
# gt.plot_3dfilters(filters=F, basis=basis)

#%%
Greg0 = 0.001
Mreg0 = 0.001
L1reg0 = 0.001
XTreg = 0.00001
num_subs = 5
nim_par = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY, num_lags], layer_sizes=[num_subs, NC], 
    layer_types=['normal','normal'], normalization=[1,-1],
    act_funcs=['relu', 'softplus'], verbose=True,
    reg_list={'d2xt':[XTreg], 'l1':[L1reg0], 'glocal':[Greg0], 'max':[None,Mreg0]})

nim_shared = NDN.NDN([nim_par],  noise_dist='poisson')
#%%
_ = nim_shared.train(input_data=[Xstim], output_data=Robs,train_indxs=Ui, test_indxs=Xi,
learning_alg='adam', opt_params=adam_params)
# %% fit

F = DU.compute_spatiotemporal_filters(stimfull)
gt.plot_3dfilters(filters=F, basis=basis)
# DU.plot_3dfilters(stimfull)
#%%
DU.plot_3dfilters(nim_shared)

gt.plot_3dfilters(DU.compute_spatiotemporal_filters(nim_shared), basis=basis)
#%%
FFmodS.networks[0].layers[0].weights = deepcopy(FFmod0.networks[0].layers[0].weights)
sacglmfull.networks[0].layers[0].weights = deep
_ = sacglmfull.train(
    input_data=[Xstim, Xsac1, Xsac2], output_data=Robs, 
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0, 
    learning_alg='adam', opt_params=adam_params)

#%%
LLfull = glm0.eval_models(input_data=[Xstim], output_data=Robs,data_indxs=Xi, nulladjusted=True)
LLshared = nim_shared.eval_models(input_data=[Xstim], output_data=Robs,
data_indxs=Xi, nulladjusted=True)

plt.plot(LLfull, LLshared, '.')
plt.plot(np.array((0,1)), np.array((0,1)))

#%%
sacglmfull.networks 



#%%
#sacnim0.set_regularization('l1', 0.02, ffnet_target = 0, layer_target=1)
#_ = sacnim0.train(
#    input_data=[Xstim, Xsac1[valdata,:], Xsac2[valdata,:]], output_data=Robs[valdata,:], 
#    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f, 
#    learning_alg='adam', opt_params=adam_params, output_dir=output_dir )

LL1u = sacnim0.eval_models(input_data=[Xstim, Xsac1[valdata,:], Xsac2[valdata,:]], 
                           output_data=Robs[valdata,:], data_indxs=Ui, nulladjusted=False)

mnim1_par = NDNutils.ffnetwork_params( 
    input_dims=[1,NY,NX, num_lags], layer_sizes=[num_tkerns, num_Esubs+num_Isubs, NC], 
    layer_types=['conv', 'normal', 'normal'], conv_filter_widths=[1], ei_layers=[None, num_Isubs],
    normalization=[1,1,0], act_funcs=['lin', 'relu', 'lin'],
    reg_list={'d2t':[Treg], 'd2x':[None, Xreg], 'l1':[None, L1reg/2], 'local':[None,LOCreg]})
sac1_par = NDNutils.ffnetwork_params( 
    input_dims=[1,1,1, num_saclags], xstim_n=[1],
    layer_sizes=[1, NC], layer_types=['normal', 'normal'], normalization=[1,0], act_funcs=['lin', 'lin'],
    reg_list={'d2t':[TSreg2]})
stim_par = NDNutils.ffnetwork_params( 
    xstim_n=None, ffnet_n=[0,1],
    layer_sizes=[NC], layer_types=['mult'], normalization=[0], act_funcs=['lin'])
sac2_par = NDNutils.ffnetwork_params( 
    input_dims=[1,1,1, num_saclags], xstim_n=[2],
    layer_sizes=[1, NC], layer_types=['normal', 'normal'], normalization=[1, 0], act_funcs=['lin', 'lin'],
    reg_list={'d2t':[TSreg2]})
comb_par = NDNutils.ffnetwork_params( 
    xstim_n=None, ffnet_n=[2,3], layer_sizes=[NC], layer_types=['add'], act_funcs=['softplus'])

gmodsac0 = NDN.NDN( [mnim1_par, sac1_par, stim_par, sac2_par, comb_par], ffnet_out=4, noise_dist='poisson' )