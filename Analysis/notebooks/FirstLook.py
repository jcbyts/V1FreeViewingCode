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
from scipy.linalg import block_diag
import pandas as pd              # for handling csv meta data
import matplotlib.pyplot as plt  # plotting
from copy import deepcopy

import time
import random
import NDN3.NDN as NDN
#import TNDN as TNDN
import NDN3.Utils.DanUtils as DU
import NDN3.Utils.HadiUtils as HU
import NDN3.Utils.NDNplot as NDNplot
import importlib # for reimporting modules
output_dir = '/home/jcbyts/PyPlay/tensorboard' + str(which_gpu)
print(output_dir)

# %% load raw data
# sessionid = 59
# matdat = gt.load_data(sessionid)

#%% load sessions
sesslist = gt.list_sessions()
sesslist = list(sesslist)

indexlist = [1,2]
sess = [sesslist[i] for i in indexlist]

# shared basis
basis = {'name': 'tent', 'nori': 7, 'nsf': 6, 'endpoints': [0.25, 15]}

stim, sacon, sacoff, Robs, DF, basis, opts = gt.load_sessions(sess, basis=basis) 

#%% plot basis functions
gt.plot_basis(basis)

# matdat = gt.load_data(sesslist[-1])
# plt.figure(figsize=(5,5))
# plt.plot(matdat['grating']['ori'], matdat['grating']['cpd'], '.')


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
# for d in adam_params:
#     print("%20s:\t %s" %(d, adam_params[d]))

lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_params['maxiter'] = 1000
# print('\nLBFGS:')
# for d in lbfgs_params:
#     print("%20s:\t %s" %(d, lbfgs_params[d]))

adam_loc = adam_params.copy()
#adam_loc['batch_size'] = len(valtot)//200
adam_loc['early_stop'] = 100
adam_locX = adam_loc.copy()
adam_locX['early_stop_mode'] = 2


# %% create time-embedded design matrix
num_saclags = 60
back_shifts = 20
num_lags = 15
NX = basis['nori']
NY = basis['nsf']
NC = Robs.shape[1]
NT = Robs.shape[0]

Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1 )
XsacOn = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacon,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
XsacOff = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacoff,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
XsacOnCausal = NDNutils.create_time_embedding( sacon, [num_saclags, 1, 1], tent_spacing=1)
XsacOffCausal = NDNutils.create_time_embedding( sacoff, [num_saclags, 1, 1], tent_spacing=1)



# %% get STA
# Robs = NDNutils.shift_mat_zpad(RobsAll[:,valcell], -1, dim=0)
# STAs
stas = Xstim.T @ Robs / np.sum(Robs)
stas = stas.reshape([NX*NY,num_lags,NC])

# %% plot STA
import seaborn as sns
sx, sy = U.get_subplot_dims(NC)
plt.figure(figsize=(8.5,10))
# fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
# fig.set_size_inches(16, row_height*num_rows)
for cc in range(NC):
    ax = plt.subplot(sx,sy,cc+1)
    plt.plot(np.transpose(stas[:,:,cc]))
    # plt.axis("off")
    # bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    # plt.imshow(np.reshape(stas[:,bestlag,cc], [NY, NX]), cmap='gray')  #RdBu_r 
               #vmin=-np.max(abs(stas[:,:,cc])), vmax=np.max(abs(stas[:,:,cc])))
    # plt.title(str(bestlag)+' cell'+str(cc),)
    plt.title('cell'+str(cc),)
sns.despine()
plt.show()

#%% plot spatial stas at all lags
fig, ax = plt.subplots(nrows=NC, ncols=num_lags, sharex=True, sharey=True, figsize=(8, NC*1))
for cc in range(NC):
    w = stas[:,:,cc]
    wB = basis['B']@w
    vmin = np.min(wB)
    vmax = np.max(wB)
    for i in range(num_lags):
        ax[cc,i].imshow(np.reshape(wB[:,i], (basis['support'], basis['support'])), vmin=vmin, vmax=vmax, cmap='Blues')
        ax[cc,i].axis("off")
        if cc==0:
            ax[cc,i].set_title('lag %d' %i)

plt.tight_layout()
fig.text(0.5, 0.04, 'Orientation', ha='center')
fig.text(0.04, 0.5, 'Spatial Frequency', va='center', rotation='vertical')

# %% setup training indices
valdata = np.arange(0,NT,1)

Ui, Xi = NDNutils.generate_xv_folds(NT, num_blocks=2)

#%% shared NIM
Greg0 = None
Mreg0 = None
L1reg0 = None
XTreg = 0.001
num_subs = 10
num_tkerns = 4

par = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY], layer_sizes=[num_tkerns, num_subs, NC], time_expand=[num_lags],
    layer_types=['temporal', 'normal', 'normal'], normalization=[1, 1, 0],
    act_funcs=['lin', 'lin', 'softplus'], verbose=True, conv_filter_widths=[num_lags],
    reg_list={'orth': [1e-1, 1e-1],'d2t':[0.01],'d2x':[None, XTreg], 'l1':[None, None, L1reg0], 'glocal':[None, Greg0], 'max':[None, None,Mreg0]})

sacpar = NDNutils.ffnetwork_params( 
    input_dims=[1,1,1], layer_sizes=[2, NC], xstim_n=[1], time_expand=[num_saclags, 3],
    layer_types=['temporal', 'normal'], normalization=[1, 1], conv_filter_widths=[num_saclags],
    act_funcs=['lin', 'lin'], verbose=True,
    reg_list={'orth': [1e-1], 'd2t':[0.1], 'l1':[None, L1reg0]})

# sacpar = NDNutils.ffnetwork_params( 
#     input_dims=[1,1,1,num_saclags], layer_sizes=[2, NC], xstim_n=[1],
#     layer_types=['normal', 'normal'], normalization=[1, 0],
#     act_funcs=['lin', 'lin'], verbose=True,
#     reg_list={'orth': [1e-1], 'd2t':[0.1], 'l1':[None, L1reg0]})

combpar = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus']
)


nim_shared = NDN.NDN([par],  noise_dist='poisson')
par2 = deepcopy(par)
par2['activation_funcs'][-1] = 'lin'
nim_shared_sac = NDN.NDN([par2, sacpar, combpar],  noise_dist='poisson', ffnet_out=2)

v2f0 = nim_shared.fit_variables(fit_biases=False)
v2f0[-1][-1]['biases']=True

#%% train
sacuse = NDNutils.shift_mat_zpad(sacon,-back_shifts,dim=0)
nim_shared.time_spread = 75
_ = nim_shared.train(input_data=[stim], output_data=Robs,train_indxs=Ui, test_indxs=Xi, data_filters=DF,
learning_alg='adam', opt_params=adam_params, fit_variables=v2f0)

#%%
nim_shared_sac.networks[0].layers[0].weights = deepcopy(nim_shared.networks[0].layers[0].weights)
nim_shared_sac.networks[0].layers[1].weights = deepcopy(nim_shared.networks[0].layers[1].weights)
nim_shared_sac.networks[-1].layers[-1].biases = deepcopy(nim_shared.networks[-1].layers[-1].biases)

v2f1 = nim_shared_sac.fit_variables(layers_to_skip=[0,1], fit_biases=False)
v2f1[-1][-1]['biases']=True

_ = nim_shared_sac.train(input_data=[stim, sacuse], output_data=Robs,train_indxs=Ui, test_indxs=Xi, data_filters=DF,
learning_alg='adam', opt_params=adam_params, fit_variables=v2f1)

# %% plot it
# importlib.reload(gt)
DU.plot_3dfilters(nim_shared)

filters = DU.compute_spatiotemporal_filters(nim_shared)
gt.plot_3dfilters(filters, basis=basis)
# DU.plot_3dfilters(nim_shared1)

plt.figure()
wt = nim_shared.networks[0].layers[0].weights
plt.subplot(1,2,1)
plt.plot(wt)
plt.xlabel('lags')
plt.subplot(1,2,2)
plt.imshow(wt.T@wt)

#%%
xax = np.arange(-back_shifts, num_saclags-back_shifts, 1)
wsac=nim_shared_sac.networks[1].layers[0].weights
plt.subplot(1,2,1)
plt.plot(xax,wsac)
plt.subplot(1,2,2)
plt.imshow(wsac.T@wsac)

# ws2 = nim_shared_sac2.networks[1].layers[1].weights
# a = NDNutils.create_time_embedding(wsac2, [3, 2, 1])@ws2


#%%
sacpar = NDNutils.ffnetwork_params( 
    input_dims=[1,1,1], layer_sizes=[1,NC], xstim_n=[1], time_expand=[num_saclags, 3],
    layer_types=['temporal', 'normal'], normalization=[1, 1],
    act_funcs=['lin', 'lin'], verbose=True,
    reg_list={'orth': [1e-1], 'd2t':[0.1], 'l1':[None, L1reg0]})

sacparNeg = NDNutils.ffnetwork_params( 
    input_dims=[1,1,1], layer_sizes=[2,NC], xstim_n=[2], time_expand=[num_saclags, 3],
    layer_types=['temporal', 'normal'], normalization=[1, 1],
    act_funcs=['lin', 'lin'], verbose=True,
    reg_list={'orth': [1e-1], 'd2t':[0.1], 'l1':[None, L1reg0]})    

combpar = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1,2], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus']
)
sacpar['pos_constraints'] = [True,None]
sacparNeg['pos_constraints'] = [True,None]

nim_shared_sac2 = NDN.NDN([par2, sacpar, sacparNeg, combpar],  noise_dist='poisson', ffnet_out=3)

nim_shared_sac2.networks[0].layers[0].weights = deepcopy(nim_shared_sac.networks[0].layers[0].weights)
nim_shared_sac2.networks[0].layers[1].weights = deepcopy(nim_shared_sac.networks[0].layers[1].weights)
nim_shared_sac2.networks[-1].layers[-1].biases = deepcopy(nim_shared_sac.networks[-1].layers[-1].biases)

# from scipy.linalg import orth
# nim_shared_sac2.networks[1].layers[1].weights = orth(nim_shared_sac2.networks[1].layers[1].weights)

# initialize temporal lag from saccade kernel
nim_shared_sac2.networks[1].layers[1].weights[0,:] = 1.0
nim_shared_sac2.networks[1].layers[1].weights[1:,:] = 0.0
nim_shared_sac2.networks[2].layers[1].weights[0,:] = 1.0
nim_shared_sac2.networks[2].layers[1].weights[1:,:] = 0.0

v2f1 = nim_shared_sac2.fit_variables(layers_to_skip=[0,1,4,6], fit_biases=False)
v2f1[-1][-1]['biases']=True


_ = nim_shared_sac2.train(input_data=[stim, sacuse, -sacuse], output_data=Robs,train_indxs=Ui, test_indxs=Xi, data_filters=DF,
learning_alg='adam', opt_params=adam_params, fit_variables=v2f1)

#%%
plt.figure()
wsac2 = nim_shared_sac2.networks[2].layers[0].weights
ws2 = nim_shared_sac2.networks[2].layers[1].weights
# ws3 = nim_shared_sac2.networks[1].layers[2].weights
plt.subplot(1,2,1)
plt.plot(wsac2)
plt.subplot(1,2,2)
plt.plot(ws2)

plt.figure()
a = NDNutils.create_time_embedding(wsac2, [3, 1, 1])@ws2
plt.plot(a)
#%%
LLs0 = nim_shared.eval_models(input_data=[Xstim], output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)
# LLs1 = nim_shared1.eval_models(input_data=[Xstim], output_data=Robs, data_filters=DF,
#                         data_indxs=Xi, nulladjusted=True) 

plt.figure()
plt.plot(LLs0, LLs1, '.')
plt.plot([0,1],[0,1], 'k')
plt.xlabel('LL (No Orth)')
plt.ylabel('LL (Orth)')

plt.figure()
plt.subplot(2,2,1)
w = nim_shared.networks[0].layers[0].weights
plt.plot(w)
plt.subplot(2,2,3)
plt.imshow(w.T@w)

w = nim_shared.networks[0].layers[0].weights
plt.subplot(2,2,2)
plt.plot(w)
plt.subplot(2,2,4)
plt.imshow(w.T@w)





# DU.plot_2dweights(nim_shared.networks[0].layers[-1].weights)


# %% full GLM
XTreg = None
L1reg = None
valdata = np.arange(0,NT,1)

Ui, Xi = NDNutils.generate_xv_folds(NT, num_blocks=2)

glm_par = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY, num_lags], layer_sizes=[NC], layer_types=['normal'],
    normalization=[1], act_funcs=['softplus'],
    reg_list={'d2x':[XTreg], 'l1':[L1reg]})

glm0 = NDN.NDN( glm_par, noise_dist='poisson' )
    
_ = glm0.train(input_data=[Xstim], output_data=Robs, train_indxs=Ui, test_indxs=Xi, data_filters=DF,
                   learning_alg='lbfgs', opt_params=lbfgs_params)

LLs0 = glm0.eval_models(input_data=[Xstim], output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)
print(np.mean(LLs0))

# %% Plot it
DU.plot_3dfilters(glm0)
#%%
rrhat = glm0.generate_prediction(input_data=[Xstim], ffnet_target=-1, layer_target=-1)
# plt.plot(rrhat[0:100,:])

plt.subplot(1,2,1)
plt.imshow(rrhat[0:100,:], aspect='auto')
plt.subplot(1,2,2)
plt.imshow(Robs[0:100,:], aspect='auto')
#%% plot spatial glm weights at all lags
fig, ax = plt.subplots(nrows=NC, ncols=num_lags, sharex=True, sharey=True, figsize=(8, NC*1))
wall = np.reshape(glm0.networks[0].layers[0].weights, (NX*NY, num_lags, NC))
for cc in range(NC):
    w = wall[:,:,cc]
    wB = basis['B']@w
    vmin = np.min(wB)
    vmax = np.max(wB)
    for i in range(num_lags):
        ax[cc,i].imshow(np.reshape(wB[:,i], (basis['support'], basis['support'])), vmin=vmin, vmax=vmax, cmap='Blues')
        ax[cc,i].axis("off")
        if cc==0:
            ax[cc,i].set_title('lag %d' %i)

plt.tight_layout()
fig.text(0.5, 0.04, 'Orientation', ha='center')
fig.text(0.04, 0.5, 'Spatial Frequency', va='center', rotation='vertical')

#%% plot on basis
w = np.reshape(glm0.networks[0].layers[0].weights, (NX*NY, num_lags, NC))
sx, sy = U.get_subplot_dims(NC)
plt.figure(figsize=(8.5,10))
# fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
# fig.set_size_inches(16, row_height*num_rows)
for cc in range(NC):
    ax = plt.subplot(sx,sy,cc+1)
    plt.plot(np.transpose(w[:,:,cc]))
    plt.axis("off")
    # bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    # plt.imshow(np.reshape(stas[:,bestlag,cc], [NY, NX]), cmap='gray')  #RdBu_r 
               #vmin=-np.max(abs(stas[:,:,cc])), vmax=np.max(abs(stas[:,:,cc])))
    # plt.title(str(bestlag)+' cell'+str(cc),)
    plt.title('cell'+str(cc),)
plt.show()

#%% try estimating regularization regularization

# xt reg path
RLLsT, RmodsT = NDNutils.reg_path(
    ndn_mod=glm0, reg_type='d2x', ffnet_target=0, layer_target=0, data_filters=DF,
    input_data=[Xstim], output_data=Robs, train_indxs=Ui, test_indxs=Xi, opt_params=adam_params)
plt.plot(RLLsT,'b')
plt.show()

glm1 = RmodsT[np.where(RLLsT==np.min(RLLsT))[0][0]].copy_model()

# %%
RLLsT, RmodsT = NDNutils.reg_path(
    ndn_mod=glm1, reg_type='l1', ffnet_target=0, layer_target=0,
    input_data=[Xstim], output_data=Robs, train_indxs=Ui, test_indxs=Xi, opt_params=adam_params)
plt.plot(RLLsT,'b')
plt.show()

# store max
glm2 = RmodsT[np.where(RLLsT==np.min(RLLsT))[0][0]].copy_model()

# %%
DU.plot_3dfilters(glm1)

#%% low rank? Initialize with SVD
filters = DU.compute_spatiotemporal_filters(glm2)
F = np.reshape(filters, (np.prod(filters.shape[:3]), filters.shape[3]))
u,s,v = np.linalg.svd(F, full_matrices=False)
u = np.reshape(u, filters.shape)
gt.plot_3dfilters(filters=u, basis=basis)

# %% saccade kernels


# %% saccade-triggered average
delta = 1/0.0083
sacta = XsacOn.T @ Robs / np.sum(XsacOn) * delta
sacta /= np.mean(sacta, axis=0)
xax = np.arange(-back_shifts, num_saclags-back_shifts, 1)
h = plt.plot(xax,sacta)
plt.show



#%% full GLM
Xreg = 0.00001
L1reg = None
glmFull_par = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags], layer_sizes=[NC], 
        layer_types=['normal'], act_funcs=['softplus'], 
        ei_layers=[None], normalization=[1], # normalization really matters
        reg_list={'d2x':[Xreg],'l1':[L1reg],'max_filt':[None]} )
glmFull = NDN.NDN( [glmFull_par], tf_seed = 5, noise_dist='poisson')

_ = glmFull.train(input_data=[Xstim], output_data=Robs, data_filters=DF, train_indxs=Ui, test_indxs=Xi,
    learning_alg='adam', opt_params=adam_params)
LLx0 = glmFull.eval_models(input_data=Xstim, output_data=Robs, data_filters=DF,
    data_indxs=Xi, nulladjusted=True)

#%% plot it
DU.plot_3dfilters(glmFull)
# DU.plot_3dfilters(glm1)
#%% Low-rank filters
w = glmFull.networks[0].layers[0].weights
u,s,v = np.linalg.svd(w, full_matrices=False)

plt.plot(s, '-o')

# %% Shared GLM
Treg = 0.0001
Xreg = 0.0001
L1reg = 0.00001
num_tkerns = 0
glmShare = []
LLxShare = []
subs_num = [10]#np.arange(1,NC,1)
for num_subs in list(subs_num):
    print('Fitting %d-dimensional shared GLM:' %(num_subs))
    
    if num_tkerns > 0: # use temporal convolution layer?
        glmShare_par = NDNutils.ffnetwork_params( 
            input_dims=[1,NX,NY,num_lags], layer_sizes=[num_tkerns, num_subs, NC], 
            layer_types=['conv', 'normal','normal'], act_funcs=['lin', 'lin','softplus'], 
            ei_layers=[None,None], normalization=[1,1,0], conv_filter_widths=[1],
            reg_list={'orth': [1e-3, None], 'd2x':[None, Xreg],'l1':[None, None,L1reg],'max_filt':[None,None]} )
    else:
        glmShare_par = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags], layer_sizes=[num_subs, NC], 
        layer_types=['normal','normal'], act_funcs=['lin','softplus'], 
        ei_layers=[None,None], normalization=[1,0],
        reg_list={'orth': 1e-3,'d2x':[Xreg],'l1':[None,L1reg],'max_filt':[None,None]} )

    # reduce degeneracy: only fit biases on the final layer
    glmShare0 = NDN.NDN( [glmShare_par], tf_seed = 5, noise_dist='poisson')
    v2f0 = glmShare0.fit_variables(fit_biases=False)
    v2f0[-1][-1]['biases']=True

    # train
    _ = glmShare0.train(input_data=[Xstim], output_data=Robs, data_filters=DF, train_indxs=Ui, test_indxs=Xi,
        learning_alg='adam', opt_params=adam_params, fit_variables=v2f0)
    # null-adjusted test-likelihood
    LLx0 = glmShare0.eval_models(input_data=Xstim, output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)
    glmShare.append(glmShare0.copy_model())
    LLxShare.append(LLx0)
    
LLxShare = np.asarray(LLxShare)

#%% Plot test-likelihoods as a function of the number of dimensions
plt.plot(np.array((1.0, np.max(subs_num))), np.mean(LLx0)*np.array((1.0, 1.0)), 'k')
plt.plot(subs_num, np.mean(LLxShare, axis=1), '-o')

#%%
plt.figure()
# plt.plot(LLx0, LLxShare[4,:], '.')
plt.plot(LLx0, LLx1, '.')
plt.plot([0.0, 1.0], [0.0, 1.0], 'k')
plt.xlabel('Full GLM')

#%%

glmShare0 = glmShare[5].copy_model()

#%% plot full model + 
DU.plot_3dfilters(glmFull)

# glmShare0 = glmShare[8].copy_model()
filters = DU.compute_spatiotemporal_filters(glmShare0)
w1 = np.reshape(filters, (NX*NY*num_lags, glmShare0.network_list[0]['layer_sizes'][-2]))
# w1 = glmShare0.networks[0].layers[0].weights
w2 = glmShare0.networks[-1].layers[-1].weights
w = w1 @ w2
filters = np.reshape(w, (NY, NX, num_lags, NC))

DU.plot_3dfilters(filters=filters)

# %% add saccade kernels
Treg = 0.1
L1reg = None
stim_par = glmShare0.network_list
stim_par = stim_par[0]
stim_par['activation_funcs'][-1] = 'lin'
stim_par['xstim_n'] = [0]

num_sac_dim = 4 # number of saccade kernels to share
sac_on_full = NDNutils.ffnetwork_params(
    input_dims=[1,1,1,num_saclags], xstim_n=[1],
    layer_sizes=[num_sac_dim, NC], layer_types=['conv', 'normal'], normalization=[1],
    act_funcs=['lin', 'lin'], conv_filter_widths=[1],
    reg_list={'d2t':[Treg], 'orth':1e-1}
)

comb_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus']
)


sacglmadd = NDN.NDN([stim_par, sac_on_full, comb_par], ffnet_out=2,
    noise_dist='poisson')


sacglmadd.networks[0].layers[0].weights = deepcopy(glmShare0.networks[0].layers[0].weights)
sacglmadd.networks[0].layers[1].weights = deepcopy(glmShare0.networks[0].layers[1].weights)
sacglmadd.networks[0].layers[0].biases = deepcopy(glmShare0.networks[0].layers[0].biases)
sacglmadd.networks[0].layers[1].biases = deepcopy(glmShare0.networks[0].layers[1].biases)

v2f0 = sacglmadd.fit_variables(layers_to_skip=[0], fit_biases=False)
v2f0[-1][0]['biases']=True

#%% train
_ = sacglmadd.train(
    input_data=[Xstim, XsacOn], output_data=Robs, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)

sacaddcaus = sacglmadd.copy_model()
_ = sacaddcaus.train(
    input_data=[Xstim, XsacOnCausal], output_data=Robs, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)

#%% smoothing
RLLsT, RmodsT = NDNutils.reg_path(
    ndn_mod=sacglmadd, reg_type='d2t', ffnet_target=1, layer_target=0, data_filters=DF,
    input_data=[Xstim, XsacOn], output_data=Robs, train_indxs=Ui, test_indxs=Xi, opt_params=adam_params)
plt.plot(RLLsT,'b')
plt.show()

sacglmadd1 = RmodsT[np.where(RLLsT==np.min(RLLsT))[0][0]].copy_model()

#%%
LLxsac = sacglmadd.eval_models(input_data=[Xstim, XsacOn], output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)

LLxsacc = sacaddcaus.eval_models(input_data=[Xstim, XsacOn], output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)

plt.figure()
plt.subplot(1,2,1)
plt.plot(LLx0, LLxsac, '.')
plt.plot([0,1], [0, 1], 'k')
plt.xlim((0.0, 1.2))
plt.ylim((0.0, 1.2))
plt.subplot(1,2,2)
plt.plot(LLxsacc, LLxsac, '.')
plt.plot([0,1], [0, 1], 'k')
plt.xlim((0.0, 1.2))
plt.ylim((0.0, 1.2))

#%% plot
plt.plot(sacglmadd.networks[1].layers[0].weights)
w1 = sacglmadd.networks[1].layers[0].weights
w2 = sacglmadd.networks[1].layers[1].weights
plt.figure()
w = w1@w2
p = plt.plot(w)
plt.show()

plt.figure()
plt.imshow(w1.T@w1)
# %% try fitting the multiplicative weights
add_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[2,3], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus']
)

mult_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],normalization=[1],
    layer_types=['mult'], act_funcs=['lin']
)

sacglmmult = NDN.NDN([stim_par, sac_on_full, sac_on_full, mult_par, add_par], ffnet_out=4,
    noise_dist='poisson')

# inititialize stimulus
sacglmmult.networks[0].layers[0].weights = deepcopy(glmShare0.networks[0].layers[0].weights)
sacglmmult.networks[0].layers[1].weights = deepcopy(glmShare0.networks[0].layers[1].weights)
sacglmmult.networks[0].layers[0].biases = deepcopy(glmShare0.networks[0].layers[0].biases)
sacglmmult.networks[0].layers[1].biases = deepcopy(glmShare0.networks[0].layers[1].biases)

# initialize additive saccade kernels
sacglmmult.networks[1].layers[0].weights = deepcopy(sacglmadd.networks[1].layers[0].weights)
sacglmmult.networks[1].layers[1].weights = deepcopy(sacglmadd.networks[1].layers[1].weights)

# initialize multiplicative saccade kernels (with the additive-only ones)
sacglmmult.networks[2].layers[0].weights = deepcopy(sacglmadd.networks[1].layers[0].weights)
sacglmmult.networks[2].layers[1].weights = deepcopy(sacglmadd.networks[1].layers[1].weights)

# specify variables to fit
v2f0 = sacglmmult.fit_variables(layers_to_skip=[0,2], fit_biases=False)
v2f0[-1][0]['biases']=True # fit last layer biases


#%% train
_ = sacglmmult.train(
    input_data=[Xstim, XsacOn], output_data=Robs, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)

# fit causal model (identical, but the saccade times are not shifted)
sacglmmultcaus = sacglmmult.copy_model()
_ = sacglmmultcaus.train(
    input_data=[Xstim, XsacOnCausal], output_data=Robs, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)

#%% train 2, free variables
v2f1 = sacglmmult.fit_variables(fit_biases=False)
v2f1[-1][0]['biases']=True # fit last layer biases
_ = sacglmmult.train(
    input_data=[Xstim, XsacOn], output_data=Robs, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)


#%% plot it

plt.figure(figsize=(8.5,11))

w1m = sacglmmult.networks[1].layers[0].weights
w2m = sacglmmult.networks[1].layers[1].weights

w1a = sacglmmult.networks[2].layers[0].weights
w2a = sacglmmult.networks[2].layers[1].weights

plt.subplot(3,2,1)
xax = np.linspace(-back_shifts, num_saclags-back_shifts, num_saclags)
p = plt.plot(xax, w1m)
plt.plot([xax[0], xax[-1]], [0,0], 'k')
plt.xlabel('Time from saccade onset')
plt.title('Gain')

plt.subplot(3,2,2)
p = plt.plot(xax, w1a)
plt.plot([xax[0], xax[-1]], [0,0], 'k')
plt.xlabel('Time from saccade onset')
plt.title('Offset')

plt.subplot(3,2,3)
plt.plot(w2m.T)
plt.xlabel('Neuron')

plt.subplot(3,2,4)
plt.plot(w2a.T)
plt.xlabel('Neuron')


plt.subplot(3,2,5)
wm = w1m@w2m
p = plt.plot(xax,wm)

plt.subplot(3,2,6)
wa = w1a@w2a
p = plt.plot(xax,wa)

sns.despine()

plt.figure()
plt.plot(w2m[1,:], '.')

#%%
a = np.sum(wa[xax < 0,:], axis=0)
m = np.sum(wm[xax < 0,:], axis=0)
plt.figure()
plt.plot(a, m, '.')
xd = plt.xlim()
yd = plt.ylim()
plt.plot([0,0],yd,'k--')
plt.plot(xd,[0,0],'k--')
plt.xlabel('Additive')
plt.ylabel('Multiplicative')



#%%
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
sacglmfull.networks[0].layers[0].weights = deepcopy()
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

#%% Fit models sequentially
sacglmfull.networks 



#%% BIG MODEL
num_subs = nim_shared.networks[0].layers[0].weights.shape[1]
LOCreg = 0.001
TSreg2 = 0.001

# Par 0: stimulus processing (shared NIM)
mnim1_par = NDNutils.ffnetwork_params( 
    input_dims=[1,NY,NX, num_lags], layer_sizes=[num_subs, NC], 
    layer_types=['normal', 'normal'], normalization=[1,0], act_funcs=['relu', 'lin'],
    reg_list={'d2xt':[Xreg], 'l1':[None, L1reg/2], 'local':[LOCreg]})

# Par 1: saccade onset kernel
sac1_par = NDNutils.ffnetwork_params( 
    input_dims=[1,1,1, num_saclags], xstim_n=[1],
    layer_sizes=[1, NC], layer_types=['normal', 'normal'], normalization=[1,0], act_funcs=['lin', 'lin'],
    reg_list={'d2t':[TSreg2]})

# Par 2: multiply output of subunits with saccade-onset kernel
stim_par = NDNutils.ffnetwork_params( 
    xstim_n=None, ffnet_n=[0,1],
    layer_sizes=[NC], layer_types=['mult'], normalization=[0], act_funcs=['lin'])

# Par 3: saccade offset kernel
sac2_par = NDNutils.ffnetwork_params( 
    input_dims=[1,1,1, num_saclags], xstim_n=[2],
    layer_sizes=[1, NC], layer_types=['normal', 'normal'], normalization=[1, 0], act_funcs=['lin', 'lin'],
    reg_list={'d2t':[TSreg2]})

# combine the output of Par3 and Par4 (additive saccade offset kernel and saccade-modulated stimulus processing)
comb_par = NDNutils.ffnetwork_params( 
    xstim_n=None, ffnet_n=[2,3], layer_sizes=[NC], layer_types=['add'], act_funcs=['softplus'])

gmodsac0 = NDN.NDN( [mnim1_par, sac1_par, stim_par, sac2_par, comb_par], ffnet_out=4, noise_dist='poisson' )

gmodsac0.networks[0].layers[0].weights = deepcopy(nim_shared.networks[0].layers[0].weights)

v2f0 = gmodsac0.fit_variables(layers_to_skip=[[0]], fit_biases=True)


v2f0[1][1]['biases']=False
v2f0[2][0]['biases']=False
v2f0[3][1]['biases']=False
v2f0[4][0]['weights']=False

# %% train

_ = gmodsac0.train(input_data=[Xstim, XsacOn, XsacOff], output_data=Robs, train_indxs=Ui,
test_indxs=Xi, fit_variables=v2f0, learning_alg='adam', opt_params=adam_params, use_dropout=True)


# %%

DU.plot_3dfilters(gmodsac0)

# %%
plt.plot(gmodsac0.networks[1].layers[0].weights)

# %%
LLx = nim_shared.eval_models(input_data=[Xstim], output_data=Robs, data_indxs=Xi, nulladjusted=True)
LLx2 = gmodsac0.eval_models(input_data=[Xstim, XsacOn, XsacOff], output_data=Robs, data_indxs=Xi, nulladjusted=True)

plt.plot(LLx, LLx2, '.')
plt.plot([0, 1], [0, 1])
# %%
gmodsac1 = gmodsac0.copy_model()

RLLsT, RmodsT = NDNutils.reg_path(
    ndn_mod=gmodsac1, reg_type='d2t', ffnet_target=1, layer_target=0,
    input_data=[Xstim, XsacOn, XsacOff], output_data=Robs, train_indxs=Ui, test_indxs=Xi, opt_params=adam_param)
plt.plot(RLLsT,'b')
plt.show()

# %%
