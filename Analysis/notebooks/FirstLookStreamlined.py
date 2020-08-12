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
# import NDN3.Utils.HadiUtils as HU
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

indexlist = [4]
sess = [sesslist[i] for i in indexlist]

# shared basis
basis = {'name': 'cosine', 'nori': 8, 'nsf': 6, 'endpoints': [0.5, 1.8], 'support': 500}

stim, sacon, sacoff, Robs, DF, basis, opts = gt.load_sessions(sess, basis=basis) 

#%% plot basis functions
gt.plot_basis(basis)

# %% set optimization parmeters
adam_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')

early_stopping = 50

adam_params['batch_size'] = NT // 50
adam_params['display'] = 30
adam_params['epochs_training'] = early_stopping * 10
adam_params['run_diagnostics'] = False
adam_params['early_stop'] = early_stopping
adam_params['early_stop_mode'] = 3
adam_params['data_pipe_type'] = 'data_as_var' # 'feed_dict'
adam_params['learning_rate'] = 1e-2

lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_params['maxiter'] = 1000


# %% create time-embedded design matrix
num_saclags = 60
back_shifts = 20
num_time_shift = 4
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
Robs = Robs.astype('float32')

# %% get STA
stas = Xstim.T @ Robs / np.sum(Robs)
stas = stas.reshape([NX*NY,num_lags,NC])

# %% plot STA
import seaborn as sns
sx, sy = U.get_subplot_dims(NC)
plt.figure(figsize=(8.5,10))
for cc in range(NC):
    ax = plt.subplot(sx,sy,cc+1)
    plt.plot(np.transpose(stas[:,:,cc]))
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

# plt.tight_layout()
fig.text(0.5, 0.04, 'Orientation', ha='center')
fig.text(0.04, 0.5, 'Spatial Frequency', va='center', rotation='vertical')

# %% setup training indices
valdata = np.arange(0,NT,1)

Ui, Xi = NDNutils.generate_xv_folds(NT, num_blocks=2)

# Xi = Xi[:-(len(Xi) % adam_params['batch_size'])] # hack to avoid crashing on eval models
# %% saccade-triggered average
delta = 1/0.0083
sacta = XsacOn.T @ Robs / np.sum(XsacOn) * delta
sacta /= np.mean(sacta, axis=0)
xax = np.arange(-back_shifts, num_saclags-back_shifts, 1)
h = plt.plot(xax,np.mean(sacta, axis=1))
plt.show



#%% full GLM
Xreg = None
L1reg = None

useSep = False
if useSep:
    glmFull_par = NDNutils.ffnetwork_params( 
        input_dims=[num_lags,NX,NY], layer_sizes=[NC], 
        layer_types=['sep'], act_funcs=['softplus'], 
        ei_layers=[None], normalization=[-1], # normalization really matters
        reg_list={'d2x':[Xreg],'l1':[L1reg],'max_filt':[None]} )
else:
    glmFull_par = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags], layer_sizes=[NC], 
        layer_types=['normal'], act_funcs=['softplus'], 
        ei_layers=[None], normalization=[-1], # normalization really matters
        reg_list={'d2x':[Xreg],'l1':[L1reg],'max_filt':[None]} )

glmFull = NDN.NDN( [glmFull_par], tf_seed = 5, noise_dist='poisson')

glmFull.time_spread = num_lags+num_saclags+num_time_shift

# initialize with STA
stas = Xstim.T @ Robs / np.sum(Robs)
stas = stas / np.max(stas, axis=0)
glmFull.networks[0].layers[0].weights = stas.astype('float32')

adam_params['learning_rate'] = 1e-1
_ = glmFull.train(input_data=[Xstim], output_data=Robs, data_filters=DF, train_indxs=Ui, test_indxs=Xi,
    learning_alg='adam', opt_params=adam_params)

adam_params['learning_rate'] = 1e-4
_ = glmFull.train(input_data=[Xstim], output_data=Robs, data_filters=DF, train_indxs=Ui, test_indxs=Xi,
    learning_alg='adam', opt_params=adam_params)    



#%%

DU.plot_3dfilters(glmFull)

# %%
# LLx0 = glmFull.generate_prediction(input_data=Xstim)
LLx0 = glmFull.eval_models(input_data=Xstim, output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)
# , output_data=Robs, data_filters=DF,
#     data_indxs=Xi, nulladjusted=True)

print(LLx0)
#%% plot it 
# if glmFull.network_list[0]['layer_types'][0]=='sep':
#     w = deepcopy(glmFull.networks[0].layers[0].weights)
#     time = w[0:num_lags,:]
#     space = w[num_lags:,:]
#     # for i in range(NC):
# else:
#     filters = DU.compute_spatiotemporal_filters(glmFull)

gt.plot_3dfilters(filters, basis=basis)

#%%
filters

#%% Low-rank filters
w = glmFull.networks[0].layers[0].weights
u,s,v = np.linalg.svd(w, full_matrices=False)

plt.plot(np.cumsum(s)/np.sum(s), '-o')
plt.xlabel('Rank')
plt.ylabel('Var Explained Weights')
# %% Shared GLM
Treg = 0.0001
Xreg = 0.001
L1reg = None
num_tkerns = 0
glmShare = []
LLxShare = []
subs_num = [5]#np.arange(2,15,3)
for num_subs in list(subs_num):
    print('Fitting %d-dimensional shared GLM:' %(num_subs))
    
    if num_tkerns > 0: # use temporal convolution layer?
        glmShare_par = NDNutils.ffnetwork_params( 
            input_dims=[1,NX,NY,num_lags], layer_sizes=[num_tkerns, num_subs, NC], 
            layer_types=['normal', 'normal','normal'], act_funcs=['lin', 'tanh','softplus'], 
            ei_layers=[None,None], normalization=[1,1,0], conv_filter_widths=[1],
            reg_list={'orth': [1e-1, 1e-1], 'd2x':[None, Xreg],'l1':[None, None,L1reg],'max_filt':[None,None]} )
        glmShare0 = NDN.NDN( [glmShare_par], tf_seed = 5, noise_dist='poisson')
    else:
        glmShare_par = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags], layer_sizes=[num_subs, NC], time_expand=[0],
        layer_types=['normal','normal'], act_funcs=['lin','softplus'], 
        ei_layers=[None,None], normalization=[-1],
        reg_list={'orth': 1e-1,'d2x':[Xreg],'l1':[None,L1reg],'max_filt':[None,None]} )
        glmShare0 = NDN.NDN( [glmShare_par], tf_seed = 5, noise_dist='poisson')
        # initialize based on 
        glmShare0.networks[0].layers[0].weights = deepcopy(u[:,:num_subs])
        
    glmShare0.networks[-1].layers[-1].biases = deepcopy(glmFull.networks[-1].layers[-1].biases)
    v2f0 = glmShare0.fit_variables(fit_biases=False)
    v2f0[-1][-1]['biases']=True

    glmShare0.time_spread = num_lags+num_saclags+num_time_shift
    # train
    _ = glmShare0.train(input_data=[Xstim], output_data=Robs, data_filters=DF, train_indxs=Ui, test_indxs=Xi,
        learning_alg='adam', opt_params=adam_params, fit_variables=v2f0)
    # null-adjusted test-likelihood
    LLx1 = glmShare0.eval_models(input_data=Xstim, output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)
    glmShare.append(glmShare0.copy_model())
    LLxShare.append(LLx1)
    
LLxShare = np.asarray(LLxShare)

#%% Plot test-likelihoods as a function of the number of dimensions
plt.plot(np.array((1.0, np.max(subs_num))), np.mean(LLx0)*np.array((1.0, 1.0)), 'k')
plt.plot(subs_num, np.mean(LLxShare, axis=1), '-o')
plt.ylabel('Average LL')
plt.xlabel("Subspace Rank")
sns.despine()

#%%
plt.figure()
# plt.plot(LLx0, LLxShare[4,:], '.')
plt.plot(LLx0, LLx1, '.')
plt.plot([0.0, 1.0], [0.0, 1.0], 'k')
plt.xlabel('Full GLM')
plt.ylabel('Low-Rank GLM')

#%%

glmShare0 = glmShare[5].copy_model()

#%% plot full model + 
filters = DU.compute_spatiotemporal_filters(glmFull)
gt.plot_3dfilters(filters, basis=basis)

# glmShare0 = glmShare[8].copy_model()
filters = DU.compute_spatiotemporal_filters(glmShare0)
gt.plot_3dfilters(filters, basis=basis)

w1 = np.reshape(filters, (NX*NY*num_lags, glmShare0.network_list[0]['layer_sizes'][-2]))
w2 = glmShare0.networks[-1].layers[-1].weights
w = w1 @ w2
filters = np.reshape(w, (NY, NX, num_lags, NC))

gt.plot_3dfilters(filters=filters, basis=basis)

# %%
plt.imshow(glmShare0.networks[0].layers[1].weights, aspect='auto')

# %% add saccade kernels
Treg = 0.1
L1reg = None
stim_par = glmShare0.network_list
stim_par = stim_par[0]
stim_par['activation_funcs'][-1] = 'lin'
stim_par['xstim_n'] = [0]

sacshift = NDNutils.shift_mat_zpad(sacon,-back_shifts,dim=0)

sacGlmAdd = []
sacGlmAddC = []
LLxSac = []
LLxSacC = []

# loop over number of saccade kernels shared by population
sac_num_dim = np.arange(1,5,1)
for num_sac_dim in list(sac_num_dim):
    print('Fitting %d-dimensional saccade kernel:' %(num_sac_dim))
    sac_on_full = NDNutils.ffnetwork_params(
        input_dims=[1,1,1], xstim_n=[1], time_expand=[num_saclags],
        layer_sizes=[num_sac_dim, NC], layer_types=['temporal', 'normal'],
        normalization=[1], act_funcs=['lin', 'lin'],
        reg_list={'d2t':[Treg], 'orth':[1e-1]}
    )

    comb_par = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus']
    )

    sacglmadd = NDN.NDN([stim_par, sac_on_full, comb_par], ffnet_out=2,
        noise_dist='poisson')

    # copy RF parameters from shared GLM
    sacglmadd.networks[0].layers[0].weights = deepcopy(glmShare0.networks[0].layers[0].weights)
    sacglmadd.networks[0].layers[1].weights = deepcopy(glmShare0.networks[0].layers[1].weights)
    sacglmadd.networks[0].layers[0].biases = deepcopy(glmShare0.networks[0].layers[0].biases)
    sacglmadd.networks[0].layers[1].biases = deepcopy(glmShare0.networks[0].layers[1].biases)

    # don't refit the RFs
    v2f0 = sacglmadd.fit_variables(layers_to_skip=[0], fit_biases=False)
    v2f0[-1][0]['biases']=True

    sacglmadd.time_spread = num_lags+num_saclags+num_time_shift
    sacaddcaus = sacglmadd.copy_model() # causal model is identical, but the input is different

    # train
    _ = sacglmadd.train(
        input_data=[Xstim, sacshift], output_data=Robs, data_filters=DF,
        train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
        learning_alg='adam', opt_params=adam_params
    )
    
    _ = sacaddcaus.train(
        input_data=[Xstim, sacon], output_data=Robs, data_filters=DF,
        train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
        learning_alg='adam', opt_params=adam_params
    )
    
    # null-adjusted test-likelihood
    LLxs1 = sacglmadd.eval_models(input_data=[Xstim, sacshift], output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)
    LLxs1C = sacaddcaus.eval_models(input_data=[Xstim, sacon], output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)

    sacGlmAdd.append(sacglmadd)
    sacGlmAddC.append(sacaddcaus)
    LLxSac.append(LLxs1)
    LLxSacC.append(LLxs1C)
    
LLxSac = np.asarray(LLxSac)
LLxSacC = np.asarray(LLxSacC)

#%% plot output

plt.plot(sac_num_dim, np.mean(LLxSac, axis=1), '-o')
plt.plot(sac_num_dim, np.mean(LLxSacC, axis=1), '-o')
plt.xlabel("# saccade kernels")

plt.figure()
plt.subplot(1,2,1)
xax = np.arange(-back_shifts, num_saclags-back_shifts, 1)*8
plt.plot(xax, sacGlmAdd[1].networks[1].layers[0].weights)
plt.xlabel("Time from saccade onset")
plt.subplot(1,2,2)
xax = np.linspace(1, num_saclags, num_saclags)*8
plt.plot(xax,sacGlmAddC[1].networks[1].layers[0].weights)
plt.xlabel("Time from saccade onset")

plt.figure()
plt.plot(LLxSacC[1,:], LLxSac[1,:], '.')
plt.plot([0,1], [0,1], '-k')
plt.xlabel("Causal Only")
plt.ylabel("Presaccadic included")
sns.despine(offset=10)


#%% try single saccade kernel with lagged neuron responses
num_sac_dim = 2
num_time_shift = 0
if num_time_shift>0:
    sac_on_full = NDNutils.ffnetwork_params(
        input_dims=[1,1,1], xstim_n=[1], time_expand=[num_saclags, num_time_shift],
        layer_sizes=[num_sac_dim, NC], layer_types=['temporal', 'normal'],
        normalization=[1], act_funcs=['lin', 'lin'],
        reg_list={'d2t':[Treg], 'orth':[None]}
    )
else:
    sac_on_full = NDNutils.ffnetwork_params(
        input_dims=[1,1,1], xstim_n=[1], time_expand=[num_saclags],
        layer_sizes=[num_sac_dim, NC], layer_types=['temporal', 'normal'],
        normalization=[1], act_funcs=['lin', 'lin'],
        reg_list={'d2t':[Treg], 'orth':[1e-1]}
    )

comb_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus']
)

sacglmadd = NDN.NDN([stim_par, sac_on_full, comb_par], ffnet_out=2,
    noise_dist='poisson')

# copy RF parameters from shared GLM
sacglmadd.networks[0].layers[0].weights = deepcopy(glmShare0.networks[0].layers[0].weights)
sacglmadd.networks[0].layers[1].weights = deepcopy(glmShare0.networks[0].layers[1].weights)
sacglmadd.networks[0].layers[0].biases = deepcopy(glmShare0.networks[0].layers[0].biases)
sacglmadd.networks[0].layers[1].biases = deepcopy(glmShare0.networks[0].layers[1].biases)

# copy saccade kernel from 1-D saccade model
sacglmadd.networks[1].layers[0].weights = deepcopy(sacGlmAdd[num_sac_dim-1].networks[1].layers[0].weights)
sacglmadd.networks[1].layers[1].weights = deepcopy(sacGlmAdd[num_sac_dim-1].networks[1].layers[1].weights)

if num_time_shift>0:
    # initialize latencies 
    sacglmadd.networks[1].layers[1].weights[0,:] = 1.0
    sacglmadd.networks[1].layers[1].weights[1:,:] = 0.0

# don't refit the RFs or the saccade kernel
v2f0 = sacglmadd.fit_variables(layers_to_skip=[[0,1], [0]], fit_biases=False)
v2f0[-1][0]['biases']=True

#%% fit

# enforce time-spread to be the same as above
sacglmadd.time_spread = sacGlmAdd[0].time_spread
sacaddcaus = sacglmadd.copy_model() # causal model is identical, but the input is different (initial weights are differet)
sacaddcaus.networks[1].layers[0].weights = deepcopy(sacGlmAddC[num_sac_dim-1].networks[1].layers[0].weights)
sacaddcaus.networks[1].layers[1].weights = deepcopy(sacGlmAddC[num_sac_dim-1].networks[1].layers[1].weights)

# train
_ = sacglmadd.train(
    input_data=[Xstim, sacshift], output_data=Robs, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)
    
_ = sacaddcaus.train(
    input_data=[Xstim, sacon], output_data=Robs, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)
    
# null-adjusted test-likelihood
LLxs1 = sacglmadd.eval_models(input_data=[Xstim, sacshift], output_data=Robs, data_filters=DF,
    data_indxs=Xi, nulladjusted=True)
LLxs1C = sacaddcaus.eval_models(input_data=[Xstim, sacon], output_data=Robs, data_filters=DF,
    data_indxs=Xi, nulladjusted=True)

#%% refit all variables
v2f0 = sacglmadd.fit_variables(fit_biases=False)
v2f0[-1][0]['biases']=True

# train
_ = sacglmadd.train(
    input_data=[Xstim, sacshift], output_data=Robs, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)
    
_ = sacaddcaus.train(
    input_data=[Xstim, sacon], output_data=Robs, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)
    
# null-adjusted test-likelihood
LLxs1 = sacglmadd.eval_models(input_data=[Xstim, sacshift], output_data=Robs, data_filters=DF,
    data_indxs=Xi, nulladjusted=True)
LLxs1C = sacaddcaus.eval_models(input_data=[Xstim, sacon], output_data=Robs, data_filters=DF,
    data_indxs=Xi, nulladjusted=True)
#%% plot it all
binsize=1e3/opts['frate']
plt.figure()
filters = DU.compute_spatiotemporal_filters(sacglmadd)
gt.plot_3dfilters(filters, basis=basis)
plt.figure()
filters = DU.compute_spatiotemporal_filters(sacaddcaus)
gt.plot_3dfilters(filters, basis=basis)

if num_time_shift>0:
    plt.figure(figsize=(7,7))
    plt.subplot(2,2,1)
    xax = np.linspace(-back_shifts, num_saclags-back_shifts, num_saclags)
    p=plt.plot(xax, sacglmadd.networks[1].layers[0].weights)
    plt.plot([xax[0], xax[-1]], [0,0], 'k')
    plt.xlabel('Time from saccade onset')
    plt.title('Saccade Kernel')

    plt.subplot(2,2,2)
    p=plt.plot(sacglmadd.networks[1].layers[1].weights)
    plt.xlabel('Time-expansion')
    plt.title("neuron specific weights")

    w = NDNutils.create_time_embedding(sacglmadd.networks[1].layers[0].weights, [num_time_shift, 1, 1])
    w2 = w @ sacglmadd.networks[1].layers[1].weights

    plt.subplot(2,2,3)
    p=plt.plot(w)
    plt.title('Time-Expansion')

    plt.subplot(2,2,4)
    p=plt.plot(xax, w2)
    plt.xlabel('Time from saccade onset')
    plt.title('Net effect on each neuron')
else:
    plt.figure(figsize=(7,10))
    plt.subplot(2,2,1)
    xax = np.linspace(-back_shifts, num_saclags-back_shifts, num_saclags)*binsize
    p=plt.plot(xax, sacglmadd.networks[1].layers[0].weights)
    plt.plot([xax[0], xax[-1]], [0,0], 'k')
    plt.xticks([-200, -100,0, 100, 200])
    plt.xlabel('Time from saccade onset')
    plt.title('Saccade Kernel')

    plt.subplot(2,2,3)
    p=plt.plot(xax, sacglmadd.networks[1].layers[0].weights@sacglmadd.networks[1].layers[1].weights)
    plt.plot([xax[0], xax[-1]], [0,0], 'k')
    plt.xticks([-200, -100,0, 100, 200])
    plt.xlabel('Time from saccade onset')
    plt.title('Neuron Effect')

    plt.subplot(2,2,2)
    xax = np.linspace(1, num_saclags, num_saclags)*binsize
    p=plt.plot(xax, sacaddcaus.networks[1].layers[0].weights)
    plt.plot([xax[0], xax[-1]], [0,0], 'k')
    plt.xlabel('Time from saccade onset')
    plt.title('Saccade Kernel')
    plt.xticks([0,200,400])

    plt.subplot(2,2,4)
    p=plt.plot(xax, sacaddcaus.networks[1].layers[0].weights@sacaddcaus.networks[1].layers[1].weights)
    plt.plot([xax[0], xax[-1]], [0,0], 'k')
    plt.xlabel('Time from saccade onset')
    plt.title('Neuron Effect')
    plt.xticks([0,200,400])
    sns.despine(offset=10, trim=True)


#%%
plt.figure()
plt.plot(LLxs1C, LLxs1, '.')
plt.plot([0.0, 1.0], [0.0, 1.0], 'k')
plt.xlabel("Causal Only")
plt.ylabel("Pre-saccadic Lags Included")
sns.despine(offset=10, trim=True)


#%%

LLxFull = glmFull.eval_models(input_data=[Xstim], output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)

LLxStim = glmShare0.eval_models(input_data=[Xstim], output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)
LLxsac = sacglmadd.eval_models(input_data=[Xstim, sacshift], output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)

LLxsacc = sacaddcaus.eval_models(input_data=[Xstim, sacon], output_data=Robs, data_filters=DF,
                        data_indxs=Xi, nulladjusted=True)
plt.figure(figsize=(7,4))
plt.subplot(1,2,1)
plt.plot(LLxFull, LLxStim, '.')
plt.plot([0,1], [0, 1], 'k')
plt.xlim((0.0, 1.2))
plt.ylim((0.0, 1.2))
plt.xlabel("Full-Rank GLM")
plt.ylabel("Low-Rank GLM")
sns.despine(offset=10, trim=True)

#%%
plt.figure(figsize=(7,4))
plt.subplot(1,2,1)
plt.plot(LLxStim, LLxsac, '.')
plt.plot([0,1], [0, 1], 'k')
plt.xlim((0.0, 1.2))
plt.ylim((0.0, 1.2))
plt.xlabel("Stim Only")
plt.ylabel("Stim + Saccade")

plt.subplot(1,2,2)
plt.plot(LLxsacc, LLxsac, '.')
plt.xlabel("Stim + Post Saccade")
plt.ylabel("Stim + Pre + Post Saccade")
plt.plot([0,1], [0, 1], 'k')
plt.xlim((0.0, 1.2))
plt.ylim((0.0, 1.2))
sns.despine(offset=10, trim=True)

plt.figure(figsize=(7,4))
plt.subplot(1,2,1)
plt.hist(2**(LLxsac-LLxStim))
yd = plt.ylim()
# plt.plot([0,0], yd, 'k--')
plt.xlabel('L ratio (Sac+Stim/Stim)')
# plt.xlim([-0.05, 0.05])

plt.subplot(1,2,2)
plt.hist(2**(LLxsac-LLxsacc))
yd = plt.ylim()
plt.plot([0,0], yd, 'k--')
plt.xlabel('L ratio (Peri/Causal)')
sns.despine(offset=10, trim=True)
# plt.xlim([-0.05, 0.05])

plt.figure(figsize=(7,4))
plt.subplot(1,2,1)
plt.hist(2**(LLxsac-LLxStim))
yd = plt.ylim()
plt.xlabel('L ratio (Sac+Stim/Stim)')

plt.subplot(1,2,2)
plt.hist(2**(LLxsac-LLxsacc))
yd = plt.ylim()
# plt.plot([0,0], yd, 'k--')
plt.xlabel('L ratio (Peri/Causal)')
sns.despine(offset=10, trim=True)
# plt.xlim([-0.05, 0.05])
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

sac_on_full_mult = NDNutils.ffnetwork_params(
    input_dims=[1,1,1], xstim_n=[1], time_expand=[num_saclags,num_time_shift],
    layer_sizes=[1, NC], layer_types=['temporal', 'normal'],
    normalization=[1], act_funcs=['lin', 'tanh'],
    reg_list={'d2t':[Treg], 'orth':[None]}
)

# sacglmmult = NDN.NDN([stim_par, sac_on_full_mult, mult_par], ffnet_out=2,
#     noise_dist='poisson')

sacglmmult = NDN.NDN([stim_par, sac_on_full_mult, sac_on_full, mult_par, add_par], ffnet_out=4,
    noise_dist='poisson')

sacglmmult.time_spread = sacglmadd.time_spread
# inititialize stimulus
sacglmmult.networks[0].layers[0].weights = deepcopy(glmShare0.networks[0].layers[0].weights)
sacglmmult.networks[0].layers[1].weights = deepcopy(glmShare0.networks[0].layers[1].weights)
sacglmmult.networks[0].layers[0].biases = deepcopy(glmShare0.networks[0].layers[0].biases)
sacglmmult.networks[0].layers[1].biases = deepcopy(glmShare0.networks[0].layers[1].biases)

sacglmmult.networks[1].layers[1].weights[0,:] = 1.0
sacglmmult.networks[1].layers[1].weights[1:,:] = 0.0

# # initialize additive saccade kernels
# sacglmmult.networks[1].layers[0].weights = deepcopy(sacglmadd.networks[1].layers[0].weights)
# sacglmmult.networks[1].layers[1].weights = deepcopy(sacglmadd.networks[1].layers[1].weights)

# # initialize multiplicative saccade kernels (with the additive-only ones)
sacglmmult.networks[2].layers[0].weights = deepcopy(sacglmadd.networks[1].layers[0].weights)
sacglmmult.networks[2].layers[1].weights = deepcopy(sacglmadd.networks[1].layers[1].weights)

# specify variables to fit
v2f0 = sacglmmult.fit_variables(layers_to_skip=[0,1,2], fit_biases=False)
v2f0[-1][0]['biases']=True # fit last layer biases
# v2f0[1][1]['weights']=False
# v2f0[2][1]['weights']=False
# v2f0[2][0]['weights']=False


#%% train
_ = sacglmmult.train(
    input_data=[Xstim, sacshift], output_data=Robs, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)

w1m = sacglmmult.networks[1].layers[0].weights
plt.plot(w1m)

# fit causal model (identical, but the saccade times are not shifted)
# sacglmmultcaus = sacglmmult.copy_model()
# _ = sacglmmultcaus.train(
#     input_data=[Xstim, sacon], output_data=Robs, data_filters=DF,
#     train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
#     learning_alg='adam', opt_params=adam_params
# )

#%% train 2, free variables
v2f1 = sacglmmult.fit_variables(fit_biases=False)
v2f1[-1][0]['biases']=True # fit last layer biases
_ = sacglmmult.train(
    input_data=[Xstim, sacshift], output_data=Robs, data_filters=DF,
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

#%%
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

# %% reorder NDN

def reorder_by_layer
