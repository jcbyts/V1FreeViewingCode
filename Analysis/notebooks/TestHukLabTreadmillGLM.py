#%%
import warnings

from numpy.lib.type_check import _nan_to_num_dispatcher; warnings.simplefilter('ignore')

import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import V1FreeViewingCode.Analysis.notebooks.Utils as U
import V1FreeViewingCode.Analysis.notebooks.gratings as gt


import NDN3.NDNutils as NDNutils

which_gpu = NDNutils.assign_gpu()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

from scipy.ndimage import gaussian_filter
from copy import deepcopy

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt  # plotting
import seaborn as sns

import NDN3.NDN as NDN
import NDN3.Utils.DanUtils as DU
# %% Load Data
import importlib
import scipy.io as sio

fname = "/home/jake/Data/Datasets/HuklabTreadmill/processed/gru_20210525.mat"

matdat = sio.loadmat(fname)

print(matdat.keys())

plt.figure(figsize=(10,5))
plt.plot(matdat['treadTime'], matdat['treadSpeed'])
# %%

def resampleAtTimes(oldtimes, X, newtimes, sm=10):
    from scipy.ndimage import convolve1d
    from scipy.interpolate import interp1d

    kern = np.hanning(sm)   # a Hanning window with width 50
    kern /= kern.sum()      # normalize the kernel weights to sum to 1
    
    fs = 1//np.median(np.diff(oldtimes)) # sampling rate
    
    # get velocity by smoothing the derivative
    x = convolve1d(X, kern, axis=0)
    
    # resample at new times
    f = interp1d(oldtimes,x,kind='linear', axis=0, fill_value='extrapolate')
    return f(newtimes)



binSize = 10
startTime = np.min(matdat['spikeTimes'])
stopTime = np.max(matdat['spikeTimes'])
newtimes = np.arange(startTime, stopTime, binSize/1000)

#%%
# bin spikes
Robs = gt.bin_spike_times(matdat['spikeTimes'], matdat['spikeIds'], cids=np.unique(matdat['spikeIds']), bin_size=binSize, start_time=startTime, stop_time=stopTime ).T
NT = Robs.shape[0]


Robs = gaussian_filter1d(Robs, 4, axis=0)

#%%
treadSpd = resampleAtTimes(matdat['treadTime'].flatten(), matdat['treadSpeed'].flatten(), newtimes)
sacBoxcar = resampleAtTimes(matdat['eyeTime'].flatten(), (matdat['eyeLabels'].flatten()==2).astype('float32'), newtimes)

onsets = np.digitize(matdat['GratingOnsets'], newtimes)
offsets = np.digitize(matdat['GratingOffsets'], newtimes)

directions = np.unique(matdat['GratingDirections'])
ND = len(directions)
gratingDir = np.zeros( (NT, ND))
gratingCon = np.zeros( (NT, 1) )
Ntrials = len(onsets)
for iTrial in range(Ntrials):
    thcol = np.where(directions==matdat['GratingDirections'][iTrial][0])[0]
    inds = np.arange(onsets[iTrial], offsets[iTrial], 1)
    gratingDir[inds, thcol] = 1
    gratingCon[inds] = 1


sacon = np.append(0, np.diff(sacBoxcar, axis=0)==1).astype('float32')
sacoff = np.append(0, np.diff(sacBoxcar, axis=0)==-1).astype('float32')
sacon = np.expand_dims(sacon, axis=1)
sacoff = np.expand_dims(sacoff, axis=1)

#%% setup optimizer params
adam_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')

early_stopping = 100

adam_params['batch_size'] = 1000
adam_params['display'] = 30
adam_params['MAPest'] = True
adam_params['epochs_training'] = 1000
adam_params['early_stop'] = early_stopping
adam_params['early_stop_mode'] = 1
adam_params['epsilon'] = 1e-8
adam_params['data_pipe_type'] = 'data_as_var' # 'feed_dict'
adam_params['learning_rate'] = 1e-3

lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_params['maxiter'] = 10000

noise_dist = 'poisson'
seed = 5
optimizer = 'lbfgs'


if noise_dist=='poisson':
    null_adjusted = True
else:
    null_adjusted = False

if optimizer=='adam':
    opt_params = adam_params
else:
    opt_params = lbfgs_params

# %%
num_tkerns = 3
num_lags = 15
num_sactkerns = 4

back_shiftson = 20
tspacing = list(np.concatenate([np.arange(0,back_shiftson,5), np.arange(back_shiftson,back_shiftson+10,3), np.arange(back_shiftson+10,back_shiftson+40,5)]))

# saccade onset/offset with back_shifts
saconshift = NDNutils.shift_mat_zpad(sacon,-back_shiftson,dim=0)
num_onlags = max(tspacing)+1


Xdir = NDNutils.create_time_embedding(gratingDir, [num_lags, ND])
Xcon = NDNutils.create_time_embedding(gratingCon, [num_lags, 1])

Ui, Xi = NDNutils.generate_xv_folds(NT)
NC = Robs.shape[1]

Time = NDNutils.tent_basis_generate(xs=np.linspace(0, NT-1, 20))

dc_shift = NDNutils.ffnetwork_params(
    input_dims=[1,Time.shape[1]],
    layer_sizes=[NC],
    layer_types=['normal'], # readout for cell-specific regularization
    act_funcs=['lin'],
    normalization=[0],
    reg_list={'d2x':[1e-3], 'l2':[1e-4]}
)

dir_tuning_par = NDNutils.ffnetwork_params(
    input_dims=[num_lags,ND],
    xstim_n=[1],
    layer_sizes=[10, NC],
    layer_types=['normal', 'normal'], # readout for cell-specific regularization
    act_funcs=['lin', 'lin'],
    normalization=[1, 0],
    reg_list={'d2x': [1e-5], 'orth':[0], 'l2':[1e-6, 1e-3]}
)

con_onset_par = NDNutils.ffnetwork_params(
    input_dims=[num_lags,1],
    xstim_n=[1],
    layer_sizes=[NC],
    layer_types=['normal'], # readout for cell-specific regularization
    act_funcs=['lin'],
    normalization=[0],
    reg_list={'d2t': [5e-3], 'l2':[1e-2]}
)

sac_on_par = NDNutils.ffnetwork_params(
    input_dims=[1,1,1],
    time_expand=[num_onlags],
    xstim_n=[2],
    layer_sizes=[num_sactkerns, NC], # conv_filter_widths=[1],
    layer_types=['temporal', 'normal'],
    act_funcs=['lin', 'lin'],
    normalization=[1, 0],
    reg_list={'orth':[None,None], 'd2t':[1e-1],'d2x':[None, None],'l2':[None], 'l1':[None]})

base_glm_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus'])  

stim_glm_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus'])

sac_glm_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1,2], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus'])


#%% Base Model

baseglm = NDN.NDN([dc_shift, con_onset_par, base_glm_par], tf_seed=seed, noise_dist=noise_dist)

v2f0 = baseglm.fit_variables(layers_to_skip=[[], [], [0]], fit_biases=False)
v2f0[-1][-1]['biases']=True

_ = baseglm.train(input_data=[Time, Xcon], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f0)

#%% Find best Regularization
reg_results = DU.unit_reg_test(baseglm, input_data=[Time, Xcon],
    output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    reg_type='d2t', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_targets=[0], ffnet_targets=[1],
    learning_alg='lbfgs', opt_params=lbfgs_params)

baseglm = DU.unit_assign_reg(baseglm, reg_results)

#%%
f = plt.plot(baseglm.networks[1].layers[0].weights)

plt.figure(figsize=(10,5))
f = plt.plot(Time@baseglm.networks[0].layers[0].weights)


#%% Stim GLM

stimglm = NDN.NDN([dc_shift, dir_tuning_par, stim_glm_par], tf_seed=seed, noise_dist=noise_dist)

v2f1 = stimglm.fit_variables(layers_to_skip=[[], [], [0]], fit_biases=False)
v2f1[-1][-1]['biases']=True

# stimglm.networks[0].layers[0].weights = deepcopy(baseglm.networks[0].layers[0].weights)
# stimglm.networks[3].layers[0].weights = deepcopy(baseglm.networks[1].layers[0].weights)

_ = stimglm.train(input_data=[Time, Xdir], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f1)

#%%
reg_results = DU.unit_reg_test(stimglm, input_data=[Time, Xdir],
    output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    reg_type='d2t', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_targets=[0], ffnet_targets=[1],
    learning_alg='lbfgs', opt_params=lbfgs_params)

stimglm = DU.unit_assign_reg(stimglm, reg_results)

#%%


sacglm = NDN.NDN([dc_shift, dir_tuning_par, sac_on_par, sac_glm_par], tf_seed=seed, noise_dist=noise_dist)
sacglm.networks[2].layers[0].init_temporal_basis( xs=tspacing )

plt.figure(figsize=(10,5))
f = plt.plot(sacglm.networks[2].layers[0].filter_basis)
plt.title("Sac Onset Basis")

v2f2 = sacglm.fit_variables(layers_to_skip=[[], [], [], [0]], fit_biases=False)
v2f2[-1][-1]['biases']=True

# sacglm.networks[0].layers[0].weights = deepcopy(stimglm.networks[0].layers[0].weights)
# sacglm.networks[1].layers[0].weights = deepcopy(stimglm.networks[1].layers[0].weights)
# sacglm.networks[1].layers[1].weights = deepcopy(stimglm.networks[1].layers[1].weights)
# sacglm.networks[2].layers[0].weights = deepcopy(stimglm.networks[2].layers[0].weights)
# sacglm.networks[3].layers[0].weights = deepcopy(stimglm.networks[3].layers[0].weights)

_ = sacglm.train(input_data=[Time, Xdir, saconshift], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f2)

# reg_results = DU.unit_reg_test(sacglm, input_data=[Time, Xcon, gratingDir, saconshift],
#     output_data=Robs, train_indxs=Ui, test_indxs=Xi,
#     reg_type='d2t', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
#     layer_targets=[0], ffnet_targets=[2],
#     learning_alg='lbfgs', opt_params=lbfgs_params)

# sacglm = DU.unit_assign_reg(sacglm, reg_results)

# %%
def getDirWeights(glm, ffnet=1):

    TC = filters.squeeze()
    if len(glm.networks[ffnet].layers)>1:
        w = glm.networks[ffnet].layers[0].weights@glm.networks[ffnet].layers[1].weights
        num_lags = glm.networks[ffnet].input_dims[0]
        sp_dims = glm.networks[ffnet].input_dims[1:]
        nfilt = glm.networks[ffnet].layers[-1].weights.shape[-1]
        w2 = np.reshape(w, [np.prod(sp_dims), num_lags, NC])
        TC = np.transpose(w2, [1,0,2])
        
    return TC

TC = getDirWeights(stimglm)

sx = np.ceil(np.sqrt(NC))
sy = np.round(np.sqrt(NC))
win = [-4, 150]

plt.figure(figsize=(sx*2, sy*2))
for cc in range(NC):
    plt.subplot(sx, sy, cc+1)
    plt.imshow(TC[:,:,cc])


#%%
filters = DU.tbasis_recover_filters(sacglm, ffnet=2).squeeze()
f = plt.plot(filters)


# %%

def get_psth(robs, gratings, win):
    lags = np.arange(win[0], win[1])
    nlags = len(lags)
    ngratings = gratings.shape[1]
    psth = np.zeros( (nlags, ngratings))
    for i in range(ngratings):
        gonsets = np.where(np.diff(gratings[:,i])==1)[0]
        psth[:,i] = gt.psth(robs, gonsets.flatten(), win[0], win[1])[0]
    
    return psth, lags


yhat0 = baseglm.generate_prediction(input_data=[Time, Xcon])
yhat1 = stimglm.generate_prediction(input_data=[Time, Xdir])
yhat2 = sacglm.generate_prediction(input_data=[Time, Xdir, saconshift])

#%%
cc = 0

from scipy.ndimage import gaussian_filter1d
cc += 1

iix = np.arange(0, 1000)+5000
plt.plot(gaussian_filter1d(Robs[iix,cc], 10))
plt.plot(yhat0[iix,cc])
plt.plot(yhat2[iix,cc])


#%%

ND
TC = np.zeros((ND, NC))
TC1 = np.zeros((ND, NC))
TC2 = np.zeros((ND, NC))
sx = np.ceil(np.sqrt(NC))
sy = np.round(np.sqrt(NC))
win = [-4, 150]

plt.figure(figsize=(sx*2, sy*2))
for cc in range(NC):
    
    r = get_psth(Robs[:,cc], gratingDir, win)
    rhat0 = get_psth(yhat0[:,cc], gratingDir, win)
    rhat1 = get_psth(yhat1[:,cc], gratingDir, win)
    TC[:,cc] = np.sum(r[0], axis=0)
    TC1[:,cc] = np.sum(rhat0[0], axis=0)
    TC2[:,cc] = np.sum(rhat1[0], axis=0)

    plt.subplot(sx, sy, cc+1)
    plt.plot(directions, TC[:,cc], 'k-o')
    plt.plot(directions, TC1[:,cc], 'r')
    plt.plot(directions, TC2[:,cc], 'b')
    plt.xticks(np.arange(0, 360,90))
    sns.despine(offset=0, trim=True)


plt.figure()
r2Base = U.r_squared(TC, TC1)
r2Dir = U.r_squared(TC, TC2)
plt.plot(r2Base, r2Dir, 'o')
plt.xlim((0,1))
plt.ylim((0,1))
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.title("R-squared Tuning Curve")

r2Base = U.r_squared(Robs[Xi,:], yhat0[Xi,:])
r2Dir = U.r_squared(Robs[Xi,:], yhat1[Xi,:])
plt.figure()
plt.plot(r2Base, r2Dir, 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel("Var Explained (Base)")
plt.ylabel("Var Explained (Dir)")


r2Sac = U.r_squared(Robs[Xi,:], yhat2[Xi,:])
plt.figure()
plt.plot(r2Dir, r2Sac, 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel("Var Explained (Dir)")
plt.ylabel("Var Explained (Sac)")


#%%
cc += 1
if cc >= NC:
    cc = 0

win = [-10, 200]
r = get_psth(Robs[:,cc], gratingDir, win)
rhat0 = get_psth(yhat0[:,cc], gratingDir, win)
rhat1 = get_psth(yhat1[:,cc], gratingDir, win)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(r[0].T, aspect='auto', extent=[win[0], win[1], 0, 360])
plt.ylabel("Direction")
plt.title("Data Cell %d" %cc)

plt.subplot(1,3,2)
plt.imshow(rhat0[0].T, aspect='auto', extent=[win[0], win[1], 0, 360])
plt.xlabel("Time from Grating Onset (bin)")
plt.title("GLM no Direction Tuning")

plt.subplot(1,3,3)
plt.imshow(rhat1[0].T, aspect='auto', extent=[win[0], win[1], 0, 360])
plt.title("GLM with Direction Tuning")

plt.figure()
f = plt.plot(r[0])
plt.title("Cell %d" %cc)

#%% likelihoods over time

LLt0 = Robs*np.log(yhat0) - yhat0
LLt1 = Robs*np.log(yhat1) - yhat1
LLt2 = Robs*np.log(yhat2) - yhat2

#%%
for cc in range(NC):
    rll0 = get_psth(LLt0[:,cc], gratingDir, win)
    rll1 = get_psth(LLt1[:,cc], gratingDir, win)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(rll1[0].T-rll0[0].T, aspect='auto', extent=[win[0], win[1], 0, 360])
    plt.subplot(1,2,2)
    f = plt.plot(rll1[0]-rll0[0])
    plt.title("Cell %d" %cc)

#%%
LL0 = baseglm.eval_models(input_data=[Time, Xcon], output_data=Robs, data_indxs=Xi, nulladjusted=True)
LL1 = stimglm.eval_models(input_data=[Time, Xdir], output_data=Robs, data_indxs=Xi, nulladjusted=True)
LL2 = sacglm.eval_models(input_data=[Time, Xdir, saconshift], output_data=Robs, data_indxs=Xi, nulladjusted=True)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(LL0, 'o-')
plt.plot(LL1, 'o-')
plt.plot(LL2, 'o-')

plt.subplot(1,3,2)
plt.plot(LL0, LL1, 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')

for i in range(NC):
    plt.text(LL0[i], LL1[i], "%d" %(i))

plt.xlabel("Base Model")
plt.ylabel("Base + Direction Tuning")

plt.subplot(1,3,3)
plt.plot(LL1, LL2, 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')

for i in range(NC):
    plt.text(LL1[i], LL2[i], "%d" %(i))

plt.xlabel("Dir Model")
plt.ylabel("Dir + Sac")
# plt.text(LL0[0], LL1[0], "1")

# plt.text([ll for ll in LL0], [ll for ll in LL1], ["%d" %(cc+1) for cc in range(NC)])
# %%
