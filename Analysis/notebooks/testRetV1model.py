#%% Import
import warnings; warnings.simplefilter('ignore')

import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import Utils as U
import gratings as gt


import NDN3.NDNutils as NDNutils

which_gpu = NDNutils.assign_gpu()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import tensorflow as tf

import neureye as ne
import NDN3.NDN as NDN
import NDN3.Utils.DanUtils as DU

output_dir = '/home/jake/Data/tensorboard/tensorboard' + str(which_gpu)
print(output_dir)

import numpy as np
from scipy.ndimage import gaussian_filter
from copy import deepcopy
import matplotlib.pyplot as plt  # plotting
import seaborn as sns

#%% data paths
dirname = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'

flist = ['logan_20200304_Gabor_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200304_Grating_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200304_FixRsvpStim_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200304_BackImage_-20_-10_40_60_2_2_0_9_0.mat']


import importlib
importlib.reload(ne)

# %% Load data
Stim,Robs,valdat,labels,NX,NY,dt,eyeAtFrame,frameTime = ne.load_stim_files(dirname=dirname, flist=flist[0:3], corrected=True)

Stim /= np.nanstd(Stim) # normalize stimulus (necessary?)

#%% get valid indices

valdata = np.intersect1d(np.where(valdat[:,0] == 1)[0], np.where(labels[:,0] == 1)[0]) # fixations / valid

# valdata = np.where(valdat[:,0] == 1)[0]
valid_eye_rad = 5.2  # degrees -- use this when looking at eye-calibration (see below)
ppd = 37.50476617061

eyeX = (eyeAtFrame[:,0]-640)/ppd
eyeY = (eyeAtFrame[:,1]-380)/ppd

eyeCentered = np.hypot(eyeX, eyeY) < valid_eye_rad
# eyeCentered = np.logical_and(eyeX < 0, eyeCentered)
valdata = np.intersect1d(valdata, np.where(eyeCentered)[0])

#%%
import scipy.io as sio

subjstr = flist[0].split('_')
outname = subjstr[0] + '_' + subjstr[1] + '_eyetraces.mat'

dat = sio.loadmat(dirname + outname)

# %%
nt = len(frameTime)
xshift = deepcopy(dat['eyeShift'][:nt,0])
yshift = deepcopy(dat['eyeShift'][:nt,1])

assert np.all(frameTime==dat['frameTime'][:nt]), "error: frame time doesn't match"
ix = np.logical_or(np.isnan(xshift), np.isnan(yshift))

xshift[ix] = 0.0
yshift[ix] = 0.0

StimC = ne.shift_stim(Stim, xshift, yshift, [NX,NY])

#%% Make stim
num_lags = 10
cids = np.arange(0, Robs.shape[1])
Xstim0, _ = ne.create_time_embedding_valid(Stim, [num_lags, NX, NY], valdata)
Xstim, rinds = ne.create_time_embedding_valid(StimC, [num_lags, NX, NY], valdata)
Rvalid = deepcopy(Robs[rinds,:])
Rvalid = Rvalid[:,cids]
NC = Rvalid.shape[1]
#%% check STAS
stas = Xstim.T@Rvalid / np.sum(Rvalid, axis=0)
stas = np.reshape(stas, (NX*NY, num_lags, NC))
stas0 = Xstim0.T@Rvalid / np.sum(Rvalid, axis=0)
stas0 = np.reshape(stas0, (NX*NY, num_lags, NC))

#%%


plt.figure(figsize=(10,10))
sx,sy = U.get_subplot_dims(NC)
sumdensity = 0
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    
    f = plt.plot(stas[:,:,cc])
    f = plt.plot(stas0[:,:,cc], color='k')
    # bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    # plt.imshow(np.reshape(stas[:,bestlag,cc], (NY,NX)))
    # sumdensity += np.abs(stas[:,bestlag,cc])
    plt.title(cc)
    plt.axis("off")

#%% get training indices
NT = Xstim.shape[0]
Ui, Xi = NDNutils.generate_xv_folds(NT)

adam_params = U.def_adam_params()

adam_params
#%% Make model
num_lags = 10
NC = Rvalid.shape[1]

Greg0 = 1e-1
Greg = 1e-1
Creg0 = 1
Creg = 1e-2
Mreg0 = 1e-3
Mreg = 1e-1
L1reg0 = 1e-5
Xreg = 1e-2

num_tkern = 2
num_subs = 10

ndn_par = NDNutils.ffnetwork_params(
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[num_tkern, num_subs, NC],
    layer_types=['conv', 'conv', 'normal'],
    conv_filter_widths=[1,12,None],ei_layers=[None,num_subs//2],
    normalization=[2,1,-1],act_funcs=['relu', 'relu', 'lin'],
    verbose=True,
    reg_list={'d2t':[1e-4], 'd2x':[None, Xreg], 'l1':[None, L1reg0],
    'center':[None, Creg0], 'glocal':[None, Greg0], 'max':[None, None, Mreg0]}
)

auto_par = NDNutils.ffnetwork_params(input_dims=[1, NC, 1],
                xstim_n=[1],
                layer_sizes=[2, 1, NC],
                time_expand=[0, 10, 0],
                layer_types=['normal', 'temporal', 'normal'],
                conv_filter_widths=[None, 1, None],
                act_funcs=['lin', 'lin', 'lin'],
                normalization=[1, 1, -1],
                reg_list={'d2t':[None, 1e-3, None]}
                )

add_par = NDNutils.ffnetwork_params(
                xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
                layer_types=['add'], act_funcs=['softplus']
                )

retV1b = NDN.NDN( [ndn_par, auto_par, add_par], ffnet_out=2, noise_dist='poisson' )

# set output regularization on the latent
retV1b.batch_size = adam_params['batch_size']
retV1b.initialize_output_reg(network_target=1,layer_target=1, reg_vals={'d2t': .1})

retV1b.networks[0].layers[0].weights[:,:] = 0
retV1b.networks[0].layers[0].weights[2:4,0] = 1
retV1b.networks[0].layers[0].weights[2:4,1] = -1

v2fb = retV1b.fit_variables(fit_biases=True)
for nn in range(len(retV1b.networks)):
    for nl in range(len(retV1b.networks[nn].layers)):
        if retV1b.networks[nn].layers[nl].act_func=='lin':
            v2fb[nn][nl]['biases'] = False

v2fb[-1][-1]['weights'] = False
v2fb[-1][-1]['biases'] = True
# stim only
retV1 = NDN.NDN( [ndn_par], ffnet_out=0, noise_dist='poisson')
retV1.networks[0].layers[0].weights[:,:] = 0
retV1.networks[0].layers[0].weights[2:4,0] = 1
retV1.networks[0].layers[0].weights[2:4,1] = -1

v2f = retV1.fit_variables(fit_biases=True)
v2f[0][0]['biases'] = True
v2f[-1][0]['biases'] = True

#%% train
_ = retV1b.train(input_data=[Xstim, Rvalid], output_data=Rvalid, train_indxs=Ui,
    test_indxs=Xi, silent=False, learning_alg='adam', opt_params=adam_params, fit_variables=v2fb)
# %% fit
DU.plot_3dfilters(retV1b)
#%%
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(retV1b.networks[0].layers[0].weights)
plt.title("Temporal Kernels")
plt.subplot(1,2,2)
plt.plot(retV1b.networks[1].layers[1].weights)
plt.title("Latent temporal kernel")
#%% get test likelihood
LLx1 = retV1b.eval_models(input_data=[Xstim, Rvalid], output_data=Rvalid, data_indxs=Xi, nulladjusted=True, use_gpu=False)

plt.figure()
plt.hist(LLx1)

#%% generate predictions at different levels
# stimulus model
stimgen = retV1b.generate_prediction(input_data=[Xstim, Rvalid], ffnet_target=0, layer_target=-1)
latents = retV1b.generate_prediction(input_data=[Xstim, Rvalid], ffnet_target=1, layer_target=1)
rpred = retV1b.generate_prediction(input_data=[Xstim, Rvalid])
#%%
plt.figure(figsize=(10,4))
plt.plot(stimgen[:200,:])

plt.figure(figsize=(10,4))
plt.plot(latents[:200,:])
i = 200
#%%
cc = 7
plt.figure(figsize=(10,4))
i += 1000
inds = np.arange(i, i+200)
plt.plot(Rvalid[inds,cc])
plt.plot(rpred[inds,cc])
plt.ylim([0,1])

plt.figure(figsize=(10,4))
plt.plot(stimgen[inds,cc])

plt.figure(figsize=(10,4))
plt.plot(latents[inds,:])
plt.plot(labels[inds])
#%%
# v2f[0][0]['biases'] = True


_ = retV1.train(input_data=Xstim, output_data=Rvalid, train_indxs=Ui,
    test_indxs=Xi, silent=False, learning_alg='adam', opt_params=adam_params, fit_variables=v2f)

#%%
DU.plot_3dfilters(retV1)

plt.figure()
LLx0 = retV1.eval_models(input_data=Xstim, output_data=Rvalid, data_indxs=Xi, nulladjusted=True, use_gpu=False)

plt.hist(LLx0)

#%%
plt.plot(LLx0, LLx1, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')

#%%
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(retV1.networks[0].layers[0].weights)
plt.subplot(1,2,2)
plt.plot(retV1b.networks[0].layers[0].weights)

# %%

retV1.set_regularization('center', Creg, layer_target=1)
retV1.set_regularization('glocal', .1, layer_target=1)
retV1.set_regularization('glocal', .01, layer_target=0)
retV1.set_regularization('max', Mreg, layer_target=3)
retV1.set_regularization('d2x', 1e-2, layer_target=1)

retV1b = retV1.copy_model()

_ = retV1b.train(input_data=Xstim, output_data=Rvalid, train_indxs=Ui,
    test_indxs=Xi, silent=False, learning_alg='adam', opt_params=adam_params, fit_variables=v2f)
# %%
DU.plot_3dfilters(retV1b)

LLx1 = retV1b.eval_models(input_data=[Xstim, Rvalid], output_data=Rvalid, data_indxs=Xi, nulladjusted=True, use_gpu=False)
LLx2 = side2c.eval_models(input_data=Xstim2, output_data=Rvalid, data_indxs=Xi, nulladjusted=True, use_gpu=False)

#%%
plt.figure()
plt.plot(LLx1, LLx2, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k--')
# %%
plt.figure()

w = retV1b.networks[0].layers[0].weights
b = retV1b.networks[0].layers[0].biases

# wn = np.linalg.norm(w, 2, axis=0)
# plt.plot(w/wn)


plt.axvline(b[0][0])
plt.axvline(b[0][1])

# %%
sz = retV1b.networks[0].layers[2].weights.shape[0]
I = np.reshape(retV1b.networks[0].layers[2].weights[:,0], [sz//num_subs, num_subs])
plt.figure(figsize=(10,4))
for i in range(num_subs):
    plt.subplot(1,num_subs,i+1)
    plt.imshow(np.reshape(I[:,i], [NX, NY]), aspect='auto')
# DU.plot_2dweights()

