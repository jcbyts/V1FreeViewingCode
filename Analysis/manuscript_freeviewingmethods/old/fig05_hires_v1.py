#%%
import warnings; warnings.simplefilter('ignore')

import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import V1FreeViewingCode.Analysis.notebooks.Utils as U
import V1FreeViewingCode.Analysis.notebooks.gratings as gt


import NDN3.NDNutils as NDNutils

which_gpu = NDNutils.assign_gpu()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import tensorflow as tf

import V1FreeViewingCode.Analysis.notebooks.neureye as ne
import NDN3.NDN as NDN
import NDN3.Utils.DanUtils as DU

output_dir = '/home/jake/Data/tensorboard/tensorboard' + str(which_gpu)
print(output_dir)

import numpy as np
from scipy.ndimage import gaussian_filter
from copy import deepcopy
import matplotlib.pyplot as plt  # plotting
import seaborn as sns

# %% set paths for data


dirname = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'


stim_list = {
    '20200304':
    ['logan_20200304_Gabor_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200304_Grating_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200304_FixRsvpStim_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200304_BackImage_-20_-10_40_60_2_2_0_9_0.mat'],
    '20200306':
    ['logan_20200306_Gabor_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200306_Grating_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200306_FixRsvpStim_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200306_BackImage_-20_-10_40_60_2_2_0_9_0.mat']
}


flist = stim_list['20200304'][:2] # only take gabor and grating file

import importlib
importlib.reload(ne)


# %% Load Data

Stim,Robs,valdat,labels,NX,NY,dt,eyeAtFrame,frameTime = ne.load_stim_files(dirname=dirname, flist=flist, corrected=False)

Stim /= np.nanstd(Stim) # normalize stimulus (necessary?)

# %% get valid indices
valdata = np.intersect1d(np.where(valdat[:,0] == 1)[0], np.where(labels[:,0] == 1)[0]) # fixations / valid

valid_eye_rad = 5.2  # degrees -- use this when looking at eye-calibration (see below)
ppd = 37.50476617061

eyeX = (eyeAtFrame[:,0]-640)/ppd
eyeY = (eyeAtFrame[:,1]-380)/ppd

eyeCentered = np.hypot(eyeX, eyeY) < valid_eye_rad
# eyeCentered = np.logical_and(eyeX < 0, eyeCentered)
valid_inds = np.intersect1d(valdata, np.where(eyeCentered)[0])

# %% quick check STAS
stas = ne.get_stas(Stim, Robs, [NX,NY], valid=valid_inds, num_lags=10, plot=False)

plt.figure(figsize=(10,10))
NC = Robs.shape[1]
sx,sy = U.get_subplot_dims(NC)
sumdensity = 0
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    plt.imshow(np.reshape(stas[:,bestlag,cc], (NY,NX)))
    sumdensity += np.abs(stas[:,bestlag,cc])
    plt.title(cc)
    plt.axis("off")
    
plt.figure()
plt.imshow(np.reshape(sumdensity, (NY, NX)))

#%%
# Cxinds = ne.crop_indx(NX, range(9,24), range(9,24))
Cxinds = ne.crop_indx(NX, range(5,25), range(5,25))
# Cxinds = ne.crop_indx(NX, range(20,44), range(20,44))
NX2 = np.sqrt(len(Cxinds)).astype(int)
plt.figure(figsize=(10,10))
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    sta = stas[:,bestlag,cc]
    plt.imshow(np.reshape(sta[Cxinds], [NX2,NX2]))

# %% prepare stimulus for model
Xstim, Rvalid, dims, CXinds, cids = ne.prep_stim_model(Stim, Robs, [NX,NY],
    valid=valid_inds,
    num_lags=10,
    plot=True,
    Cindx=Cxinds,
    )

#%% fit GQM
NT = Xstim.shape[0]
Ui, Xi = NDNutils.generate_xv_folds(NT)

# optimizer parameters
adam_params = U.def_adam_params()

# d2ts = 1e-4*10**np.arange(0, 5)

d2xs = 1e-2*10**np.arange(0, 5)
gqms = []
LLxs = []
for step in range(len(d2xs)):

    d2t = .05
    d2x = d2xs[step]
    loc = 1e-5

    num_lags = dims[0]
    NX2 = dims[1]
    NC = Rvalid.shape[1]
    # NDN parameters for processing the stimulus
    lin = NDNutils.ffnetwork_params( 
        input_dims=[1,NX2,NX2,num_lags],
        layer_sizes=[NC],
        layer_types=['readout'], normalization=[0],
        act_funcs=['lin'], verbose=True,
        reg_list={'d2t': [d2t], 'd2x':[d2x], 'l1':[loc]})

    quad = NDNutils.ffnetwork_params( 
        input_dims=[1,NX2,NX2,num_lags],
        layer_sizes=[NC],
        layer_types=['readout'], normalization=[0],
        act_funcs=['quad'], verbose=True,
        reg_list={'d2t': [d2t], 'd2x':[d2x], 'l1':[loc]})

    quad = NDNutils.ffnetwork_params( 
        input_dims=[1,NX2,NX2,num_lags],
        layer_sizes=[NC],
        layer_types=['readout'], normalization=[0],
        act_funcs=['quad'], verbose=True,
        reg_list={'d2t': [d2t], 'd2x':[d2x], 'l1':[loc]})

    add_par = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1,2], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus'])

    # initialize GLM
    gqm0 = NDN.NDN([lin, quad, quad, add_par],  noise_dist='poisson')

    v2f0 = gqm0.fit_variables(fit_biases=False)
    v2f0[-1][-1]['biases'] = True

    stas = (Xstim.T @ (Rvalid-np.mean(Rvalid, axis=0))) / np.sum(Rvalid, axis=0)
    stas /= np.sum(stas,axis=0)
    gqm0.networks[0].layers[0].weights[:] = deepcopy(stas[:])

    # train initial model
    _ = gqm0.train(input_data=[Xstim], output_data=Rvalid,
        train_indxs=Ui, test_indxs=Xi,
        learning_alg='adam', opt_params=adam_params,
         fit_variables=v2f0)

    LLx = gqm0.eval_models(input_data=Xstim, output_data=Rvalid, data_indxs=Xi, nulladjusted=True)     

    gqms.append(gqm0)
    LLxs.append(LLx)         

#%% pick best regularization
bestreg = np.zeros(NC)
reg_path = np.asarray(LLxs)
for cc in range(NC):
    bestind = np.argmax(reg_path[:,cc])
    bestreg[cc] = d2xs[bestind]
plt.figure()
f = plt.plot(reg_path)
#%% refit with best regularization

# initialize GQM
gqm0 = NDN.NDN([lin, quad, quad, add_par],  noise_dist='poisson')

v2f0 = gqm0.fit_variables(fit_biases=False)
v2f0[-1][-1]['biases'] = True

stas = (Xstim.T @ (Rvalid-np.mean(Rvalid, axis=0))) / np.sum(Rvalid, axis=0)
stas /= np.sum(stas,axis=0)
gqm0.networks[0].layers[0].weights[:] = deepcopy(stas[:])
gqm0.set_regularization('d2x', reg_val=bestreg, ffnet_target=0)
gqm0.set_regularization('d2x', reg_val=bestreg, ffnet_target=1)
gqm0.set_regularization('d2x', reg_val=bestreg, ffnet_target=2)

# train initial model
_ = gqm0.train(input_data=[Xstim], output_data=Rvalid,
    train_indxs=Ui, test_indxs=Xi,
    learning_alg='adam', opt_params=adam_params,
     fit_variables=v2f0)
# f= plt.plot(np.asarray(LLxs))
#%% plot model
DU.plot_3dfilters(gqm0, ffnet=0)

DU.plot_3dfilters(gqm0, ffnet=1)

DU.plot_3dfilters(gqm0, ffnet=2)

#%% plot fit
LLx = gqm0.eval_models(input_data=Xstim, output_data=Rvalid, data_indxs=Xi, nulladjusted=True)
plt.figure()
plt.plot(LLx, '-o')
plt.axhline(0, color='k')

#%% Run eye correction
# Run it all once
eyeAtFrameCentered = (eyeAtFrame-(640, 380))
centers5, locs, LLspace1 = ne.get_corr_grid(gqm0, Stim, Robs, [NX,NY], cids,
        eyeAtFrameCentered, valdata, valid_eye_rad=5.2*ppd, Erad=0.5*ppd, Npos=25,
        crop_edge=0, plot=True, interpolation_steps=0, softmax=1000, autoencoder=False)

#%% plot one LL surface
ctrid = np.argmin(locs**2)
idxs = np.arange(2, 14)
nplot = len(idxs)

# plt.figure()
# I2 = np.mean(LLspace1.reshape((-1, 24, 29)), axis=0)
# I2 = 1-ne.normalize_range(I2)
# plt.imshow(I2)

plt.figure(figsize=(15,6))
for i,j in zip(idxs, range(nplot)):
    plt.subplot(1,nplot,j+1)
    irow = 10
    I = LLspace1[irow,i,:,:]
    I = 1-ne.normalize_range(I) # flip sign and normalize between 0 and 1
    Ly,Lx = I.shape

    x,y = ne.radialcenter(I**10)
    print("(%d, %d)" %(x-Lx//2,y-Ly//2))
    plt.imshow(I, aspect='equal')
    plt.axvline(Lx/2, color='w')
    plt.axhline(Ly/2, color='w')
    plt.plot(x,y, '+r')
    plt.title('(%d,%d)' %(irow,idxs[j]))

#%% Get LLsurface centers
centers5,LLspace3 = ne.get_centers_from_LLspace(LLspace1, softmax=5, plot=False, interpolation_steps=3, crop_edge=2)

plt.figure()
plt.plot(centers5[:,:,0], centers5[:,:,1], '.')

#%% Correct stimulus
centers6 = deepcopy(centers5)
# ctrid = np.argmin(locs**2)
# centers6[:,:,0]-=centers6[ctrid,ctrid,0]
# centers6[:,:,1]-=centers6[ctrid,ctrid,1]


xcorrec,ycorrec = ne.get_shifter_from_centers(centers6, locs, maxshift=1, nearest=False)

eyeAtFrameCentered = (eyeAtFrame-(640, 380))

xshift = xcorrec(eyeAtFrameCentered[:,0], eyeAtFrameCentered[:,1])
yshift = ycorrec(eyeAtFrameCentered[:,0], eyeAtFrameCentered[:,1])

ix = np.logical_or(np.isnan(xshift), np.isnan(yshift))

xshift[ix] = 0.0
yshift[ix] = 0.0

# NEAREST option shifts shifts in integer numbers of pixels
StimC = ne.shift_stim(Stim, xshift, yshift, [NX,NY], nearest=True)

#%% recompute STAs to compare progress
from copy import deepcopy
# shift eye at frame to shifted values
newEyeAtFrame = deepcopy(eyeAtFrameCentered)
newEyeAtFrame[:,0] += xshift
newEyeAtFrame[:,1] += xshift

valid = np.intersect1d(np.where(~ix)[0], valdata)
# check new STAS
stas0 = ne.get_stas(Stim, Robs, [NX,NY], valid=valid, num_lags=14, plot=False)
stas = ne.get_stas(StimC, Robs, [NX,NY], valid=valid, num_lags=14, plot=False)


plt.figure(figsize=(4,NC*2))
NC = Robs.shape[1]
# sx,sy = U.get_subplot_dims(NC)
for cc in range(NC):
    
    # plt.subplot(sx,sy,cc+1)
    bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    plt.subplot(NC, 2, 2*cc + 1)
    plt.imshow(np.reshape(stas0[:,bestlag,cc], (NY,NX)))
    plt.title(cc)
    plt.axis("off")

    plt.subplot(NC, 2, 2*cc + 2)
    plt.imshow(np.reshape(stas[:,bestlag,cc], (NY,NX)))
    plt.title(cc)
    plt.axis("off")
    

#%% Re-evaluate Log-Likelihoods to compare model performance after correction
num_lags = gqm0.input_sizes[0][-1]
NX2 = np.sqrt(len(Cxinds)).astype(int)
    
# make new cropped stimulus
Xstim0, _ = ne.create_time_embedding_valid(Stim[:,Cxinds], [num_lags, NX2, NX2], valid)
Xstim1, rinds = ne.create_time_embedding_valid(StimC[:,Cxinds], [num_lags, NX2, NX2], valid)

NT = len(rinds)
Ui, Xi = NDNutils.generate_xv_folds(NT)

Rvalid = deepcopy(Robs[rinds,:])
Rvalid = Rvalid[:,cids]
Rvalid = NDNutils.shift_mat_zpad(Rvalid,-1,dim=0) # get rid of first lag

# evaluate model before and after correction
LL0 = gqm0.eval_models(input_data=Xstim0, output_data=Rvalid, data_indxs=Xi, nulladjusted=True)
LL1 = gqm0.eval_models(input_data=Xstim1, output_data=Rvalid, data_indxs=Xi, nulladjusted=True)

# plot
plt.figure()
plt.plot(LL0, LL1, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel("No Calibration")
plt.ylabel("V1 Calibration")


#%% Build design matrix
num_lags = 14
Xstim0, rinds = ne.create_time_embedding_valid(Stim, [num_lags, NX, NY], valid_inds)
Rvalid0 = deepcopy(Robs[rinds,:])

NT0 = Rvalid0.shape[0]
Ui0, Xi = NDNutils.generate_xv_folds(NT0)

augment_steps = 0 # 0 means don't augment
Xstim = deepcopy(Xstim0)
Rvalid = deepcopy(Rvalid0)

for istep in range(augment_steps):

    NT = Rvalid.shape[0]

    # generate noise
    noise = np.random.randn(NT, Xstim0.shape[1])

    # concatenate
    Xstim = np.concatenate((Xstim, deepcopy(Xstim0[Ui0,:])+noise[Ui0,:]), axis=0)
    Rvalid = np.concatenate((Rvalid, Rvalid0[Ui0,:]), axis=0)


#%% GLM

NT = Rvalid.shape[0]
Ui = np.setdiff1d(np.arange(0, NT, 1), Xi)
# Ui = np.setdiff1d(Ui, Xi + NT0)

lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_params['maxiter'] = 10000


# NDN parameters for processing the stimulus
par = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[NC],
    layer_types=['normal'], normalization=[0],
    act_funcs=['softplus'], verbose=True,
    reg_list={'d2t': [.1], 'd2x':[1], 'local':[1]})

# initialize GLM
glm0 = NDN.NDN([par],  noise_dist='poisson')
glm1 = NDN.NDN([par],  noise_dist='poisson')

v2f0 = glm0.fit_variables(fit_biases=False)
v2f0[-1][-1]['biases'] = True

stas = (Xstim.T @ (Rvalid-np.mean(Rvalid, axis=0))) / np.sum(Rvalid, axis=0)
glm0.networks[0].layers[0].weights[:] = deepcopy(stas[:])

# train initial model
_ = glm0.train(input_data=[Xstim], output_data=Rvalid,
    train_indxs=Ui0, test_indxs=Xi,
    learning_alg='lbfgs', opt_params=lbfgs_params,
     fit_variables=v2f0)

if augment_steps==0:
    glm1 = glm0.copy_model()
else:
    _ = glm1.train(input_data=[Xstim], output_data=Rvalid,
        train_indxs=Ui, test_indxs=Xi,
        learning_alg='lbfgs', opt_params=lbfgs_params,
        fit_variables=v2f0)

#%% plot filters
DU.plot_3dfilters(glm1)
#%%

LLx0 = glm0.eval_models(input_data=[Xstim], output_data=Rvalid, 
                data_indxs=Xi, nulladjusted=True)
LLx1 = glm1.eval_models(input_data=[Xstim], output_data=Rvalid, 
                data_indxs=Xi, nulladjusted=True) 

plt.figure()
plt.plot(LLx0, '-o')
plt.plot(LLx1, '-o')
plt.axhline(0)


#%% Find best regularization

glmbest = glm1.copy_model()

[LLpath, glms] = NDNutils.reg_path(glmbest, input_data=[Xstim],
    output_data=R, train_indxs=Ui, test_indxs=Xi,
    reg_type='glocal', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_target=0, ffnet_target=0,
    learning_alg='lbfgs', opt_params=lbfgs_params, output_dir=datadir,
    silent=True)


glmbest = glms[np.argmin(LLpath)].copy_model()

[LLpath, glms] = NDNutils.reg_path(glmbest, input_data=[Xstim],
    output_data=R, train_indxs=Ui, test_indxs=Xi,
    reg_type='d2x', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_target=0, ffnet_target=0,
    learning_alg='lbfgs', opt_params=lbfgs_params, output_dir=datadir,
    silent=True)


glmbest = glms[np.argmin(LLpath)-1].copy_model()

[LLpath, glms] = NDNutils.reg_path(glmbest, input_data=[Xstim],
    output_data=R, train_indxs=Ui, test_indxs=Xi,
    reg_type='d2t', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_target=0, ffnet_target=0,
    learning_alg='lbfgs', opt_params=lbfgs_params, output_dir=datadir,
    silent=True)


glmbest = glms[np.argmin(LLpath)-1].copy_model()


DU.plot_3dfilters(glmbest)

#%% Get null-adjusted test likelihoods

LLx = glmbest.eval_models(input_data=[Xstim], output_data=R, 
                data_indxs=Xi, nulladjusted=True)

LLx0 = glm0.eval_models(input_data=[Xstim], output_data=R, 
                data_indxs=Xi, nulladjusted=True)                

plt.plot(LLx0, '-o')
plt.plot(LLx, '-o')
plt.axhline(0)

# %% get crop indices
# Cxinds = ne.crop_indx(NX, range(1,30), range(1,30))
Cxinds = ne.crop_indx(NX, range(9,24), range(9,24))
# Cxinds = ne.crop_indx(NX, range(5,20), range(5,20))
# Cxinds = ne.crop_indx(NX, range(20,44), range(20,44))
NX2 = np.sqrt(len(Cxinds)).astype(int)
plt.figure(figsize=(10,10))
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    sta = stas[:,bestlag,cc]
    plt.imshow(np.reshape(sta[Cxinds], [NX2,NX2]))

#%% 

# %% Get stimulus model
side2b, Xstim, Rvalid, rinds, cids, Cxinds = ne.get_stim_model(Stim, Robs, [NX,NY], valid=valid_inds, num_lags=10, plot=True, Cindx=Cxinds, autoencoder=False)

# %% plot filters (check if they're spatially localized)
DU.plot_3dfilters(side2b)

#%% Refine stimulus model

side2c, Xstim, Rvalid, rinds, cids, Cxinds = ne.get_stim_model(Stim, Robs, [NX,NY], valid=valid_inds, num_lags=10, plot=True,
            XTreg=[0.001, 0.01],L1reg=5e-3,MIreg=0.1,MSCreg=0.1,Greg=0.5,Mreg=1e-4,Cindx=Cxinds, base_mod=side2b, cids=cids)

#%% plot filters
DU.plot_3dfilters(side2c)

# %% if second model is better copy

side2b = side2c.copy_model()

#%% Run eye correction
# Run it all once
eyeAtFrameCentered = (eyeAtFrame-(640, 380))
centers5, locs, LLspace1 = ne.get_corr_grid(side2b, Stim, Robs, [NX,NY], cids,
        eyeAtFrameCentered, valdata, valid_eye_rad=5.2*ppd, Erad=0.5*ppd, Npos=25,
        crop_edge=3, plot=True, interpolation_steps=2, softmax=1000, autoencoder=False)

#%% plot one LL surface
ctrid = np.argmin(locs**2)
idxs = np.arange(2, 14)
nplot = len(idxs)

# plt.figure()
# I2 = np.mean(LLspace1.reshape((-1, 24, 29)), axis=0)
# I2 = 1-ne.normalize_range(I2)
# plt.imshow(I2)

plt.figure(figsize=(15,6))
for i,j in zip(idxs, range(nplot)):
    plt.subplot(1,nplot,j+1)
    irow = 10
    I = LLspace1[irow,i,:,:]
    I = 1-ne.normalize_range(I) # flip sign and normalize between 0 and 1
    Ly,Lx = I.shape

    x,y = ne.radialcenter(I**10)
    print("(%d, %d)" %(x-Lx//2,y-Ly//2))
    plt.imshow(I, aspect='equal')
    plt.axvline(Lx/2, color='w')
    plt.axhline(Ly/2, color='w')
    plt.plot(x,y, '+r')
    plt.title('(%d,%d)' %(irow,idxs[j]))

#%% Get LLsurface centers
centers5,LLspace3 = ne.get_centers_from_LLspace(LLspace1, softmax=10, plot=False, interpolation_steps=2, crop_edge=2)

plt.figure()
plt.plot(centers5[:,:,0], centers5[:,:,1], '.')

#%% Correct stimulus
centers6 = deepcopy(centers5)
ctrid = np.argmin(locs**2)
centers6[:,:,0]-=centers6[ctrid,ctrid,0]
centers6[:,:,1]-=centers6[ctrid,ctrid,1]

xcorrec,ycorrec = ne.get_shifter_from_centers(centers6, locs, maxshift=1, nearest=False)

eyeAtFrameCentered = (eyeAtFrame-(640, 380))

xshift = xcorrec(eyeAtFrameCentered[:,0], eyeAtFrameCentered[:,1])
yshift = ycorrec(eyeAtFrameCentered[:,0], eyeAtFrameCentered[:,1])

ix = np.logical_or(np.isnan(xshift), np.isnan(yshift))

xshift[ix] = 0.0
yshift[ix] = 0.0

# NEAREST option shifts shifts in integer numbers of pixels
StimC = ne.shift_stim(Stim, xshift, yshift, [NX,NY], nearest=True)

#%% recompute STAs to compare progress
from copy import deepcopy
# shift eye at frame to shifted values
newEyeAtFrame = deepcopy(eyeAtFrameCentered)
newEyeAtFrame[:,0] += xshift
newEyeAtFrame[:,1] += xshift

valid = np.intersect1d(np.where(~ix)[0], valdata)
# check new STAS
stas0 = ne.get_stas(Stim, Robs, [NX,NY], valid=valid, num_lags=14, plot=False)
stas = ne.get_stas(StimC, Robs, [NX,NY], valid=valid, num_lags=14, plot=False)


plt.figure(figsize=(4,NC*2))
NC = Robs.shape[1]
# sx,sy = U.get_subplot_dims(NC)
for cc in range(NC):
    
    # plt.subplot(sx,sy,cc+1)
    bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    plt.subplot(NC, 2, 2*cc + 1)
    plt.imshow(np.reshape(stas0[:,bestlag,cc], (NY,NX)))
    plt.title(cc)
    plt.axis("off")

    plt.subplot(NC, 2, 2*cc + 2)
    plt.imshow(np.reshape(stas[:,bestlag,cc], (NY,NX)))
    plt.title(cc)
    plt.axis("off")
    

#%% Re-evaluate Log-Likelihoods to compare model performance after correction
num_lags = side2b.input_sizes[0][-1]
NX2 = np.sqrt(len(Cxinds)).astype(int)
    
# make new cropped stimulus
Xstim0, _ = ne.create_time_embedding_valid(Stim[:,Cxinds], [num_lags, NX2, NX2], valid)
Xstim1, rinds = ne.create_time_embedding_valid(StimC[:,Cxinds], [num_lags, NX2, NX2], valid)

NT = len(rinds)
Ui, Xi = NDNutils.generate_xv_folds(NT)

Rvalid = deepcopy(Robs[rinds,:])
Rvalid = Rvalid[:,cids]
Rvalid = NDNutils.shift_mat_zpad(Rvalid,-1,dim=0) # get rid of first lag

# evaluate model before and after correction
LL0 = side2b.eval_models(input_data=Xstim0, output_data=Rvalid, data_indxs=Xi, nulladjusted=True)
LL1 = side2b.eval_models(input_data=Xstim1, output_data=Rvalid, data_indxs=Xi, nulladjusted=True)

# plot
plt.figure()
plt.plot(LL0, LL1, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel("No Calibration")
plt.ylabel("V1 Calibration")


#%% plot stas all lags
num_lags = stas.shape[1]
# plot individual units

for cc in range(NC):
    plt.figure(figsize=(15,2))
    for ilag in range(num_lags):
        ax = plt.subplot(1,num_lags, ilag+1)
        sta = stas[:,:,cc]
#         sta = (sta - np.min(sta)) / (np.max(sta) - np.min(sta))
        sta -= np.mean(sta)
        sta /= np.max(sta)
        I = np.reshape(sta[:,ilag], (NY,NX))
        plt.imshow(I, vmin=-1, vmax=1)#, cmap='coolwarm')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(8.3*(ilag+1))

        # 

# %% try to do better: get a new stimulus model
side2c, _, _, _, _, _ = ne.get_stim_model(StimC, Robs, [NX,NY], valid=valdata, num_lags=10, plot=True, Cindx=Cxinds, cids=cids)
#%%
side2c, _, _, _, _, _ = ne.get_stim_model(StimC, Robs, [NX,NY], valid=valdata, num_lags=10, plot=True,
            XTreg=[0.01, 0.05],L1reg=5e-3,MIreg=0.1,MSCreg=0.1,Greg=0.05,Mreg=1e-4,Cindx=Cxinds, base_mod=side2b, cids=cids)

#%% plot model
DU.plot_3dfilters(side2c)

#%% evaluate new model
LL2 = side2c.eval_models(input_data=Xstim1, output_data=Rvalid, data_indxs=Xi, nulladjusted=True)
plt.figure()
plt.plot(LL1, LL2, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel("Trained before calibration")
plt.ylabel("Trained after V1 Calibration")
#%% if side2c is better
side2b = side2c.copy_model()
#%% compute likelihood surface
eyeAtFrameCentered = (eyeAtFrame-(640, 380))
centers5, locs, LLspace1 = ne.get_corr_grid(side2b, Stim, Robs, [NX,NY], cids,
        eyeAtFrameCentered, valdata, valid_eye_rad=5.2*ppd, Erad=0.5*ppd, Npos=25,
        crop_edge=0, plot=True, interpolation_steps=0, softmax=1000)

#%% plot one LL surface
ctrid = np.argmin(locs**2)
idxs = np.arange(10, 14)
nplot = len(idxs)
plt.figure(figsize=(10,4))
for i,j in zip(idxs, range(nplot)):
    plt.subplot(1,nplot,j+1)
    I = LLspace1[16,i,:,:]
    I = 1-ne.normalize_range(I) # flip sign and normalize between 0 and 1
    Lx,Ly = I.shape

    x,y = ne.radialcenter(I**10)
    print()
    plt.imshow(I, aspect='equal')
    plt.axvline(Ly/2, color='w')
    plt.axhline(Lx/2, color='w')
    plt.plot(x,y, '+r')
    plt.title('(%d,%d)' %(16,i))

#%% Get LLsurface centers
softmax = 10
interpolation_steps = 0
centers5,LLspace3 = ne.get_centers_from_LLspace(LLspace1, softmax=softmax, plot=False, interpolation_steps=interpolation_steps, crop_edge=3)

#%% Correct stimulus
centers6 = deepcopy(centers5)
ctrid = np.argmin(locs**2)
centers6[:,:,0]-=centers6[ctrid,ctrid,0]
centers6[:,:,1]-=centers6[ctrid,ctrid,1]

xcorrec,ycorrec = ne.get_shifter_from_centers(centers6, locs, maxshift=4)


xshift = xcorrec(eyeAtFrameCentered[:,0], eyeAtFrameCentered[:,1])
yshift = ycorrec(eyeAtFrameCentered[:,0], eyeAtFrameCentered[:,1])

ix = np.logical_or(np.isnan(xshift), np.isnan(yshift))

xshift[ix] = 0.0
yshift[ix] = 0.0

StimC = ne.shift_stim(Stim, xshift, yshift, [NX,NY])

#%% recompute STAs
from copy import deepcopy
# shift eye at frame to shifted values
newEyeAtFrame = deepcopy(eyeAtFrameCentered)
newEyeAtFrame[:,0] += xshift
newEyeAtFrame[:,1] += xshift

valid = np.intersect1d(np.where(~ix)[0], valdata)
# check new STAS
stas = ne.get_stas(StimC, Robs, [NX,NY], valid=valid, num_lags=14, plot=False)

plt.figure(figsize=(15,10))
NC = Robs.shape[1]
sx,sy = U.get_subplot_dims(NC)
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    plt.imshow(np.reshape(stas[:,bestlag,cc], (NY,NX)))
    plt.title(cc)
    plt.axis("off")
# %%
plt.figure(figsize=(15,10))
NC = Robs.shape[1]
sx,sy = U.get_subplot_dims(NC)
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    sta = stas[:,:,cc]
    f = plt.plot(sta)
    plt.title(cc)
    plt.axis("off")

#%% plot stas all lags
num_lags = stas.shape[1]
# plot individual units

for cc in range(NC):
    plt.figure(figsize=(15,2))
    for ilag in range(num_lags):
        ax = plt.subplot(1,num_lags, ilag+1)
        sta = stas[:,:,cc]
#         sta = (sta - np.min(sta)) / (np.max(sta) - np.min(sta))
        sta -= np.mean(sta)
        sta /= np.max(sta)
        I = np.reshape(sta[:,ilag], (NY,NX))
        plt.imshow(I, vmin=-1, vmax=1)#, cmap='coolwarm')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(8.3*(ilag+1))


#%%        
# if happy with output, run the correction on all eye positions
eye,ft = ne.load_eye_at_frame(dirname=dirname, flist=flist, corrected=False)

eyeCentered = (eye - (640, 380))
# eyeCentered = eye

xshift = xcorrec(eyeCentered[:,0], eyeCentered[:,1])
yshift = ycorrec(eyeCentered[:,0], eyeCentered[:,1])

ix = np.logical_or(np.isnan(xshift), np.isnan(yshift))

xshift[ix] = 0.0
yshift[ix] = 0.0

eyeCentered[:,0] += xshift
eyeCentered[:,1] += yshift


eyeShift = np.asarray((xshift, yshift)).T

import scipy.io as sio

subjstr = flist[0].split('_')
outname = subjstr[0] + '_' + subjstr[1] + '_eyetraces.mat'

sio.savemat(dirname + outname, {'frameTime': ft, 'eyeAtFrame': eyeCentered, 'eyeShift': eyeShift})            


# %% try analyzing at finer timescale

valid_eye_rad=5.2*ppd
Erad=1*ppd
Npos=25
crop_edge=3
plot=True
interpolation_steps=2
softmax=1

base_mod = side2c.copy_model()

from tqdm import tqdm # progress bar

eyeX = eyeAtFrameCentered[:,0]
eyeY = eyeAtFrameCentered[:,1]

# get valid indices when eye position is within a specified radius
eyeVal = np.hypot(eyeX, eyeY) < valid_eye_rad
valdata = np.intersect1d(valid, np.where(eyeVal)[0])

num_lags = base_mod.network_list[0]['input_dims'][-1]

# recreate full Xstim
# Xstim = NDNutils.create_time_embedding(Stim, [num_lags, NX, NY], tent_spacing=1)
Xstim, rinds = ne.create_time_embedding_valid(Stim, [num_lags, NX, NY], valdata)
Rvalid = deepcopy(Robs[rinds,:])
Rvalid = Rvalid[:,cids]
Rvalid = NDNutils.shift_mat_zpad(Rvalid,-1,dim=0) # get rid of first lag
NC = Rvalid.shape[1]
eyeX = eyeX[rinds]
eyeY = eyeY[rinds]
fixStarts = np.where(np.abs(np.diff(np.hypot(eyeX, eyeY)))>1)[0]

# old network list
netlist_old = deepcopy(base_mod.network_list)[0] 

# convert scaffold network into a convolutional network
scaff_par = deepcopy(base_mod.network_list[0])
scaff_par['input_dims'] = [1, NX, NY] + [scaff_par['input_dims'][-1]]
scaff_par['layer_types']=['conv', 'conv']
scaff_par['conv_filter_widths'] = [netlist_old['input_dims'][1], 1] # base_mod 

side_par = deepcopy(base_mod.network_list[1])
side_par['layer_types'] = ['conv']
side_par['conv_filter_widths'] = [1]

cell_shift_mod = NDN.NDN( [scaff_par, side_par], ffnet_out=1, noise_dist='poisson' )
# copy first network verbatim (only thing diff is output is a convolution)
cell_shift_mod.networks[0].layers[0].weights = deepcopy(base_mod.networks[0].layers[0].weights)
cell_shift_mod.networks[0].layers[0].biases = deepcopy(base_mod.networks[0].layers[0].biases)
cell_shift_mod.networks[0].layers[1].weights = deepcopy(base_mod.networks[0].layers[1].weights)
cell_shift_mod.networks[0].layers[1].biases = deepcopy(base_mod.networks[0].layers[1].biases)
cell_shift_mod.networks[1].layers[0].weights = deepcopy(base_mod.networks[1].layers[0].weights)
cell_shift_mod.networks[1].layers[0].biases = deepcopy(base_mod.networks[1].layers[0].biases)

num_space = np.prod(cell_shift_mod.input_sizes[0][:-1])

# locations in the grid
locs = np.linspace(-valid_eye_rad, valid_eye_rad, Npos)
print(locs)

#%%
# loop over grid and calculate likelihood surfaces
LLspace0 = np.zeros([Npos,Npos,NY-2*crop_edge,NX-2*crop_edge])

# Loop over positions (this is the main time-consuming operation)
xx = 12
yy = 12
    
# get index for when the eye position was withing the boundaries
# rs = np.hypot(eyeX-locs[xx], eyeY-locs[yy])
# ecc = np.hypot(locs[xx],locs[yy])/ppd
# valE = np.where(rs < (Erad + ecc*.5)[0]
# valtot = valE

NT = Rvalid.shape[0]

# valtot = np.intersect1d(valdata, valE)

iFix += 1
valtot = np.arange(fixStarts[iFix],fixStarts[iFix]+100)

Rcc = ne.conv_expand( Rvalid[valtot,:], num_space )
LLs = cell_shift_mod.eval_models(input_data=[Xstim[valtot,:]], output_data=Rcc, nulladjusted=False)

# reshape into spatial map
LLcc = np.reshape(LLs, [NY,NX,NC])
plt.figure()
LLs = np.mean(-LLcc, axis=2)
rx,ry = ne.radialcenter(LLs**100)
plt.imshow(LLs**10)
plt.axvline(NX//2, color='r')
plt.axhline(NY//2, color='r')
plt.plot(rx, ry, '+b')


#%%

if len(valtot) > 100: # at least 100 samples to evaluate a likelihood

    Rcc = ne.conv_expand( Rvalid[valtot,:], num_space )

    # get negative log-likelihood at all spatial shifts
    LLs = cell_shift_mod.eval_models(input_data=[Xstim[valtot,:]], output_data=Rcc, nulladjusted=False)

    # reshape into spatial map
    LLcc = np.reshape(LLs, [NY,NX,NC])

    LLpos = np.mean(LLcc,axis=2)
    if crop_edge == 0:
        LLspace1[xx,yy,:,:] = deepcopy(LLpos)
    else:
        LLspace1[xx,yy,:,:] = deepcopy(LLpos)[crop_edge:-crop_edge,:][:,crop_edge:-crop_edge]
# %% Try fitting Ret/LGN -> V1 model


#%% Make stim
Xstim, rinds = ne.create_time_embedding_valid(StimC, [num_lags, NX, NY], valdata)
Xstim2, rinds2 = ne.create_time_embedding_valid(StimC[:,Cxinds], [num_lags, NX2, NX2], valdata)
Rvalid = deepcopy(Robs[rinds,:])
Rvalid = Rvalid[:,cids]

#%%
NT = Xstim.shape[0]
Ui, Xi = NDNutils.generate_xv_folds(NT)

adam_params = U.def_adam_params()

adam_params
#%% Make model

NC = Rvalid.shape[1]


Greg0 = 1e-3
Greg = 1e-1
Creg0 = 1
Creg = 1e-2
Mreg0 = 1e-1
Mreg = 1e-1
L1reg0 = 1e-5
Xreg = 1e-1

num_tkern = 2
num_subs = 8
num_hidden = 10

ndn_par = NDNutils.ffnetwork_params(
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[num_tkern, num_subs, NC],
    layer_types=['conv', 'conv', 'normal'],
    conv_filter_widths=[1,12,None],ei_layers=[None,num_subs//2],
    normalization=[2,1,-1],act_funcs=['relu', 'relu', 'softplus'],
    verbose=True,
    reg_list={'d2t':[1e-4], 'd2x':[None, Xreg], 'l1':[None, L1reg0],
    'center':[None, Creg0], 'glocal':[None, Mreg0], 'max':[None, None, Mreg0]}
)

auto_par = NDNutils.ffnetwork_params(input_dims=[1, NC, 1],
                xstim_n=[1],
                layer_sizes=[9, 2, 2, 4, 9, NC],
                time_expand=[0, 0, 15, 0, 0, 0],
                layer_types=['normal', 'normal', 'temporal', 'normal', 'normal', 'normal'],
                conv_filter_widths=[None, None, 1, None, None, None],
                act_funcs=['relu', 'relu', 'add', 'relu', 'relu', 'lin'],
                normalization=[1, 1, 1, 1, 1, -1],
                reg_list={'d2t':[None, None, 1e-5, None]}
                )

add_par = NDNutils.ffnetwork_params(
                xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
                layer_types=['add'], act_funcs=['softplus']
                )

retV1b = NDN.NDN( [ndn_par, auto_par, add_par], ffnet_out=2, noise_dist='poisson' )

# set output regularization on the latent
retV1b.batch_size = adam_params['batch_size']
retV1b.initialize_output_reg(network_target=1,layer_target=2, reg_vals={'d2t': 1e-1})

# input_data = [Xstim, Rvalid]

# ndn_par = NDNutils.ffnetwork_params(
#     input_dims=[1,NX,NY,num_lags],
#     layer_sizes=[num_tkern, num_subs, num_hidden, NC],
#     layer_types=['conv', 'conv', 'normal', 'normal'],
#     conv_filter_widths=[1,12,None],ei_layers=[None,num_subs//2],
#     normalization=[2,1,1,-1],act_funcs=['relu', 'relu', 'relu', 'softplus'],
#     verbose=True,
#     reg_list={'d2t':[1e-4], 'd2x':[None, Xreg], 'l1':[None, L1reg0],
#     'center':[None, Creg0], 'glocal':[None, Mreg0], 'max':[None, None, None, Mreg0]}
# )

retV1 = NDN.NDN( [ndn_par], ffnet_out=0, noise_dist='poisson')
retV1.networks[0].layers[0].weights[:,:] = 0
retV1.networks[0].layers[0].weights[2:4,0] = 1
retV1.networks[0].layers[0].weights[2:4,1] = -1

v2f = retV1.fit_variables(fit_biases=True)
v2f[0][0]['biases'] = True
v2f[-1][0]['biases'] = True

#%%

_ = retV1b.train(input_data=[Xstim, Rvalid], output_data=Rvalid, train_indxs=Ui,
    test_indxs=Xi, silent=False, learning_alg='adam', opt_params=adam_params)
# %% fit
DU.plot_3dfilters(retV1b)

LLx1 = retV1b.eval_models(input_data=[Xstim, Rvalid], output_data=Rvalid, data_indxs=Xi, nulladjusted=True, use_gpu=False)

#%%
plt.figure()
plt.hist(LLx1)

#%%

stimgen = retV1b.generate_prediction(input_data=[Xstim, Rvalid], ffnet_target=0, layer_target=-1)
latents = retV1b.generate_prediction(input_data=[Xstim, Rvalid], ffnet_target=1, layer_target=3)
rpred = retV1b.generate_prediction(input_data=[Xstim, Rvalid])
#%%
plt.figure(figsize=(10,4))
plt.plot(stimgen[:200,:])

plt.figure(figsize=(10,4))
plt.plot(latents[:200,:])

#%%
cc = 4
plt.figure(figsize=(10,4))
i += 1000
inds = np.arange(i, i+200)
plt.plot(Rvalid[inds,cc])
plt.plot(rpred[inds,cc])
plt.ylim([0,1])

plt.figure(figsize=(10,4))
plt.plot(stimgen[inds,cc])
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
# plt.imshow( np.reshape(retV1b.networks[0].layers[2].weights[:,0], [700,12]), aspect='auto')
# %%
