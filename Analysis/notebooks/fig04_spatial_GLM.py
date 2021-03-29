#%% set paths
sys.path.insert(0, '/home/jake/Data/Repos/')
# import deepdish as dd
import V1FreeViewingCode.Analysis.notebooks.Utils as U
import V1FreeViewingCode.Analysis.notebooks.gratings as gt

import warnings; warnings.simplefilter('ignore')
import NDN3.NDNutils as NDNutils

which_gpu = NDNutils.assign_gpu()
from scipy.ndimage import gaussian_filter
from copy import deepcopy

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt  # plotting
import seaborn as sns

import NDN3.NDN as NDN
import NDN3.Utils.DanUtils as DU


# %% load example sessions
sesslist = ['ellie_20190107', 'ellie_20170731', 'logan_20200304']

ROIs = {'ellie_20190107': np.array([-14.0, -10.0, 14.0, 10]),
    'ellie_20170731': np.array([-14.0, -10.0, 14.0, 10]), #np.array([-1, -3, 3, .5]), #np.array([0.0, -2.5, 2, 0])
    'logan_20200304': np.array([-14.0, -10.0, 14.0, 10])} #np.array([-.5, -1.5, 1.5, 0.5])}

binSizes = {'ellie_20190107': .3,
    'ellie_20170731': .3,
    'logan_20200304': .3}

pixperdeg = {'ellie_20190107': 43.8059,
    'ellie_20170731': 37.4400,
    'logan_20200304': 37.5048}

example_unit = {'ellie_20190107': 0,
    'ellie_20170731': 28,
    'logan_20200304': 8}


isess = 0
print("Loading %s" %sesslist[isess])
matdat = gt.load_data(sesslist[isess])

sacboxcar,valid,eyepos = gt.get_eyepos_at_frames(matdat['eyepos'], matdat['dots']['frameTime'], slist=matdat['slist'])

# process data
print("Preprocess spatial mapping data")
defopts = {'frate': 120}
eyePos = matdat['dots']['eyePosAtFrame']

ex = eyePos[:,0] #eyepos[:,1] *ppd
ey = eyePos[:,1] #-eyepos[:,2] *ppd

# plt.figure(figsize=(10,5))
# plt.plot(eyePos[:1000,0])
# plt.plot(ex[:1000])

# plt.xlabel('Frames')

xd = matdat['dots']['xpos']-ex[:,np.newaxis]
yd = -matdat['dots']['ypos']+ey[:,np.newaxis]
ft = matdat['dots']['frameTime']

# valid frames
valid = np.where(matdat['dots']['validFrames'])[0]

# build grid
ppd = pixperdeg[matdat['exname']]
ROI = ROIs[matdat['exname']]*ppd
binSize = binSizes[matdat['exname']]*ppd
xax = np.arange(ROI[0], ROI[2], binSize)
yax = np.flip(np.arange(ROI[1], ROI[3], binSize))

xx,yy = np.meshgrid(xax, yax)
xx = xx.flatten()
yy = yy.flatten()

NX = len(xax)
NY = len(yax)
ND = xd.shape[1]
NT = len(ft)

# bin stimulus
print("Binning stimulus")
stim = np.zeros((NT,NX*NY))
for i in range(ND):
    x = (xx[:,np.newaxis]-xd[:,i]).T
    y = (yy[:,np.newaxis]-yd[:,i]).T
    d = np.hypot(x,y)<binSize
    stim = stim + d

plt.figure()
plt.plot(xd[0,:], yd[0,:], '.')
plt.plot(xx,yy,'.')

plt.figure()
plt.imshow(np.reshape(stim[0,:], (NY,NX)), interpolation='none')

# bin spikes
NC = len(matdat['spikes']['cids'])

print("Binning Spikes")
RobsAll = np.zeros((NT,NC))
for i in range(NC):
    cc = matdat['spikes']['cids'][i]
    st = matdat['spikes']['st'][matdat['spikes']['clu']==cc]
    RobsAll[:,i] = gt.bin_at_frames(st,ft,0.1).flatten()

# do downsampling if necessary
t_downsample = np.round(1/np.median(np.diff(matdat['dots']['frameTime'])))/defopts['frate']

if t_downsample > 1:
    stim = gt.downsample_time(stim, t_downsample.astype(int))
    RobsAll = gt.downsample_time(RobsAll, t_downsample.astype(int))
    sacboxcar = gt.downsample_time(sacboxcar, t_downsample.astype(int))
    valid = (valid // t_downsample).astype(int)

print('Done')

#%% get STAs
Robs = deepcopy(RobsAll)
Robs = Robs - np.mean(Robs, axis=0)

from tqdm import tqdm
from scipy.signal import fftconvolve
num_lags = 15
stas = np.zeros((num_lags, NX*NY, NC))
I = np.eye(num_lags)
print("Computing STA")
for idim in tqdm(range(NX*NY)):
    X = fftconvolve(np.expand_dims(stim[:,idim], axis=1), I, 'full')
    X = X[:-num_lags+1,:]
    stas[:,idim,:] = X[valid,:].T @ Robs[valid,:]
cc = 0
#%% plot stas
if cc >= NC:
    cc = 0
plt.figure(figsize=(15,5))

sta = stas[:,:,cc]
spmx = np.where(sta==np.max(sta))[1][0]
tkern = sta[:,spmx]
sta = sta.reshape( (num_lags, NY, NX))

for ilag in range(num_lags):    
    plt.subplot(2, np.ceil(num_lags/2), ilag+1)
    plt.imshow(sta[ilag,:,:], vmin=np.min(sta), vmax=np.max(sta))

plt.figure()
plt.plot(tkern)
cc += 1

#%% get temporal kernels
from scipy.linalg import svd, norm
tkern = np.zeros((num_lags,NC))
for cc in range(NC):
    sta = stas[:,:,cc]
    spmx = np.where(sta==np.max(sta))[1][0]
    tkern[:,cc] = sta[:,spmx]

plt.figure()
plt.imshow(tkern)

plt.figure()
C = np.cov(tkern)
plt.imshow(C)
usv = svd(C)

tkerns = usv[0][:,:2]*np.sign(np.sum(usv[0][:,:2], axis=0))
plt.figure()
f = plt.plot(tkerns)

spatstas = np.zeros( (NX*NY, 2, NC))
for ikern in range(2):
    fstim = fftconvolve(stim, np.expand_dims(tkerns[:,ikern], axis=1))
    fstim = fstim[:-num_lags+1,:]
    spatstas[:,ikern,:] = fstim[valid,:].T @ Robs[valid,:]

#%% plot spatial RFs
plt.figure(figsize=(4,NC*2))
for cc in range(NC):
    plt.subplot(NC, 2, cc*2 + 1)
    plt.imshow(spatstas[:,0,cc].reshape((NY, NX)))
    plt.subplot(NC, 2, cc*2 + 2)
    plt.imshow(spatstas[:,1,cc].reshape((NY, NX)))



#%%
# set optimization parmeters
adam_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')

lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_params['maxiter'] = 1000

# setup training indices
NT = stim.shape[0]
valdata = np.arange(0,NT,1)

NC = Robs.shape[1]
Ui, Xi = NDNutils.generate_xv_folds(NT, num_blocks=2)
Ui = np.intersect1d(Ui, valid)
Xi = np.intersect1d(Xi, valid)

# GLM
numbasis = 5
# NDN parameters for processing the stimulus
par = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY], time_expand=[num_lags],
    layer_sizes=[2, NC],
    layer_types=['temporal', 'normal'], normalization=[1, 0],
    act_funcs=['lin', 'softplus'], verbose=True,
    reg_list={'d2t': [0], 'd2x':[None, 1e-5], 'l1':[None, 1e-5]})


# initialize GLM
glm0 = NDN.NDN([par],  noise_dist='poisson')

# initialize weights with STA
# sta = (sta - np.min(sta)) / (np.max(sta) - np.min(sta))
# glm0.networks[0].layers[0].weights[:,0]=deepcopy((sta - np.min(sta)) / (np.max(sta) - np.min(sta)))

v2f0 = glm0.fit_variables(fit_biases=False)

glm0.networks[0].layers[0].weights[:,:] = tkerns.astype('float32')
glm0.networks[0].layers[1].weights[:,:] = spatstas.reshape((-1, NC)).astype('float32')
#%%
v2f0[-1][0]['weights'] = False
v2f0[-1][-1]['biases'] = True

# train initial model
_ = glm0.train(input_data=[stim], output_data=RobsAll,
    train_indxs=Ui, test_indxs=Xi,
    learning_alg='lbfgs', opt_params=lbfgs_params,
     fit_variables=v2f0)

#%%
# plot filters
DU.plot_3dfilters(glm0)

#%%
LLx0 = glm0.eval_models(input_data=[stim], output_data=RobsAll, data_indxs=Xi, nulladjusted=True)
LLx0
#%%
# Find best regularization

glmbest = glm0.copy_model()

[LLpath, glms] = NDNutils.reg_path(glmbest, input_data=[stim],
    output_data=RobsAll, train_indxs=Ui, test_indxs=Xi,
    reg_type='d2x', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_target=1, ffnet_target=0, fit_variables=v2f0,
    learning_alg='lbfgs', opt_params=lbfgs_params,
    silent=True)

glmbest = glms[np.argmin(LLpath)-1].copy_model()

[LLpath, glms] = NDNutils.reg_path(glmbest, input_data=[stim],
    output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    reg_type='l1', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_target=1, ffnet_target=0, fit_variables=v2f0,
    learning_alg='lbfgs', opt_params=lbfgs_params,
    silent=True)

glmbest = glms[np.argmin(LLpath)].copy_model()

#%%

LLx0 = glmbest.eval_models(input_data=[stim], output_data=RobsAll, data_indxs=Xi, nulladjusted=True)
LLx0
#%%
# plt.plot(LLx0, '-o')
LLx0
#%%



# glmbest = glms[np.argmin(LLpath)-1].copy_model()


DU.plot_3dfilters(glmbest)

# plots

wts = deepcopy(glmbest.networks[0].layers[0].weights)
wts = np.reshape(wts, (-1, num_lags))
wts = (wts - np.min(wts)) / (np.max(wts)-np.min(wts))

fig = plt.figure(figsize=(10,3))
for lag in range(num_lags):
    plt.subplot(1,num_lags, lag+1)
    plt.imshow(np.reshape(wts[:,lag], (NY, NX)), vmin=0, vmax=1, interpolation='none')
    plt.axis("off")


fname = '/home/jake/Data/Datasets/MitchellV1FreeViewing/FVmanuscript/' + 'Fig04_glm_' + matdat['exname'] + '_' + str(cid) + '.mat'
glmbest.matlab_export(fname)

import scipy.io as sio

a = sio.loadmat(fname)
a['NX'] = NX
a['NY'] = NY
a['num_lags'] = num_lags
a['xax'] = xax
a['yax'] = yax
sio.savemat(fname, a)
# %%
