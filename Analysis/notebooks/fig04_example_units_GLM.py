#%% set paths
sys.path.insert(0, '/home/jake/Data/Repos/')
# import deepdish as dd
import Utils as U
import gratings as gt

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

ROIs = {'ellie_20190107': np.array([-14.0, -3.5, -6.5, 3.5]),
    'ellie_20170731': np.array([-1, -3, 3, .5]), #np.array([0.0, -2.5, 2, 0])
    'logan_20200304': np.array([-.5, -1.5, 1.5, 0.5])}

binSizes = {'ellie_20190107': .5,
    'ellie_20170731': .1,
    'logan_20200304': .05}

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

import neureye as ne

print("Creating Time Embedding for stimulus shape (%d,%d)" %(NT,NX*NY))
num_lags = 12
Xstim, rinds = ne.create_time_embedding_valid(stim, [num_lags, NY, NX], valid)
# Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1 )
print("Computing STA")

Y = RobsAll[rinds,:] - np.mean(RobsAll[rinds,:], axis=0)
cid = example_unit[matdat['exname']]

sta = Xstim.T @ Y

print("Done")

# # plot STAS
# from scipy.stats import median_absolute_deviation

# NC = RobsAll.shape[1]
# sx = np.ceil(np.sqrt(NC)).astype(int)
# sy = np.round(np.sqrt(NC)).astype(int)
# plt.figure(figsize=(10,10))

# mthresh = np.zeros(NC)
# for cc in range(NC):
#     plt.subplot(sx,sy,cc+1)
#     I = np.reshape(sta[:,cc], [NX*NY, num_lags])
#     thresh = median_absolute_deviation(I.flatten())
    
#     plt.plot(I.T, color='k')
#     plt.axhline(thresh*5, color='r')
#     plt.title(cc)
#     mthresh[cc] = np.mean(I.flatten() > thresh*5)


# plot STA for target neuron
from scipy.stats import median_absolute_deviation

Robs = deepcopy(RobsAll[rinds,:])
Robs = Robs[:,cid]

sta = Xstim.T @ (Robs - np.mean(Robs, axis=0))

plt.figure(figsize=(4,2))

plt.subplot(1, 2, 1)
I = np.reshape(sta, [NX*NY, num_lags])
tpower = np.std(I, axis=0)
peaklag = np.argmax(tpower)

spk = I[:,peaklag]
mx = np.argmax(spk)
mn = np.argmin(spk)

plt.plot(I[mx,:], color='b')
plt.plot(I[mn,:], color='r')

plt.subplot(1, 2, 2)
plt.imshow(np.reshape(spk, (NY, NX)), aspect='auto', interpolation='none', extent=ROI[np.array([0,2,1,3])]/ppd)

# set optimization parmeters
adam_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')

lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_params['maxiter'] = 1000

# setup training indices
NT = Xstim.shape[0]
valdata = np.arange(0,NT,1)

NC = 1
Ui, Xi = NDNutils.generate_xv_folds(NT, num_blocks=2)

# GLM

# NDN parameters for processing the stimulus
par = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY,num_lags], layer_sizes=[NC],
    layer_types=['normal'], normalization=[0],
    act_funcs=['softplus'], verbose=True,
    reg_list={'d2x':[0.01], 'glocal':[0.1]})

# initialize GLM
glm0 = NDN.NDN([par],  noise_dist='poisson')

# initialize weights with STA
# sta = (sta - np.min(sta)) / (np.max(sta) - np.min(sta))
# glm0.networks[0].layers[0].weights[:,0]=deepcopy((sta - np.min(sta)) / (np.max(sta) - np.min(sta)))

v2f0 = glm0.fit_variables(fit_biases=True)

# train initial model
_ = glm0.train(input_data=[Xstim], output_data=Robs,
    train_indxs=Ui, test_indxs=Xi,
    learning_alg='lbfgs', opt_params=lbfgs_params,
     fit_variables=v2f0)

# plot filters
DU.plot_3dfilters(glm0)


# Find best regularization

glmbest = glm0.copy_model()


[LLpath, glms] = NDNutils.reg_path(glmbest, input_data=[Xstim],
    output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    reg_type='glocal', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_target=0, ffnet_target=0,
    learning_alg='lbfgs', opt_params=lbfgs_params,
    silent=True)


glmbest = glms[np.argmin(LLpath)].copy_model()

[LLpath, glms] = NDNutils.reg_path(glmbest, input_data=[Xstim],
    output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    reg_type='d2x', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_target=0, ffnet_target=0,
    learning_alg='lbfgs', opt_params=lbfgs_params,
    silent=True)


glmbest = glms[np.argmin(LLpath)-1].copy_model()


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
