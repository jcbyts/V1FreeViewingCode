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


#%% load sessions
sesslist = gt.list_sessions()
sesslist = list(sesslist)
for i in range(len(sesslist)):
    print("%d %s" %(i, sesslist[i]))
# %% load data    
# 51, 23, 4
ROIs = {'ellie_20190107': np.array([-14.0, -5.5, -6.5, 3.5]),
    'ellie_20170731': np.array([-1.0, -3.5, 4, 3]),
    'logan_20200304': np.array([-.5, -1.5, 1.5, 0.5])}

binSizes = {'ellie_20190107': .5,
    'ellie_20170731': .2,
    'logan_20200304': .1}

pixperdeg = {'ellie_20190107': 43.8059,
    'ellie_20170731': 37.4400,
    'logan_20200304': 37.5048}

example_unit = {'ellie_20190107': 0,
    'ellie_20170731': 4,
    'logan_20200304': 8}


indexlist = [23]
sess = [sesslist[i] for i in indexlist]

matdat = gt.load_data(sess[0])

sacboxcar,valid,eyepos = gt.get_eyepos_at_frames(matdat['eyepos'], matdat['dots']['frameTime'], slist=matdat['slist'])
#%% process data

print("Preprocess spatial mapping data")
defopts = {'frate': 120}
eyePos = matdat['dots']['eyePosAtFrame']

ex = eyePos[:,0] #eyepos[:,1] *ppd
ey = eyePos[:,1] #-eyepos[:,2] *ppd

plt.figure(figsize=(10,5))
plt.plot(eyePos[:1000,0])
plt.plot(ex[:1000])

plt.xlabel('Frames')

xd = matdat['dots']['xpos']-ex[:,np.newaxis]
yd = matdat['dots']['ypos']-ey[:,np.newaxis]
ft = matdat['dots']['frameTime']


# valid frames
valid = np.where(matdat['dots']['validFrames'])[0]

# build grid
ppd = pixperdeg[matdat['exname']]
ROI = ROIs[matdat['exname']]*ppd
binSize = binSizes[matdat['exname']]*ppd
xax = np.arange(ROI[0], ROI[2], binSize)
yax = np.arange(ROI[1], ROI[3], binSize)

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
    print(i)
    x = (xx[:,np.newaxis]-xd[:,i]).T
    y = (yy[:,np.newaxis]-yd[:,i]).T
    d = np.hypot(x,y)<binSize
    stim = stim + d

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
#%%
import neureye as ne

print("Creating Time Embedding for stimulus shape (%d,%d)" %(NT,NX*NY))
num_lags = 12
Xstim, rinds = ne.create_time_embedding_valid(stim, [num_lags, NX, NY], valid)
# Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1 )
print("Computing STA")

Y = RobsAll[rinds,:] - np.mean(RobsAll[rinds,:], axis=0)
sta = Xstim.T @ Y

print("Done")

#%% plot STAS
from scipy.stats import median_absolute_deviation

NC = RobsAll.shape[1]
sx = np.ceil(np.sqrt(NC)).astype(int)
sy = np.round(np.sqrt(NC)).astype(int)
plt.figure(figsize=(10,10))

mthresh = np.zeros(NC)
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    I = np.reshape(sta[:,cc], [NX*NY, num_lags])
    thresh = median_absolute_deviation(I.flatten())
    
    plt.plot(I.T, color='k')
    plt.axhline(thresh*5, color='r')
    plt.title(cc)
    mthresh[cc] = np.mean(I.flatten() > thresh*5)

#%%
plt.figure()
plt.plot(mthresh, '-o')

cids = np.where(mthresh>0.01)[0]
# cids = np.intersect1d(cids, np.where(matdat['spikes']['cgs']==2)[0])
print("%d selective single units" %len(cids))
#%%

Robs = deepcopy(RobsAll[rinds,:])
Robs = Robs[:,cids]
NC = Robs.shape[1]
peaklags = np.zeros(NC)
plt.figure(figsize=(4,2*NC))
for cc in range(NC):
    plt.subplot(NC, 2, cc*2 + 1)
    I = np.reshape(sta[:,cc], [NX*NY, num_lags])
    tpower = np.std(I, axis=0)
    peaklag = np.argmax(tpower)
    peaklags[cc]=deepcopy(peaklag)
    spk = I[:,peaklag]
    mx = np.argmax(spk)
    mn = np.argmin(spk)

    plt.plot(I[mx,:], color='b')
    plt.plot(I[mn,:], color='r')

    plt.subplot(NC, 2, cc*2 + 2)
    plt.imshow(np.reshape(spk, (NY, NX)), aspect='auto', interpolation='none', extent=ROI[np.array([0,3,1,2])]/ppd)



# %% set optimization parmeters
adam_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')

lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_params['maxiter'] = 1000

# %% setup training indices
valdata = np.arange(0,NT,1)

NT = Xstim.shape[0]
Ui, Xi = NDNutils.generate_xv_folds(NT, num_blocks=2)

#%% shared NIM
Greg0 = .01
Mreg0 = None
L1reg0 = None
XTreg = .1
num_subs = 1
num_tkerns = 1

NC = Robs.shape[1]
# par = NDNutils.ffnetwork_params( 
#     input_dims=[1,NX,NY,num_lags], layer_sizes=[num_subs, NC],
#     layer_types=['normal', 'normal'], normalization=[1, -1],
#     act_funcs=['lin', 'softplus'], verbose=True,
#     reg_list={'d2xt':[XTreg], 'glocal':[Greg0]}
# )

par = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY,num_lags], layer_sizes=[NC],
    layer_types=['normal'], normalization=[0],
    act_funcs=['softplus'], verbose=True,
    reg_list={'d2t': [XTreg], 'd2x':[XTreg], 'local':[Greg0]}
)

# par = NDNutils.ffnetwork_params( 
#     input_dims=[1,NX,NY,num_lags], layer_sizes=[num_tkerns, num_subs, NC],
#     layer_types=['conv', 'normal', 'normal'], normalization=[1, 1, 0],
#     conv_filter_widths=[1],
#     act_funcs=['lin', 'lin', 'softplus'], verbose=True,
#     reg_list={'d2t':[0.1],'d2x':[None, XTreg], 'local':[None, Greg0]}
# )

glm0 = NDN.NDN([par],  noise_dist='poisson')
# sta = Xstim.T @ np.sum(Y, axis=1) / np.sum(Y)
# sta/=np.max(sta)
# glm0.networks[0].layers[0].weights[:,0]=sta

# v2f0 = glm0.fit_variables(layers_to_skip=[0], fit_biases=False)
v2f0 = glm0.fit_variables(fit_biases=False)
v2f0[-1][-1]['biases']=True

# glm0.networks[0].layers[0].weights[:] = 0
# glm0.networks[0].layers[0].weights[3:5,0] = 1.0#tkern
# time_spread = 75
# glm0.time_spread = time_spread
_ = glm0.train(input_data=[Xstim], output_data=Robs,
    train_indxs=Ui, test_indxs=Xi,
    learning_alg='lbfgs', opt_params=lbfgs_params,
     fit_variables=v2f0)
DU.plot_3dfilters(glm0)


#%%

[LLpath, glms] = NDNutils.reg_path(glm0, input_data=[Xstim],
    output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    reg_type='d2x', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_target=0, ffnet_target=0,
    learning_alg='lbfgs', opt_params=lbfgs_params,
    silent=True)


glmbest = glms[np.argmin(LLpath)].copy_model()
[LLpath, glms] = NDNutils.reg_path(glmbest, input_data=[Xstim],
    output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    reg_type='d2t', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_target=0, ffnet_target=0,
    learning_alg='lbfgs', opt_params=lbfgs_params,
    silent=True)

glmbest = glms[np.argmin(LLpath)].copy_model()

[LLpath, glms] = NDNutils.reg_path(glmbest, input_data=[Xstim],
    output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    reg_type='local', reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
    layer_target=0, ffnet_target=0,
    learning_alg='lbfgs', opt_params=lbfgs_params,
    silent=True)

glmbest = glms[np.argmin(LLpath)].copy_model()

DU.plot_3dfilters(glmbest)
# LLx = nim_shared.eval_models(input_data=[Xstim], output_data=Robs,
#                         data_indxs=Xi, nulladjusted=True)

#%%
cc = 8

#%%
glm1 = glm0.copy_model()
glm1.noise_dist = 'poisson'
_ = glm1.train(input_data=[stim], output_data=RobsAll,
    train_indxs=Ui, test_indxs=Xi,
    learning_alg='adam', opt_params=adam_params,
     fit_variables=v2f0)
DU.plot_3dfilters(glm1)

#%%
def r_squared(true, pred, data_indxs=None):
    """
    START.

    :param true: vector containing true values
    :param pred: vector containing predicted (modeled) values
    :param data_indxs: obv.
    :return: R^2

    It is assumed that vectors are organized in columns

    END.
    """

    assert true.shape == pred.shape, 'true and prediction vectors should have the same shape'

    if data_indxs is None:
        dim = true.shape[0]
        data_indxs = np.arange(dim)
    else:
        dim = len(data_indxs)

    ss_res = np.sum(np.square(true[data_indxs, :] - pred[data_indxs, :]), axis=0) / dim
    ss_tot = np.var(true[data_indxs, :], axis=0)

    return 1 - ss_res/ss_tot
#%%
yhat0 = glm0.generate_prediction(input_data=[stim])
yhat1 = glm1.generate_prediction(input_data=[stim])

plt.figure(figsize=(10,NC*2))
for i in range(NC):
    plt.subplot(NC,1,i+1)
    ix = Xi[0:100]
    plt.plot(yhat0[ix,i])
    plt.plot(yhat1[ix,i])
    plt.plot(RobsAll[ix,i], 'k')
    
plt.figure(figsize=(5,5))
plt.plot(r_squared(RobsAll, yhat0, data_indxs=Xi), r_squared(RobsAll, yhat1, data_indxs=Xi), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
#%%
v2f0 = glm1.fit_variables(fit_biases=False)
v2f0[-1][-1]['biases']=True
print("Training with %s" %('lbfgs'))
_ = glm1.train(input_data=[Xstim], output_data=RobsAll,
    train_indxs=Ui, test_indxs=Xi,
    learning_alg='lbfgs', opt_params=lbfgs_params,
     fit_variables=v2f0)

DU.plot_3dfilters(glm1)
#%%
plt.plot(glm1.networks[0].layers[-1].weights[0,:])


# %%

#%%
# cc = 0
cc += 1
I = np.reshape(sta[:,cc], [NX*NY, num_lags])
tpower = np.std(I, axis=0)
peaklag = np.argmax(tpower)
peaklags[cc]=deepcopy(peaklag)
spk = I[:,peaklag]
plt.figure(figsize=(6,6))
plt.imshow(np.reshape(spk, (NY, NX)), aspect='auto')
#%% saccade lagged

Xstim = NDNutils.create_time_embedding(stim, [num_lags, NX, NY])

sacon = np.where(np.diff(sacboxcar, axis=0)==1)[0]
sacoff = np.where(np.diff(sacboxcar, axis=0)==-1)[0]

#%%
offsets = [-20, -10, 0, 10, 20, 30]
noffsets = len(offsets)
Robs = RobsAll[:,cids]
NC = len(cids)

stas = np.zeros( (NX*NY*num_lags, NC, noffsets))

plt.figure(figsize=(10,2*NC))
for oo in range(noffsets):
    vinds = np.unique(sacon[:,np.newaxis] + (offsets[oo] + np.arange(0, 20)))
    vinds = np.intersect1d(valid, vinds)

    sta = Xstim[vinds,:].T @ Robs[vinds,:]
    sta = sta / np.expand_dims(np.mean(Xstim, axis=0), axis=1)
    stas[:,:,oo] = deepcopy(sta)

    for cc in range(NC):
        plt.subplot(NC, noffsets, cc*noffsets + oo + 1)

        I = np.reshape(sta[:,cc], [NX*NY, num_lags])
        spk = I[:,peaklags[cc].astype(int)]
        plt.imshow(np.reshape(spk, (NX, NY)), aspect='auto')


#%%
plt.figure(figsize=(4,2*NC))
for cc in range(NC):
    plt.subplot(NC, 2, cc*2 + 1)
    I = np.reshape(sta[:,cc], [NX*NY, num_lags])
    tpower = np.std(I, axis=0)
    peaklag = np.argmax(tpower)
    spk = I[:,peaklag]
    mx = np.argmax(spk)
    mn = np.argmin(spk)

    plt.plot(I[mx,:], color='b')
    plt.plot(I[mn,:], color='r')

    plt.subplot(NC, 2, cc*2 + 2)
    plt.imshow(np.reshape(spk, (NX, NY)), aspect='auto')