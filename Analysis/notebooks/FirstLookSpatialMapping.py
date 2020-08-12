#%% set paths
sys.path.insert(0, '/home/jcbyts/Repos/')
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

indexlist = [2]
sess = [sesslist[i] for i in indexlist]

matdat = gt.load_data(sess[0])

#%% process data
print("Preprocess spatial mapping data")
defopts = {'frate': 120}
ex =matdat['dots']['eyePosAtFrame'][:,0]
ey =-matdat['dots']['eyePosAtFrame'][:,1]
xd = matdat['dots']['xpos']-ex[:,np.newaxis]
yd = matdat['dots']['ypos']-ey[:,np.newaxis]
ft = matdat['dots']['frameTime'].flatten()

# valid frames
valid = np.where(matdat['dots']['validFrames'])
valid = valid[0]
ft = ft[valid]
xd = xd[valid,:]
yd = yd[valid,:]

binSize = 40
xax = np.arange(-400, 400, binSize)
yax = np.arange(-400, 400, binSize)
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

print('Done')
#%%

print("Creating Time Embedding for stimulus shape (%d,%d)" %(NT,NX*NY))
num_lags = 12
Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1 )
print("Computing STA")
Y = RobsAll - np.mean(RobsAll, axis=0)
sta = Xstim.T @ Y

#%% plot STAS
sx = np.ceil(np.sqrt(NC)).astype(int)
sy = np.round(np.sqrt(NC)).astype(int)
plt.figure(figsize=(10,10))
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    I = np.reshape(sta[:,cc], [NX*NY, num_lags])
    plt.imshow(I, aspect='auto')

# %% set optimization parmeters
adam_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')

early_stopping = 100

adam_params['batch_size'] = 1000 #NT // 50
adam_params['display'] = 30
adam_params['epochs_training'] = early_stopping * 100
adam_params['run_diagnostics'] = False

adam_params['epsilon'] = 0.1#1e-8
adam_params['early_stop'] = early_stopping
adam_params['early_stop_mode'] = 1
#adam_params['data_pipe_type'] = 'iterator'
adam_params['data_pipe_type'] = 'data_as_var'
adam_params['learning_rate'] = 0.01# 1e-3

lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_params['maxiter'] = 1000

# %% setup training indices
valdata = np.arange(0,NT,1)

Ui, Xi = NDNutils.generate_xv_folds(NT, num_blocks=2)

#%% shared NIM
Greg0 = .001
Mreg0 = None
L1reg0 = None
XTreg = .001
num_subs = 1
num_tkerns = 1


# par = NDNutils.ffnetwork_params( 
#     input_dims=[1,NX,NY,num_lags], layer_sizes=[num_subs, NC],
#     layer_types=['normal', 'normal'], normalization=[1, -1],
#     act_funcs=['lin', 'softplus'], verbose=True,
#     reg_list={'d2xt':[XTreg], 'glocal':[Greg0]}
# )

par = NDNutils.ffnetwork_params( 
    input_dims=[1,NX,NY], layer_sizes=[num_tkerns, num_subs, NC], time_expand=[num_lags],
    layer_types=['temporal', 'normal', 'normal'], normalization=[1, 1, 0],
    act_funcs=['lin', 'lin', 'softplus'], verbose=True,
    reg_list={'d2t':[0.0001],'d2x':[None, XTreg], 'local':[None, Greg0]}
)

glm0 = NDN.NDN([par],  noise_dist='gaussian')
# sta = Xstim.T @ np.sum(Y, axis=1) / np.sum(Y)
# sta/=np.max(sta)
# glm0.networks[0].layers[0].weights[:,0]=sta

#%%
# v2f0 = glm0.fit_variables(layers_to_skip=[0], fit_biases=False)
v2f0 = glm0.fit_variables(fit_biases=False)
v2f0[-1][-1]['biases']=True

time_spread = 75
glm0.time_spread = time_spread
_ = glm0.train(input_data=[stim], output_data=RobsAll,
    train_indxs=Ui, test_indxs=Xi,
    learning_alg='adam', opt_params=adam_params,
     fit_variables=v2f0)
DU.plot_3dfilters(glm0)

# LLx = nim_shared.eval_models(input_data=[stim], output_data=Robs, data_filters=DF,
#                         data_indxs=Xi, nulladjusted=True)

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
