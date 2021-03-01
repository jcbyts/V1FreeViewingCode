# %% Import libraries
sys.path.insert(0, '/home/jake/Data/Repos/')
import deepdish as dd
import Utils as U
import gratings as gt


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

# %% list sessions
sesslist = gt.list_sessions()
sesslist = list(sesslist)
for i in range(len(sesslist)):
    print("%d %s" %(i, sesslist[i]))

# %% Load one session
indexlist = [51]
stim, sacon, sacoff, Robs, DF, basis, opts, sacbc, valid, eyepos = gt.load_and_setup(indexlist,npow=1.8)
# # %% save to matlab
# opts['stim'] = stim
# opts['sacon'] = sacon
# opts['Robs'] = Robs

# from scipy.io import savemat

# savemat('testdata.mat', opts)

# %% build time-embedded stimulus
num_saclags = 60
back_shifts = 20
num_lags = 15
NX,NY = opts['NX'],opts['NY']
NT,NC=Robs.shape
# build time-embedded stimulus
Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1 )
XsacOn = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacon,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
XsacOff = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacoff,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
XsacOnCausal = NDNutils.create_time_embedding( sacon, [num_saclags, 1, 1], tent_spacing=1)
XsacOffCausal = NDNutils.create_time_embedding( sacoff, [num_saclags, 1, 1], tent_spacing=1)
Robs = Robs.astype('float32')

#%% optimization params
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
seed = 66
optimizer = 'adam'


if noise_dist=='poisson':
    null_adjusted = True
else:
    null_adjusted = False

if optimizer=='adam':
    opt_params = adam_params
else:
    opt_params = lbfgs_params
    

# %% fit GLM
Ui = opts['Ui']
Xi = opts['Xi']
Ti = opts['Ti']

glm_par = NDNutils.ffnetwork_params(
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[NC],
    layer_types=['readout'],
    act_funcs=['lin'],
    normalization=[0],
    reg_list={'l2':1e-2, 'd2xt': 1e-5, 'l1':1e-5}
    )

autoencoder = NDNutils.ffnetwork_params(input_dims=[1, NC, 1],
    xstim_n=[1],
    layer_sizes=[2, 1, NC],
    time_expand=[0, 15, 0],
    layer_types=['normal', 'temporal', 'normal'],
    conv_filter_widths=[None, 1, None],
    act_funcs=['relu', 'lin', 'lin'],
    normalization=[1, 1, 0],
    reg_list={'d2t':[None, 1e-5, None]}
    )

add_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus']
)

glm = NDN.NDN([glm_par, autoencoder, add_par], tf_seed=seed, noise_dist=noise_dist)

# time_expand=[0, 0, num_ca_lags], normalization=[0,0,0], 
#     layer_types=['conv','normal', 'conv'], conv_filter_widths=[L, None, 1],

glm.batch_size = adam_params['batch_size']
glm.initialize_output_reg(network_target=1,layer_target=1, reg_vals={'d2t': 1e-1})
glm.time_spread = 100

v2f = glm.fit_variables()
for nn in range(len(v2f[1])-1):
    v2f[1][nn]['biases'] = False
#%%
_ = glm.train(input_data=[Xstim, Robs], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f)

print("Done")

#%%

# rpred = glm0.generate_prediction(input_data=[Xstim, Robs], ffnet_target=-1, layer_target=-1)
rpred1 = glm.generate_prediction(input_data=[Xstim, Robs], ffnet_target=1, layer_target=1)
plt.figure(figsize=(10,4))
# f = plt.plot(rpred[:200])
f = plt.plot(rpred1[:200])

#%%
glm0 = glm.copy_model()
#%%
reg_min=gt.find_best_reg(glm, input_data=[Xstim], output_data=Robs,
    train_indxs=Ui, test_indxs=Xi, reg_type='l2',
    opt_params=lbfgs_params, learning_alg='lbfgs')

#%%
glm0 = glm.copy_model() # store initial GLM
glm.set_regularization(reg_type='l2', reg_val=reg_min, ffnet_target=0, layer_target=0)

#%%
_ = glm.train(input_data=[Xstim], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    learning_alg=optimizer, opt_params=opt_params, use_dropout=False)

# %% evaluate models
Ti = opts['Ti']
LLx0 = glm.eval_models(input_data=[Xstim], output_data=Robs, data_indxs=Ti, nulladjusted=null_adjusted)
print(LLx0)
# %% plot learned RFs

filters = DU.compute_spatiotemporal_filters(glm)
gt.plot_3dfilters(filters, basis=basis)


# %%
Rpred0 = glm.generate_prediction(input_data=[Xstim])

#%%
# cc +=1
cc = 26
Ti = opts['Ti']
r = np.reshape(Robs[Ti,cc], (opts['num_repeats'],-1))
r0 = np.reshape(Rpred0[Ti,cc], (opts['num_repeats'],-1))
r = np.average(r, axis=0)
r0 = np.average(r0, axis=0)
plt.plot(r)
plt.plot(r0)
plt.title("cell %d" %cc)
U.r_squared(np.reshape(r, (-1,1)), np.reshape(r0, (-1,1)))

# %%

# %% add saccade kernels

stim_par = glm_par()
stim_par['act_funcs'] = 'lin'

num_subs = 4

l2=1e-2

sac_on_par = NDNutils.ffnetwork_params( 
    input_dims=[1,1,1,num_saclags],
    xstim_n=[1],
    layer_sizes=[num_subs,NC], 
    layer_types=['normal', 'normal'], act_funcs=['lin', 'lin'], 
    normalization=[0],
    reg_list={'d2t':[10],'l2':[l2,l2], 'l1':[None,1e-5]}
)

sac_off_par = NDNutils.ffnetwork_params( 
    input_dims=[1,1,1,num_saclags],
    xstim_n=[2],
    layer_sizes=[NC], 
    layer_types=['normal', 'normal'], act_funcs=['lin', 'lin'], 
    normalization=[0],
    reg_list={'d2t':[10],'l2':[l2,l2], 'l1':[None, 1e-5]}
)

# # model 3: stim x saccade + saccade
mult_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
    layer_types=['mult'], act_funcs=['lin']
)

# only combine the onset kernel
comb_par = NDNutils.ffnetwork_params(
    xstim_n=None, ffnet_n=[2,3], layer_sizes=[NC],
    layer_types=['add'], act_funcs=['softplus']
)


ndn0 = NDN.NDN([stim_par, sac_on_par, sac_off_par, mult_par, comb_par], noise_dist='poisson')
    


# num_subs = 10

# glm_par = NDNutils.ffnetwork_params(input_dims=[1,NX,NY,num_lags],
#     layer_sizes=[num_subs, NC],
#     layer_types=['normal', 'normal'],
#     act_funcs=['relu', 'softplus'],
#     normalization=[0, -1],
#     reg_list={'l2':1e-2, 'l1':1e-5, 'd2xt':1e-5}
#     )

# glm = NDN.NDN([glm_par], tf_seed=seed, noise_dist=noise_dist)

# _ = glm.train(input_data=[Xstim], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
#     learning_alg=optimizer, opt_params=opt_params, use_dropout=False)

# print("Done")
# %%
