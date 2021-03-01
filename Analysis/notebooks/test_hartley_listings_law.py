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
sessid = 51

matdat = gt.load_data(sessid)
sessmincpd = 2.0
npow = 1.8
nsteps = 5
# basis
basis = {'name': 'cosine', 'nori': 8, 'nsf': int(nsteps), 'endpoints': [sessmincpd, npow], 'support': 500}

# load session
stim, Robs, sacon, sacoff, basis, opts, sacbc, valid,eyepos = gt.load_and_preprocess(sessid, basis=basis, opts={})


# %% save to matlab
# opts['stim'] = stim
# opts['sacon'] = sacon
# opts['Robs'] = Robs

# from scipy.io import savemat

# savemat('testdata.mat', opts)



# %% build time-embedded stimulus
num_lags = 15
NX,NY = opts['NX'],opts['NY']

# build time-embedded stimulus
Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1 )
# XsacOn = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacon,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
Robs = Robs.astype('float32')

NT,NC=Robs.shape

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
    

# %% fit NIM
Ui = opts['Ui'] # these already are valid indices
Xi = opts['Xi']
Ti = opts['Ti']
import seaborn as sns
eyeX = deepcopy(eyepos[:,1])
eyeY = deepcopy(eyepos[:,2])

plt.figure()
# f = plt.hist2d( eyeX[Ui], eyeY[Ui], bins=100)
sns.kdeplot(eyeX[Ui], eyeY[Ui], bw=.05, shade=True, shade_lowest=False)
plt.title('Eye Position Density')

rs = np.hypot(eyeX, eyeY)

Ui = np.intersect1d(Ui, np.where(np.logical_and(eyeX < 0, eyeY < 0))[0])
num_subs = NC//2

nim_par = NDNutils.ffnetwork_params(
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[num_subs, NC],
    layer_types=['normal', 'normal'],
    act_funcs=['relu', 'softplus'],
    normalization=[0],
    reg_list={'l2':1e-2, 'd2xt': 1e-5, 'glocal':1e-1}
    )

nim0 = NDN.NDN([nim_par], tf_seed=seed, noise_dist=noise_dist)

v2f = nim0.fit_variables(fit_biases=True)

# train
_ = nim0.train(input_data=[Xstim], output_data=Robs, train_indxs=Ui, test_indxs=Xi,
    learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f)

print("Done")

DU.plot_3dfilters(nim0)

# only include cells that are well fit
LLx = nim0.eval_models(input_data=[Xstim], output_data=Robs,data_indxs=Xi, nulladjusted=True)
plt.plot(LLx, 'o')

cids = np.where(LLx > 0.05)[0]
NC = len(cids)
Robsv = deepcopy(Robs[:,cids])

num_subs = NC//2

nim_par = NDNutils.ffnetwork_params(
    input_dims=[1,NX,NY,num_lags],
    layer_sizes=[num_subs, NC],
    layer_types=['normal', 'normal'],
    act_funcs=['relu', 'softplus'],
    normalization=[0],
    reg_list={'l2':1e-2, 'd2xt': 1e-5, 'glocal':1e-1}
    )

nim0 = NDN.NDN([nim_par], tf_seed=seed, noise_dist=noise_dist)

_ = nim0.train(input_data=[Xstim], output_data=Robsv, train_indxs=Ui, test_indxs=Xi,
    learning_alg=optimizer, opt_params=opt_params, use_dropout=False, fit_variables=v2f)

print("Done")

DU.plot_3dfilters(nim0)

#%% Loop over eye position on the monitor and evaluate likelihood with different rotations of the stimulus

vinds = np.where(valid)[0] # valid indices as integers instead of boolean

from tqdm import tqdm # progress bar

eyeX = deepcopy(eyepos[:,1])
eyeY = deepcopy(eyepos[:,2])

max_rotate = 5
nsteps = 20
rotate = np.linspace(-max_rotate, max_rotate, nsteps)

valid_eye_rad=5.2
Erad=0.5
Npos=25

locs = np.linspace(-valid_eye_rad, valid_eye_rad, Npos)

LLsurf = np.zeros([Npos, Npos, nsteps])

# Loop over positions (this is the main time-consuming operation)
for ith in tqdm(range(nsteps)):

    # rotate stimulus
    orientation = matdat['grating']['ori']-rotate[ith]
    # bin stimulus (on specified basis)
    stim1, basisopts = gt.bin_on_basis(orientation, matdat['grating']['cpd'], basis)

    # do downsampling if necessary
    t_downsample = (np.round(1/np.median(np.diff(matdat['grating']['frameTime'])))/opts['frate']).astype(int)
    
    if t_downsample > 1:
        stim1 = gt.downsample_time(stim1, t_downsample)

    # embed time
    Xstim = NDNutils.create_time_embedding( stim1, [num_lags, NX, NY])

    # loop over spatial positions
    for xx in range(Npos):
        for yy in range(Npos):

            # get index for when the eye position was withing the boundaries
            rs = np.hypot(eyeX-locs[xx], eyeY-locs[yy])
            
            # eccentricity dependent 
            ecc = np.hypot(locs[xx],locs[yy])
            Ethresh = Erad + .2*ecc # eccentricity dependent threshold
                
            valE = np.where(rs < Ethresh)[0]
            valtot = np.intersect1d(valE, vinds)

            if len(valtot) > 100: # at least 100 samples to evaluate a likelihood
                
                # evaluate model on rotated stimulus
                LLs = nim0.eval_models(input_data=Xstim, output_data=Robsv, nulladjusted=False, data_indxs=valtot)

                LLpos = np.mean(LLs)

                LLsurf[xx,yy,ith] = deepcopy(LLpos)

# #%%
# import scipy.io as sio
# outname = 'LLsurfTest.mat'
# sio.savemat(dirname + outname, {'LLsurf': LLsurf})
# %%

def softmax(x,y,pow=2):
    # minmax it
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    y = y**pow / np.sum(y**pow)

    return x@y

def LLinterpolate(LLsurf, xx, yy):
    sz = LLsurf.shape
    a=np.squeeze(-deepcopy(LLsurf[xx,yy,:]))
    ctr=1
    if xx < sz[0]-1:
        a+=np.squeeze(-deepcopy(LLsurf[xx+1,yy,:]))
        ctr+=1
    if xx > 0:
        a+=np.squeeze(-deepcopy(LLsurf[xx-1,yy,:]))
        ctr+=1
    if yy > 0:
        a+=np.squeeze(-deepcopy(LLsurf[xx,yy-1,:]))
        ctr+=1
    if yy < sz[1]-1:
        a+=np.squeeze(-deepcopy(LLsurf[xx,yy+1,:]))
        ctr+=1

    return a/ctr

xx = 21
yy = 7

# LLth = deepcopy(LLsurf[xx,yy,:])

y = LLinterpolate(LLsurf, xx, yy)
# y = (y - np.min(y)) / (np.max(y) - np.min(y))
# y = y / np.sum(y)


plt.figure()
plt.plot(rotate, y)
plt.axvline(softmax(rotate,y, pow=10))


#%%
plt.figure(figsize=(Npos*2, Npos*2))
theta_peak = np.zeros((Npos, Npos))

for xx in range(Npos):
    for yy in range(Npos):
        plt.subplot(Npos, Npos, (Npos-1-yy)*Npos + xx + 1)

        LLth = LLinterpolate(LLsurf, xx, yy)
        # LLth = -deepcopy(LLsurf[xx,yy])
        plt.plot(rotate, LLth)
        theta_peak[xx,yy] = softmax(rotate,LLth, pow=10)
        plt.axvline(theta_peak[xx,yy])
        plt.axvline(0, color='k')
        plt.axis("off")


#%%
xx = np.meshgrid(locs, locs)
plt.figure(figsize=(10,10))
ax = plt.subplot(1,1,1)
sctr = ax.scatter(x=xx[0].flatten(), y=xx[1].flatten(), c=theta_peak.flatten(), cmap='RdYlGn')
plt.colorbar(sctr, ax=ax, format='%d',label='Optimal Rotation')
plt.xlabel('Horizontal eye position (d.v.a)')
plt.ylabel('Vertical eye position (d.v.a)')

#%%
# rpred = glm0.generate_prediction(input_data=[Xstim, Robs], ffnet_target=-1, layer_target=-1)
rpred1 = nim0.generate_prediction(input_data=[Xstim], ffnet_target=-1, layer_target=-1)
plt.figure(figsize=(10,4))
# f = plt.plot(rpred[:200])
f = plt.plot(rpred1[:1000])

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

stim_par = nim_par()
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
