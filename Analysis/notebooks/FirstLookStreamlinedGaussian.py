#%% set paths
sys.path.insert(0, '/home/jcbyts/Repos/')
import Utils as U
import gratings as gt

import NDN3.NDNutils as NDNutils

which_gpu = NDNutils.assign_gpu()

from copy import deepcopy

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt  # plotting
import seaborn as sns

import NDN3.NDN as NDN
import NDN3.Utils.DanUtils as DU
import NDN3.Utils.NDNplot as NDNplot
import importlib # for reimporting modules
output_dir = '/home/jcbyts/PyPlay/tensorboard' + str(which_gpu)
print(output_dir)

#%% load sessions
sesslist = gt.list_sessions()
sesslist = list(sesslist)

indexlist = [14]

# TODO: use this to find groups where the stimulus was the same
sessmincpd = []
sessmaxcpd = []
for i in indexlist:
    matdat = gt.load_data(sesslist[i])
    sessmincpd.append(np.min(matdat['grating']['cpd'][matdat['grating']['cpd']>0.0]))
    sessmaxcpd.append(np.max(matdat['grating']['cpd']))

sessmincpd = np.min(np.asarray(sessmincpd))
sessmaxcpd = np.max(np.asarray(sessmaxcpd))

npow = 1.8
nsteps = np.ceil((np.log10(sessmaxcpd) - np.log10(sessmincpd)) / np.log10(npow))

ymax = gt.invnl(nsteps-1,sessmincpd,npow)
print(ymax*1.5)

#%% load data
sess = [sesslist[i] for i in indexlist]

# shared basis
basis = {'name': 'cosine', 'nori': 8, 'nsf': int(nsteps), 'endpoints': [sessmincpd, npow], 'support': 500}

stim, sacon, sacoff, Robs, DF, basis, opts,_ = gt.load_sessions(sess, basis=basis) 

#%% plot basis

gt.plot_basis(basis)

# %% create time-embedded design matrix
from scipy.ndimage import gaussian_filter

num_saclags = 60
back_shifts = 20
num_time_shift = 4
num_lags = 15
NX = basis['nori']
NY = basis['nsf']
NT,NC=Robs.shape

# build time-embedded stimulus
Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1 )
XsacOn = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacon,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
XsacOff = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacoff,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
XsacOnCausal = NDNutils.create_time_embedding( sacon, [num_saclags, 1, 1], tent_spacing=1)
XsacOffCausal = NDNutils.create_time_embedding( sacoff, [num_saclags, 1, 1], tent_spacing=1)
Robs = Robs.astype('float32')

# training data is smoothed spikes (does it help with fitting?)
RobsTrain = gaussian_filter(Robs.copy(), [1.0, 0.0]).astype('float32')

valdata = np.arange(0,NT,1) # is used

# build train, validate, test indices (use frozen trials if they exist)
Ui, Xi = NDNutils.generate_xv_folds(NT, num_blocks=2)

grating = gt.load_data(sess[0]) # reload data

if len(grating['grating']['frozen_repeats']) > 0:
    print("Using frozen repeats as test set")
    has_frozen = True
    Ti = np.reshape(grating['grating']['frozen_repeats'], (-1, grating['grating']['frozen_seq_dur'][0]+1)).astype(int)
    num_repeats = Ti.shape[0]
    Ti = Ti.flatten()
else:
    # make test indices from the training / validation set
    Ti = np.concatenate((Ui[:Ui.shape[0]//20], Xi[:Xi.shape[0]//10])).astype(int)
    has_frozen = False
Ui = np.setdiff1d(Ui, Ti).astype(int)
Xi = np.setdiff1d(Xi, Ti).astype(int)

# %% get STA
stas = Xstim.T @ (Robs - np.average(Robs, axis=0))
stas = stas.reshape([NX*NY,num_lags,NC])

# plot STA
sx, sy = U.get_subplot_dims(NC)
plt.figure(figsize=(8.5,10))
for cc in range(NC):
    ax = plt.subplot(sx,sy,cc+1)
    plt.plot(np.transpose(stas[:,:,cc]))
    plt.title('cell'+str(cc),)
sns.despine(offset=0, trim=True)
plt.show()


# %% saccade-triggered average
delta = 1/0.0083
sacta = XsacOn.T @ RobsTrain / np.sum(XsacOn) * delta
sacta /= np.mean(sacta, axis=0)
xax = np.arange(-back_shifts, num_saclags-back_shifts, 1)

# plot STA
sx, sy = U.get_subplot_dims(NC)
plt.figure(figsize=(8.5,10))
for cc in range(NC):
    ax = plt.subplot(sx,sy,cc+1)
    plt.plot(xax,sacta[:,cc])
    plt.title('cell'+str(cc),)
sns.despine(offset=0, trim=True)
plt.show()


#%% optimization params
adam_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')

early_stopping = 100

adam_params['batch_size'] = 5000
adam_params['display'] = 30
adam_params['MAPest'] = True
adam_params['epochs_training'] = 1000
adam_params['early_stop'] = early_stopping
adam_params['early_stop_mode'] = 1
adam_params['epsilon'] = 1e-3
adam_params['data_pipe_type'] = 'data_as_var' # 'feed_dict'
adam_params['learning_rate'] = 1e-2

lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True, 'display': True}, learning_alg='lbfgs')
lbfgs_params['maxiter'] = 10000


noise_dist = 'poisson'
seed = 5
optimizer = 'adam'


if noise_dist=='poisson':
    null_adjusted = True
else:
    null_adjusted = False

if optimizer=='adam':
    opt_params = adam_params
else:
    opt_params = lbfgs_params

#%% Fit Full GLM
Xreg = None
L2reg = 1e-2

useSep = False
if useSep:
    glmFull_par = NDNutils.ffnetwork_params( 
        input_dims=[num_lags,NX,NY], layer_sizes=[NC], 
        layer_types=['sep'], act_funcs=['softplus'], 
        ei_layers=[None], normalization=[-1], # normalization really matters
        reg_list={'d2x':[Xreg],'l2':[L2reg],'max_filt':[None]} )
else:
    glmFull_par = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags], layer_sizes=[NC], 
        layer_types=['normal'], act_funcs=['softplus'], 
        normalization=[-1], # normalization really matters
        reg_list={'d2xt':[Xreg],'l2':[L2reg], 'local':[None]} )

glmFull = NDN.NDN( [glmFull_par], tf_seed=seed, noise_dist=noise_dist)

glmFull.time_spread = num_lags+num_saclags+num_time_shift

# initialize with STA
stas = Xstim.T @ (Robs - np.average(Robs, axis=0)) 
stas = stas / np.sum(stas, axis=0)

if not useSep:
    glmFull.networks[0].layers[0].weights = stas.astype('float32')
    glmFull.networks[0].layers[0].biases = np.average(Robs[Ui,:], axis=0).astype('float32')

_ = glmFull.train(input_data=[Xstim], output_data=RobsTrain, train_indxs=Ui, test_indxs=Xi,
    learning_alg=optimizer, opt_params=opt_params, use_dropout=False)

# use data filters
# _ = glmFull.train(input_data=[Xstim], output_data=RobsTrain, data_filters=DF, train_indxs=Ui, test_indxs=Xi,
#     learning_alg=optimizer, opt_params=opt_params, use_dropout=False)


#%%

LLx0 = glmFull.eval_models(input_data=[Xstim], output_data=Robs, data_indxs=Ti, data_filters=DF, nulladjusted=null_adjusted)
print("Test-likelihoods (DF):")
print(LLx0)

LLx0 = glmFull.eval_models(input_data=[Xstim], output_data=Robs, data_indxs=Ti, nulladjusted=null_adjusted)

print("Test-likelihoods:")
print(LLx0)

# g = glmFull.generate_prediction(input_data=Xstim, pre_activation=True)
Rpred0 = glmFull.generate_prediction(input_data=Xstim, pre_activation=False)

print("r-squared:")
r20 = U.r_squared(RobsTrain, Rpred0, data_indxs=Ti)

print(r20)
#%% plot it 
if glmFull.network_list[0]['layer_types'][0]=='sep':
    w = deepcopy(glmFull.networks[0].layers[0].weights)
    time = w[0:num_lags,:]
    space = w[num_lags:,:]
    Nrows = 8
    Ncols = NC // (Nrows-1)
    plt.figure(figsize=(10,12))
    for cc in range(NC):
        plt.subplot(Nrows,Ncols,cc+1)
        plt.plot(time[:,cc])
    sns.despine(trim=False, offset=10)
    plt.figure(figsize=(10,12))
    for cc in range(NC):
        plt.subplot(Nrows,Ncols,cc+1)
        plt.imshow(np.reshape(basis['B'] @ space[:,cc], (basis['support'],basis['support'])))
else:
    print("GLM")
    filters = DU.compute_spatiotemporal_filters(glmFull)
    gt.plot_3dfilters(filters, basis=basis)

print("STAS")
gt.plot_3dfilters(np.reshape(stas, filters.shape), basis=basis)

#%% Low-rank filters
w = glmFull.networks[0].layers[0].weights
u,s,v = np.linalg.svd(w, full_matrices=False)

plt.plot(np.cumsum(s)/np.sum(s), '-o')
plt.xlabel('Rank')
plt.ylabel('Var Explained Weights')

# %% Shared GLM
Treg = 0.01
Xreg = 1e-8
L2reg = 1e-2
num_tkerns = 0
glmShare = []
LLxShare = []
subs_num = [NC//2] #np.arange(2,15,2)
for num_subs in list(subs_num):
    print('Fitting %d-dimensional shared GLM:' %(num_subs))
    
    if num_tkerns > 0: # use temporal convolution layer?
        glmShare_par = NDNutils.ffnetwork_params( 
            input_dims=[1,NX,NY,num_lags], layer_sizes=[num_tkerns, num_subs, NC], 
            layer_types=['normal', 'normal','normal'], act_funcs=['lin', 'lin','softplus'], 
            ei_layers=[None,None], normalization=[1,1,0], conv_filter_widths=[1],
            reg_list={'orth': [1e-1, 1e-1], 'd2x':[None, Xreg],'l2':[None, None,L2reg],'max_filt':[None,None]} )
        glmShare0 = NDN.NDN( [glmShare_par], tf_seed = seed, noise_dist=noise_dist)
    else:
        glmShare_par = NDNutils.ffnetwork_params( 
        input_dims=[1,NX,NY,num_lags], layer_sizes=[num_subs, NC], time_expand=[0],
        layer_types=['normal','normal'], act_funcs=['relu','softplus'], 
        ei_layers=[None,None], normalization=[-1, -1],
        reg_list={'orth': None,'d2x':[Xreg],'l2':[L2reg,L2reg],'local':[1e-3,None]} )
        glmShare0 = NDN.NDN( [glmShare_par], tf_seed = seed, noise_dist=noise_dist)
        # initialize based on 
        # glmShare0.networks[0].layers[0].weights = deepcopy(u[:,:num_subs])
        
    glmShare0.networks[-1].layers[-1].biases = deepcopy(glmFull.networks[-1].layers[-1].biases)
    v2f0 = glmShare0.fit_variables(fit_biases=False)
    v2f0[-1][-1]['biases']=True

    glmShare0.time_spread = num_lags+num_saclags+num_time_shift

    # train
    _ = glmShare0.train(input_data=[Xstim], output_data=Robs, data_filters=DF, train_indxs=Ui, test_indxs=Xi,
        learning_alg=optimizer, opt_params=opt_params, fit_variables=v2f0)
    
    
    LLx1 = glmShare0.eval_models(input_data=[Xstim], output_data=Robs, data_indxs=Xi, nulladjusted=null_adjusted)
    # r-squared
    # Rpred1 = glmShare0.generate_prediction(input_data=Xstim)
    # r2 = r_squared(Robs, Rpred1, data_indxs=Xi)
    
    glmShare.append(glmShare0.copy_model())
    LLxShare.append(LLx1)
    
LLxShare = np.asarray(LLxShare)

# #%% Plot test-likelihoods as a function of the number of dimensions
# plt.plot(np.array((1.0, np.max(subs_num))), np.mean(LLx0)*np.array((1.0, 1.0)), 'k')
# plt.plot(subs_num, np.mean(LLxShare, axis=1), '-o')
# plt.ylabel('Average LL')
# plt.xlabel("Subspace Rank")
# sns.despine()

# LLx1 = LLxShare[1,:]
#%%
LLx0 = glmFull.eval_models(input_data=[Xstim], output_data=Robs, data_indxs=Ti, nulladjusted=null_adjusted)
LLx1 = glmShare0.eval_models(input_data=[Xstim], output_data=Robs, data_indxs=Ti, nulladjusted=null_adjusted)

plt.figure()
plt.plot(LLx0, LLx1, '.')
plt.plot([0.0, .2], [0.0, .2], 'k')
plt.xlabel('Full GLM')
plt.ylabel('Low-Rank GLM')


filters = DU.compute_spatiotemporal_filters(glmShare0)
gt.plot_3dfilters(filters, basis=basis)

# #%%

# glmShare0 = glmShare[1].copy_model()

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

# %% add saccade kernels
Treg = 0.001
L2reg = 1e-2
stim_par = glmShare0.network_list
stim_par = stim_par[0]
stim_par['xstim_n'] = [0]

sacshift = NDNutils.shift_mat_zpad(sacon,-back_shifts,dim=0)

sacGlmAdd = []
sacGlmAddC = []
LLxSac = []
LLxSacC = []

# loop over number of saccade kernels shared by population
sac_num_dim = np.arange(2,3,1)
for num_sac_dim in list(sac_num_dim):
    print('Fitting %d-dimensional saccade kernel:' %(num_sac_dim))
    sac_on_full = NDNutils.ffnetwork_params(
        input_dims=[1,1,1,num_saclags], xstim_n=[1],
        layer_sizes=[num_sac_dim, NC], layer_types=['normal', 'normal'],
        normalization=[1], act_funcs=['lin', 'lin'],
        reg_list={'d2t':[Treg], 'orth':[1e-1], 'l2':[L2reg]}
    )
    
    comb_par = NDNutils.ffnetwork_params(
        xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],
        layer_types=['add'], act_funcs=['softplus']
    )

    sacglmadd = NDN.NDN([stim_par, sac_on_full, comb_par], ffnet_out=2,
        noise_dist=noise_dist)

    # copy RF parameters from shared GLM
    sacglmadd.networks[0].layers[0].weights = deepcopy(glmShare0.networks[0].layers[0].weights)
    sacglmadd.networks[0].layers[1].weights = deepcopy(glmShare0.networks[0].layers[1].weights)
    sacglmadd.networks[0].layers[0].biases = deepcopy(glmShare0.networks[0].layers[0].biases)
    sacglmadd.networks[0].layers[1].biases = deepcopy(glmShare0.networks[0].layers[1].biases)

    # don't refit the RFs
    v2f0 = sacglmadd.fit_variables(layers_to_skip=[[0,1]], fit_biases=False)
    v2f0[-1][0]['biases']=True

    sacglmadd.time_spread = num_lags+num_saclags+num_time_shift
    sacaddcaus = sacglmadd.copy_model() # causal model is identical, but the input is different

    # train
    _ = sacglmadd.train(
        input_data=[Xstim, XsacOn], output_data=RobsTrain, #data_filters=DF,
        train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
        learning_alg=optimizer, opt_params=opt_params
    )
    
    _ = sacaddcaus.train(
        input_data=[Xstim, XsacOnCausal], output_data=RobsTrain,# data_filters=DF,
        train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
        learning_alg=optimizer, opt_params=opt_params
    )

    # now fit everything
    v2f0 = sacglmadd.fit_variables(fit_biases=False)
    v2f0[-1][0]['biases']=True

    _ = sacglmadd.train(
        input_data=[Xstim, XsacOn], output_data=RobsTrain, #data_filters=DF,
        train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
        learning_alg=optimizer, opt_params=opt_params
    )
    
    _ = sacaddcaus.train(
        input_data=[Xstim, XsacOnCausal], output_data=RobsTrain,# data_filters=DF,
        train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
        learning_alg=optimizer, opt_params=opt_params
    )
    
    
    # Rpred1 = sacglmadd.generate_prediction(input_data=[Xstim, sacshift])
    # LLxs1 = r_squared(Robs, Rpred1, data_indxs=Xi)

    # Rpred2 = sacaddcaus.generate_prediction(input_data=[Xstim, sacon])
    # LLxs1C = r_squared(Robs, Rpred2, data_indxs=Xi)

    # LLxs1 = sacglmadd.eval_models(input_data=[Xstim, sacshift], output_data=Robs, data_filters=DF,
    #                     data_indxs=Xi, nulladjusted=True)
    # LLxs1C = sacaddcaus.eval_models(input_data=[Xstim, sacon], output_data=Robs, data_filters=DF,
    #                     data_indxs=Xi, nulladjusted=True)

    LLxs1 = sacglmadd.eval_models(input_data=[Xstim, XsacOn], output_data=Robs,
                        data_indxs=Ti, nulladjusted=True)
    LLxs1C = sacaddcaus.eval_models(input_data=[Xstim, XsacOnCausal], output_data=Robs,
                        data_indxs=Ti, nulladjusted=True)

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
plt.plot(xax, sacGlmAdd[0].networks[1].layers[0].weights)
plt.xlabel("Time from saccade onset")
plt.subplot(1,2,2)
xax = np.linspace(1, num_saclags, num_saclags)*8
plt.plot(xax,sacGlmAddC[0].networks[1].layers[0].weights)
plt.xlabel("Time from saccade onset")

plt.figure()
plt.plot(LLxSacC[1,:], LLxSac[1,:], '.')
plt.plot([0,1], [0,1], '-k')
plt.xlabel("Causal Only")
plt.ylabel("Presaccadic included")
sns.despine(offset=10)


#%% plot it all
binsize=1e3/opts['frate']
plt.figure()
filters = DU.compute_spatiotemporal_filters(sacglmadd)
gt.plot_3dfilters(filters, basis=basis)
plt.figure()
filters = DU.compute_spatiotemporal_filters(sacaddcaus)
gt.plot_3dfilters(filters, basis=basis)
num_time_shift = 0

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
                        data_indxs=Ti, nulladjusted=True)

LLxStim = glmShare0.eval_models(input_data=[Xstim], output_data=Robs, data_filters=DF,
                        data_indxs=Ti, nulladjusted=True)
LLxsac = sacglmadd.eval_models(input_data=[Xstim, XsacOn], output_data=Robs, data_filters=DF,
                        data_indxs=Ti, nulladjusted=True)

LLxsacc = sacaddcaus.eval_models(input_data=[Xstim, XsacOnCausal], output_data=Robs, data_filters=DF,
                        data_indxs=Ti, nulladjusted=True)
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
    xstim_n=None, ffnet_n=[0,1], layer_sizes=[NC],normalization=[-1],
    layer_types=['mult'], act_funcs=['lin']
)

sac_on_full_mult = NDNutils.ffnetwork_params(
    input_dims=[1,1,1, num_saclags], xstim_n=[1],
    layer_sizes=[2, NC], layer_types=['normal', 'normal'],
    normalization=[-1,-1], act_funcs=['lin', 'lin'],
    reg_list={'orth':[.1],'d2t':[Treg], 'l2':[1e-2,1e-2]}
)

sacglmmult = NDN.NDN([stim_par, sac_on_full_mult, sac_on_full, mult_par, add_par], ffnet_out=4,
    noise_dist=noise_dist)

sacglmmult.time_spread = sacglmadd.time_spread
# inititialize stimulus
sacglmmult.networks[0].layers[0].weights = deepcopy(sacglmadd.networks[0].layers[0].weights)
sacglmmult.networks[0].layers[1].weights = deepcopy(sacglmadd.networks[0].layers[1].weights)
sacglmmult.networks[0].layers[0].biases = deepcopy(sacglmadd.networks[0].layers[0].biases)
sacglmmult.networks[0].layers[1].biases = deepcopy(sacglmadd.networks[0].layers[1].biases)

sacglmmult.networks[1].layers[0].weights *=0
sacglmmult.networks[1].layers[1].weights *=0

sacglmmult.networks[2].layers[0].weights = deepcopy(sacglmadd.networks[1].layers[0].weights)
sacglmmult.networks[2].layers[1].weights = deepcopy(sacglmadd.networks[1].layers[1].weights)

# specify variables to fit
v2f0 = sacglmmult.fit_variables(layers_to_skip=[[0,1],[0,1],[0]], fit_biases=False)
v2f0[-1][0]['biases']=True # fit last layer biases

#  pre-train (get biases and gains in right place)
_ = sacglmmult.train(
    input_data=[Xstim, XsacOn], output_data=RobsTrain, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)

# fit mult kernel (fixing all else)
v2f0 = sacglmmult.fit_variables(layers_to_skip=[[0,1],[],[1]], fit_biases=False)
v2f0[-1][0]['biases']=True # fit last layer biases

_ = sacglmmult.train(
    input_data=[Xstim, XsacOn], output_data=RobsTrain, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f0,
    learning_alg='adam', opt_params=adam_params
)

w1m = sacglmmult.networks[1].layers[0].weights
plt.plot(w1m)

#%% train 2, free all variables


v2f1 = sacglmmult.fit_variables(fit_biases=False)
v2f1[-1][0]['biases']=True # fit last layer biases
_ = sacglmmult.train(
    input_data=[Xstim, XsacOn], output_data=RobsTrain, data_filters=DF,
    train_indxs=Ui, test_indxs=Xi, fit_variables=v2f1,
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
plt.subplot(1,2,1)
wm = w1m@w2m
p = plt.plot(xax,wm)

plt.subplot(1,2,2)
wa = w1a@w2a
p = plt.plot(xax,wa)

sns.despine()

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

# %%
