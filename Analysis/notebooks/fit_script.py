# %% Import libraries
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

# %% list sessions
sesslist = gt.list_sessions()
sesslist = list(sesslist)
for i in range(len(sesslist)):
    print("%d %s" %(i, sesslist[i]))

# %% setup session
for iSess in range(1): #range(len(sesslist)):
    print("Session # %d" %iSess)
    indexlist = [iSess]

# indexlist = [10]
    try:
        stim, sacon, sacoff, Robs, DF, basis, opts, _ = gt.load_and_setup(indexlist,npow=1.8) #,num_saclags=60,back_shifts=20,num_lags=15)
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
        # %% fit models
        ndns, names = gt.fit_models(Xstim, Robs, XsacOn, opts,
            datapath='/home/jcbyts/Data/MitchellV1FreeViewing/',
            tag=opts['exname'][0]
            )
    except FileNotFoundError:
        print("Error loading session [%s]" %(sesslist[iSess]))


# %% Load one model
indexlist = [17]
stim, sacon, sacoff, Robs, DF, basis, opts, sacbc, valid, eyepos = gt.load_and_setup(indexlist,npow=1.8, opts={'padding':0})

# build time-embedded stimulus
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
XsacDur = NDNutils.create_time_embedding( sacbc, [3, 1, 1], tent_spacing=1)
Robs = Robs.astype('float32')
ed = np.hypot(eyepos[:,1], eyepos[:,2]) < 20
v = np.intersect1d(np.where(valid)[0], np.where(ed)[0])
# %% fit models
ndns, names = gt.fit_models(Xstim, Robs, XsacOn, XsacOff, XsacDur, opts,
    valid=v,
    datapath='/home/jake/Data/Datasets/MitchellV1FreeViewing/grating_analyses/lbfgs',
    tag=opts['exname'][0]
)

# %% evaluate models
Ti = opts['Ti']
LLx = []
for i in range(len(ndns)):
    print("%d) %s" %(i, names[i]))
    ndn0 = ndns[i].copy_model()
    l=len(ndn0.input_sizes)
    if l==1:
        LLx0 = ndn0.eval_models(input_data=[Xstim], output_data=Robs,
        data_indxs=Ti, nulladjusted=True)
    elif l==2:
        LLx0 = ndn0.eval_models(input_data=[Xstim, XsacOn], output_data=Robs,
        data_indxs=opts['Ti'], nulladjusted=True)
    elif l==3:
        LLx0 = ndn0.eval_models(input_data=[Xstim, XsacOn, XsacOff], output_data=Robs,
        data_indxs=opts['Ti'], nulladjusted=True)
    elif l==4:
        LLx0 = ndn0.eval_models(input_data=[Xstim, XsacOn, XsacOff, XsacDur], output_data=Robs,
        data_indxs=opts['Ti'], nulladjusted=True)

    LLx.append(LLx0)

#%% compare two models
plt.figure()
modi = 1
modj = 6
plt.plot(LLx[modi], LLx[modj], '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel(names[modi])
plt.ylabel(names[modj])

# %% plot learned RFs

i = 6
print(names[i])
filters = DU.compute_spatiotemporal_filters(ndns[i])
gt.plot_3dfilters(filters, basis=basis)
# %% plot gain and offset
modj = 6
xax = np.arange(-back_shifts, num_saclags-back_shifts, 1)
ix = LLx[modj]>0.5

plt.figure(figsize=(5,4))

if len(ndns[modj].networks)>3:
    plt.subplot(1,2,2)
    f = plt.plot(xax, ndns[modj].networks[2].layers[0].weights[:,ix], '#3ed8e6')
    plt.subplot(1,2,1)

f = plt.plot(xax, ndns[modj].networks[1].layers[0].weights[:,ix], '#3ed8e6')

#%%
plt.figure()
f = plt.plot(ndns[modj].networks[2].layers[0].weights)
# f = plt.plot(xax, ndns[modj].networks[2].layers[0].weights[:,ix], '#3ed8e6')

plt.figure()
f = plt.plot(ndns[modj].networks[3].layers[0].weights)
# f = plt.plot(xax, ndns[modj].networks[3].layers[0].weights[:,ix], '#3ed8e6')

#%%
plt.subplot(1,3,2)
f = plt.plot(xax, ndns[2].networks[2].layers[0].weights, 'b')
plt.subplot(1,3,3)
f = plt.plot(xax, ndns[1].networks[1].layers[0].weights, 'r')

# %% 
# f = plt.plot(ndns[2].networks[2].layers[0].weights)

f = plt.plot(ndns[1].networks[1].layers[0].weights)

# %%
# Rpred0 = ndns[0].generate_prediction(input_data=[Xstim])
Rpred1 = ndns[1].generate_prediction(input_data=[Xstim])
# Rpred2 = ndns[2].generate_prediction(input_data=[Xstim, XsacOn])

# sacta = XsacOn.T @ (Robs - np.average(Robs, axis=0))
# sacta0 = XsacOn.T @ (Rpred0 - np.average(Rpred0, axis=0))
# sacta1 = XsacOn.T @ (Rpred1 - np.average(Rpred1, axis=0))
# sacta2 = XsacOn.T @ (Rpred2 - np.average(Rpred2, axis=0))
ev = np.where(sacon)[0]
win = [-40,40]
NC = Robs.shape[1]
sx,sy = U.get_subplot_dims(NC)
plt.figure(figsize=(10,10))
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    m,xax,wf = gt.psth(Robs[:,cc], ev, win[0], win[1])
    # m0,xax,wf0 = gt.psth(Rpred0[:,cc], ev, win[0], win[1])
    m1,xax,wf1 = gt.psth(Rpred1[:,cc], ev, win[0], win[1])
    # m2,xax,wf2 = gt.psth(Rpred2[:,cc], ev,  win[0], win[1])

    plt.plot(xax, m, 'k')
    # plt.plot(xax, m0)
    plt.plot(xax, m1)
    # plt.plot(xax, m2)

    # plt.plot(xax, np.std(wf, axis=0), 'k')
    # plt.plot(xax, np.std(wf0, axis=0))
    # plt.plot(xax, np.std(wf1, axis=0))
    # plt.plot(xax, np.std(wf2, axis=0))


# %%
sx,sy = U.get_subplot_dims(NC)
ev = np.where(sacon)[0]
plt.figure(figsize=(10,10))
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    a,xax,wf = gt.psth(Robs[:,cc], ev, -100, 100)
    plt.plot(xax, a)
    plt.plot(xax, np.std(wf, axis=0))

# %%
# cc = cc+1
Rpred0 = ndns[1].generate_prediction(input_data=[Xstim])

cc =26
Ti = opts['Ti']
r = np.reshape(Robs[Ti,cc], (opts['num_repeats'],-1))
r0 = np.reshape(Rpred0[Ti,cc], (opts['num_repeats'],-1))
r = np.average(r, axis=0)
r0 = np.average(r0, axis=0)
plt.plot(r)
plt.plot(r0)

U.r_squared(np.reshape(r, (-1,1)), np.reshape(r0, (-1,1)))

# %%

son = np.where(sacon)[0]
soff = np.where(sacoff)[0]



# %%
soff[0:4000]-son[0:4000]

# %%
opts

# %%
