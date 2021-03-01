#%% Import / Setup
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

figdir = '/home/jake/Data/Repos/V1FreeViewingCode/Figures/eye_corretion_affine_test'

# %% set paths for data
dirname = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'

# flist = ['logan_20200304_Gabor_-20_-10_40_60_2_2_0_9_0.mat',
#     'logan_20200304_Grating_-20_-10_40_60_2_2_0_9_0.mat',
#     'logan_20200304_FixRsvpStim_-20_-10_40_60_2_2_0_9_0.mat',
#     'logan_20200304_BackImage_-20_-10_40_60_2_2_0_9_0.mat']
# exname = 'logan_20200304'

flist = ['logan_20200306_Gabor_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200306_Grating_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200306_FixRsvpStim_-20_-10_40_60_2_2_0_9_0.mat',
    'logan_20200306_BackImage_-20_-10_40_60_2_2_0_9_0.mat']

exname = 'logan_20200306'

import importlib
importlib.reload(ne)

# %% Load Data

Stim,Robs,valdat,labels,NX,NY,dt,eyeAtFrame,frameTime = ne.load_stim_files(dirname=dirname, flist=flist[0:2], corrected=True)

Stim /= np.nanstd(Stim) # normalize stimulus (necessary?)


# %% get valid indices
valdata = np.intersect1d(np.where(valdat[:,0] == 1)[0], np.where(labels[:,0] == 1)[0]) # fixations / valid

valid_eye_rad = 5.2  # degrees -- use this when looking at eye-calibration (see below)
ppd = 37.50476617061

eyeX = (eyeAtFrame[:,0]-640)/ppd
eyeY = (eyeAtFrame[:,1]-380)/ppd

eyeCentered = np.hypot(eyeX, eyeY) < valid_eye_rad

valdata = np.intersect1d(valdata, np.where(eyeCentered)[0])

blockstart = np.append(1, np.where(np.diff(valdata)>1)[0])
blockstop = np.append(np.where(np.diff(valdata)>1)[0]-1,valdata[-1])
blocks = np.stack((blockstart, blockstop)).T
# %% quick check STAS
num_lags = 10
dims = [num_lags, NX, NY]
Xstim, rinds = ne.create_time_embedding_valid(Stim, dims, valdata)
Rvalid = deepcopy(Robs[rinds,:])
NC = Robs.shape[1]

stas = Xstim.T@ (Rvalid - np.mean(Rvalid))
stas = np.reshape(stas, (NX*NY, num_lags, NC))

plt.figure(figsize=(10,10))

sx,sy = U.get_subplot_dims(NC)
sumdensity = 0
mu = np.zeros(NC)
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    plt.imshow(np.reshape(stas[:,bestlag,cc], (NY,NX)))
    sumdensity += np.abs(stas[:,bestlag,cc])
    tlevel = np.median(np.abs(stas[:,:,cc]-np.average(stas[:,:,cc])))*6
    mu[cc] = np.average(np.abs(stas[:,:,cc])>tlevel)
    plt.title(cc)
    plt.axis("off")
    
plt.figure()
plt.imshow(np.reshape(sumdensity, (NY, NX)))

#%% threshold good STAS
thresh = 0.51
    
plt.figure()
plt.plot(mu, '-o')
plt.axhline(thresh, color='k')
plt.show()

cids = np.where(mu > thresh)[0] # units to analyze

print("%d/%d will go into the model" %(len(cids), len(mu)))
# %% get crop indices
NX2 = 21 # must be odd
inds, wts = ne.get_crop_indices(np.array((NX,NY)), outsize=np.array((NX2,NX2)),
    translation=np.array((-1,0)),
    theta=0,
    scale=1.0)

plt.figure(figsize=(10,10))
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    sta = stas[:,bestlag,cc]
    I = 0
    for i in range(4):
        I = I + sta[inds[i]]*wts[i]

    plt.imshow(np.reshape(I, [NX2,NX2]))

# %% Get stimulus model

adam_params = U.def_adam_params()

# apply interpolated crop
stmp = 0
for i in range(4):
    stmp += Stim[:,inds[i]]*wts[i]

# make time-embedded cropped stimulus
Xstim, rinds = ne.create_time_embedding_valid(stmp, [num_lags, NX2, NX2], valdata)

# index into Robs
Rvalid = deepcopy(Robs[rinds,:])
Rvalid = Rvalid[:,cids]
Rvalid = NDNutils.shift_mat_zpad(Rvalid,-1,dim=0) # get rid of first lag

NC = Rvalid.shape[1] # new number of units
NT = Rvalid.shape[0]
print('%d valid samples of %d possible' %(NT, Stim.shape[0]))
print('%d good units' %NC)
    
# double-check STAS work with cropped stimulus
stas = Xstim.T@(Rvalid - np.mean(Rvalid))
stas = np.reshape(stas, [NX2*NX2, num_lags, NC])/NT
    
plt.figure(figsize=(10,15))
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    bestlag = np.argmax(np.max(abs(stas[:,:,cc]),axis=0))
    # plt.plot(stas[:,:,cc].T)
    plt.imshow(np.reshape(stas[:,bestlag,cc], (NX2,NX2)))
    plt.title(cc)
plt.show()

#%%
Rvalid = deepcopy(Robs[:,cids])
# train, test indices    
Ui, Xi = NDNutils.generate_xv_folds(NT)
    
# fit SCAFFOLD MODEL
    
# Best regularization arrived at
Greg0 = 1e-1
Mreg0 = 1e-6
L1reg0 = 1e-5

XTreg=0.05
L1reg=5e-3
MIreg=0.1
MSCreg=10.0
Greg=0.1
Mreg=1e-4


num_subs=NC
num_hid=num_subs//2
        
# ndn_par = NDNutils.ffnetwork_params( 
#     input_dims=[1,NX2,NX2, num_lags],
#     layer_sizes=[num_subs, num_hid], 
#     layer_types=['normal','normal'], ei_layers=[num_subs//2, num_hid//2],
#     normalization=[1, 1], act_funcs=['relu', 'relu'], verbose=True,
#     reg_list={'d2xt':[XTreg], 'l1':[L1reg0, L1reg0], 'glocal':[Greg0]})

t_layer = NDNutils.ffnetwork_params(
    input_dims=[1,NX2,NX2], time_expand=num_lags,
    layer_sizes=[1],
    layer_types=['temporal'],
    normalization=[1],
    act_funcs=['lin'],
    verbose=True,
    reg_list={'d2t':[0.05]})

ndn_par = NDNutils.ffnetwork_params( 
    xstim_n=None,
    ffnet_n=0,
    input_dims=[1,NX2,NX2],
    layer_sizes=[num_subs, num_hid], 
    layer_types=['normal','normal'],
    normalization=[1, 1],
    act_funcs=['relu', 'relu'],
    verbose=True,
    reg_list={'d2x':[XTreg], 'l1':[L1reg0, L1reg0], 'glocal':[Greg0]})


side_par = NDNutils.ffnetwork_params( 
    network_type='side', xstim_n=None, ffnet_n=1, layer_sizes=[NC], 
    layer_types=['normal'], normalization=[-1], act_funcs=['softplus'], verbose=True,
    reg_list={'max':[Mreg0]})

side_par['pos_constraints']=True

side2 = NDN.NDN( [t_layer, ndn_par, side_par], ffnet_out=2, noise_dist='poisson')

#%%
gab_array = DU.gabor_array(NX2//2, num_angles=num_subs//2, both_phases=True)
side2.networks[1].layers[0].weights = deepcopy(gab_array)
            
input_data = stmp

NumBlocks = blocks.shape[0]
bad_blocks = np.where((blocks[:,1]-blocks[:,0]) < 10)[0]
good_blocks = np.setdiff1d(np.arange(0, NumBlocks-1), bad_blocks)
blocks = blocks[good_blocks,:]
NumBlocks = blocks.shape[0]

Ui,Xi = NDNutils.generate_xv_folds(NumBlocks)
_ = side2.train(input_data=input_data, output_data=Rvalid, train_indxs=Ui, test_indxs=Xi, silent=False, 
    learning_alg='adam', opt_params=adam_params, blocks=blocks+1)

# adjust regularization and re-train
side2.set_regularization('glocal', Greg, ffnet_target=1, layer_target=0)
side2.set_regularization('l1', L1reg, ffnet_target=1, layer_target=0)
side2.set_regularization('max', MIreg, ffnet_target=1, layer_target=1)
side2.set_regularization('max', MSCreg, ffnet_target=2, layer_target=0)

side2b = side2.copy_model()

_ = side2b.train(input_data=input_data, output_data=Rvalid, train_indxs=Ui, test_indxs=Xi, silent=False, 
    learning_alg='adam', opt_params=adam_params, blocks=blocks)

LLs2n = side2b.eval_models(input_data=input_data, output_data=Rvalid, data_indxs=Xi, nulladjusted=True, blocks=blocks)
print(np.mean(LLs2n))
#%%
plt.hist(LLs2n)
plt.xlabel('Nats/Spike')
plt.show()
plt.savefig(figdir + '/' + exname + 'testLLraw.pdf', format='pdf')

#%%
DU.plot_filters(side2b)
# %% convert model to be convolutional (make spatial expand fast!)

base_mod = side2b.copy_model()

# old network list
netlist_old = deepcopy(base_mod.network_list)[0]

use_tlayer = True

core_net = 1
side_net = 2

pre_net = deepcopy(base_mod.network_list[0])
pre_net['input_dims'] = [1, NX, NY]

# convert scaffold network into a convolutional network
scaff_par = deepcopy(base_mod.network_list[core_net])
scaff_par['input_dims'] = [1, NX, NY]
scaff_par['layer_types']=['conv', 'conv']
scaff_par['conv_filter_widths'] = [netlist_old['input_dims'][1], 1] # base_mod 

side_par = deepcopy(base_mod.network_list[side_net])
side_par['layer_types'] = ['conv']
side_par['conv_filter_widths'] = [1]

cell_shift_mod = NDN.NDN( [pre_net, scaff_par, side_par], ffnet_out=2, noise_dist='poisson' )
# copy first network verbatim (only thing diff is output is a convolution)
cell_shift_mod.networks[core_net].layers[0].weights = deepcopy(base_mod.networks[core_net].layers[0].weights)
cell_shift_mod.networks[core_net].layers[0].biases = deepcopy(base_mod.networks[core_net].layers[0].biases)
cell_shift_mod.networks[core_net].layers[1].weights = deepcopy(base_mod.networks[core_net].layers[1].weights)
cell_shift_mod.networks[core_net].layers[1].biases = deepcopy(base_mod.networks[core_net].layers[1].biases)
cell_shift_mod.networks[side_net].layers[0].weights = deepcopy(base_mod.networks[side_net].layers[0].weights)
cell_shift_mod.networks[side_net].layers[0].biases = deepcopy(base_mod.networks[side_net].layers[0].biases)

num_space = np.prod(cell_shift_mod.input_sizes[0][:])

#%% Run eye correction
from tqdm import tqdm # progress bar

base_mod = side2b.copy_model()

eyeAtFrameCentered = (eyeAtFrame-(640, 380))
eyeX = deepcopy(eyeAtFrameCentered[:,0])
eyeY = deepcopy(eyeAtFrameCentered[:,1])

max_rotate = 10
nsteps = 10
rotate = np.linspace(-max_rotate, max_rotate, nsteps)

valid_eye_rad=5.2*ppd
Erad=0.5*ppd
Npos=25

locs = np.linspace(-valid_eye_rad, valid_eye_rad, Npos)

LLsurf = np.zeros([Npos, Npos, NY, NX, nsteps])

# mean eye position in each fixation
eyeXb = np.asarray([np.mean(eyeX[blocks[i,0]:blocks[i,1]]) for i in range(blocks.shape[0])])
eyeYb = np.asarray([np.mean(eyeY[blocks[i,0]:blocks[i,1]]) for i in range(blocks.shape[0])])


# Loop over positions (this is the main time-consuming operation)
for ith in tqdm(range(nsteps)):

    th = rotate[ith]

    # get crop indices
    inds, wts = ne.get_crop_indices(np.array((NX,NY)), outsize=np.array((NX,NY)),
        translation=np.array((0,0)),
        theta=th,
        scale=1.0)

    # inds, wts = ne.get_crop_indices(np.array((NX,NY)), outsize=np.array((NX2,NX2)),
    #     translation=np.array((0,0)),
    #     theta=th,
    #     scale=1.0)

    # apply interpolated crop
    stmp = 0
    for i in range(4):
        stmp += Stim[:,inds[i]]*wts[i]

    
    for xx in range(Npos):
        for yy in range(Npos):

            print(yy)
            # get index for when the eye position was withing the boundaries
            rs = np.hypot(eyeXb-locs[xx], eyeYb-locs[yy])
            
            # eccentricity dependent 
            ecc = np.hypot(locs[xx],locs[yy])
            Ethresh = Erad + .2*ecc # eccentricity dependent threshold
                
            valE = np.where(rs < Ethresh)[0]
            valtot = np.random.choice(valE, np.minimum(len(valE), 50))

            idxs = [np.arange(blocks[i,0], blocks[i,1]) for i in valtot]
            nblocks = len(idxs)
            n = np.asarray([len(idxs[i]) for i in range(nblocks)]).cumsum()
            bstarts = np.append(1, n[np.arange(0,nblocks-1)]+1)
            bstops = n[np.arange(0,nblocks)]
            bblocks = np.stack( (bstarts, bstops)).T
            
            idxs = np.concatenate(idxs)

            Rcc = ne.conv_expand( Rvalid[idxs,:], num_space) # risky with memory load

            if len(valtot) > 10: # at least 10 fixations
                
                # LLs = base_mod.eval_models(input_data=stmp, output_data=Rvalid, nulladjusted=False, data_indxs=valtot, blocks=blocks)
                # LLs = np.mean(LLs)
                
                LLs = cell_shift_mod.eval_models(input_data=stmp[idxs,:], output_data=Rcc, nulladjusted=False, blocks=bblocks)

                # reshape into spatial map
                LLcc = np.reshape(LLs, [NY,NX,NC])

                LLpos = np.mean(LLcc,axis=2)
                # if crop_edge == 0:
                #     LLspace1[xx,yy,:,:] = deepcopy(LLpos)
                # else:
                #     LLspace1[xx,yy,:,:] = deepcopy(LLpos)[crop_edge:-crop_edge,:][:,crop_edge:-crop_edge]

                LLsurf[xx,yy,:,:,ith] = deepcopy(LLpos)

#%%
import scipy.io as sio
outname = exname + 'LLsurfTest.mat'
sio.savemat(dirname + outname, {'LLsurf': LLsurf})

#%%
dat = sio.loadmat(dirname+outname)

#%%
xx = 21
yy = 7

I0 = np.mean(deepcopy(LLsurf[xx,yy,:,:,:]), axis=2)
am = np.argmax(-I0)
xi = am % NX
yi = (am - xi)//NX

plt.figure()
plt.imshow(-I0)
plt.plot(xi,yi,'.r')

plt.figure(figsize=(10,4))
m = np.zeros(nsteps)
for ith in range(nsteps):
    plt.subplot(3,4,ith+1)
    I = np.squeeze(deepcopy(LLsurf[xx,yy,:,:,ith]))
    plt.imshow(-I)
    plt.plot(xi,yi,'.r')
    plt.title("%02.1f" %rotate[ith])
    plt.axis("off")
    m[ith] = np.max(-I.flatten())

plt.savefig(figdir + '/' + exname + 'LLsurf_sample.pdf', format='pdf')

plt.figure()
I = -np.squeeze(deepcopy(LLsurf[xx,yy,yi,xi,:]))
# plt.plot(rotate, I)
plt.plot(rotate, m)
plt.xlabel('Rotation')
plt.ylabel('Max Log-Likelihood')
plt.savefig(figdir + '/' + exname + 'maxLLsurf_rotation.pdf', format='pdf')
#%% measure max shift / rotation over all positions

def LLinterpolate(LLsurf, xx, yy):
    sz = LLsurf.shape
    a=np.squeeze(-deepcopy(LLsurf[xx,yy,:,:,:]))
    if xx < sz[0]-1:
        a+=np.squeeze(-deepcopy(LLsurf[xx+1,yy,:,:,:]))
    if xx > 0:
        a+=np.squeeze(-deepcopy(LLsurf[xx-1,yy,:,:,:]))
    if yy > 0:
        a+=np.squeeze(-deepcopy(LLsurf[xx,yy-1,:,:,:]))
    if yy < sz[1]-1:
        a+=np.squeeze(-deepcopy(LLsurf[xx,yy+1,:,:,:]))

    return a

cx = NX//2
cy = NY//2
Im = DU.gabor_sized(cx,90)
Im = np.zeros((NY, NX))
Im[:,cx-2:cx+2]=1.0
Im[cy-2:cy+2,:]=1.0
plt.imshow(Im)

X = Im.flatten() # flattened image




dims = np.asarray((NX, NY))
outsize=deepcopy(dims)
plt.figure(figsize=(10,10))
for xx in range(Npos):
    for yy in range(Npos):
        plt.subplot(Npos, Npos, (Npos-1-yy)*Npos + xx + 1)
        a = LLinterpolate(LLsurf, xx, yy)
        # a=np.squeeze(-deepcopy(LLsurf[xx,yy,:,:,:]))
        indx = np.unravel_index(a.argmax(), a.shape)
        ty = indx[0]-cy
        tx = indx[1]-cx
        th = rotate[indx[2]]

        ix,w = ne.get_crop_indices(dims, outsize=outsize, translation=np.array((tx,ty)), theta=th, scale=1.0)

        I2 = 0
        for i in range(len(w)):
            I2 = I2 + w[i]*X[ix[i]]
        
        plt.imshow(np.reshape(I2, (NY, NX)), cmap=plt.cm.gray_r)
        plt.axvline(cx, color='r', linestyle='--')
        plt.axhline(cy, color='r', linestyle='--')
        plt.axis("off")

plt.savefig(figdir + '/' + exname + 'corrMat.pdf', format='pdf')
#%% correct eye position

eyeAtFrameCentered = (eyeAtFrame-(640, 380))
eyeX = deepcopy(eyeAtFrameCentered[:,0])
eyeY = deepcopy(eyeAtFrameCentered[:,1])

StimC = deepcopy(Stim)
for xx in range(Npos):
    for yy in range(Npos):
        # get index for when the eye position was withing the boundaries
        rs = np.hypot(eyeX-locs[xx], eyeY-locs[yy])
            
        # eccentricity dependent 
        ecc = np.hypot(locs[xx],locs[yy])
        Ethresh = Erad + .2*ecc # eccentricity dependent threshold
            
        valE = np.where(rs < Ethresh)[0]
        valtot = valE
        
        if len(valtot)<1000:
            LLsurf[xx,yy,:,:,:] = 0

        a = LLinterpolate(LLsurf, xx, yy)
        # a=np.squeeze(-deepcopy(LLsurf[xx,yy,:,:,:]))
        indx = np.unravel_index(a.argmax(), a.shape)
        ty = indx[0]-cy
        tx = indx[1]-cx
        if np.abs(tx)>4:
            tx = 0
        if np.abs(ty)>4:
            ty = 0

        th = rotate[indx[2]]
        # print(th)

        ix,w = ne.get_crop_indices(dims, outsize=outsize, translation=np.array((tx,ty)), theta=th, scale=1.0)
        
        s0 = deepcopy(Stim[valtot,:])
        stmp = 0
        for i in range(len(w)):
            stmp += s0[:,ix[i]]*w[i]
        StimC[valtot,:] = deepcopy(stmp)

#%%

valdata = np.intersect1d(np.where(valdat[:,0] == 1)[0], np.where(labels[:,0] == 1)[0]) # fixations / valid

valid_eye_rad = 4.5  # degrees -- use this when looking at eye-calibration (see below)
ppd = 37.50476617061

eyeX = (eyeAtFrame[:,0]-640)/ppd
eyeY = (eyeAtFrame[:,1]-380)/ppd

eyeCentered = np.hypot(eyeX, eyeY) < valid_eye_rad
# eyeCentered = np.logical_and(eyeX < 0, eyeCentered)
valdata = np.intersect1d(valdata, np.where(eyeCentered)[0])

valid = valdata
# check new STAS
stas0 = ne.get_stas(Stim, Robs, [NX,NY], valid=valid, num_lags=14, plot=False)
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


#%%
cc = 0

#%%
cc+=1
if cc > NC-1:
    cc = 0
plt.figure(figsize=(20,3))
nlags = stas.shape[1]
sta_ = stas[:,:,cc]
mn = np.min(sta_.flatten())
mx = np.max(sta_.flatten())

for ilag in range(nlags):
    plt.subplot(2,nlags,ilag+1)
    plt.imshow(np.reshape(stas[:,ilag,cc], (NY, NX)),vmin=mn, vmax=mx)
    plt.axis("off")
    plt.subplot(2,nlags,nlags+ilag+1)
    plt.imshow(np.reshape(stas0[:,ilag,cc], (NY, NX)), vmin=mn, vmax=mx)
    plt.axis("off")

#%%plt.figure(figsize=(15,10))
sx,sy = U.get_subplot_dims(NC)
for cc in range(NC):
    plt.subplot(sx, sy, cc+1)
    plt.plot(np.std(stas0[:,:,cc], axis=0), 'k')
    plt.plot(np.std(stas[:,:,cc], axis=0), 'r')





# translation=np.array((0,0))
# scale = 1.0
# theta=15

# ix,w = ne.get_crop_indices(dims, outsize=outsize, translation=translation, theta=theta, scale=scale)

# I2 = 0
# for i in range(len(w)):
#     I2 = I2 + w[i]*X[ix[i]]
#%% Run eye correction
# Run it all once
eyeAtFrameCentered = (eyeAtFrame-(640, 380))
centers5, locs, LLspace1 = ne.get_corr_grid(side2b, Stim, Robs, [NX,NY], cids,
        eyeAtFrameCentered, valdata, valid_eye_rad=5.2*ppd, Erad=0.5*ppd, Npos=25,
        crop_edge=3, plot=True, interpolation_steps=2, softmax=1000, autoencoder=False)

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
    plt.imshow(I, aspect='equal')
    plt.axvline(Ly/2, color='w')
    plt.axhline(Lx/2, color='w')
    plt.plot(x,y, '+r')
    plt.title('(%d,%d)' %(16,i))

#%% Get LLsurface centers
centers5,LLspace3 = ne.get_centers_from_LLspace(LLspace1, softmax=10, plot=True, interpolation_steps=6, crop_edge=3)

#%% Correct stimulus
centers6 = deepcopy(centers5)
ctrid = np.argmin(locs**2)
centers6[:,:,0]-=centers6[ctrid,ctrid,0]
centers6[:,:,1]-=centers6[ctrid,ctrid,1]

xcorrec,ycorrec = ne.get_shifter_from_centers(centers6, locs, maxshift=8)

eyeAtFrameCentered = (eyeAtFrame-(640, 380))

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

        # 

# %% try to do better: get a new stimulus model
side2b, Xstim, Rvalid, rinds, cids, Cxinds = ne.get_stim_model(StimC, Robs, [NX,NY], valid=valdata, num_lags=10, plot=True, Cindx=Cxinds, autoencoder=True)
DU.plot_3dfilters(side2b)
# %% Try again with less regularization (starting with previous best model)
side2c, Xstim, Rvalid, rinds, cids, Cxinds = ne.get_stim_model(StimC, Robs, [NX,NY], valid=valdata, num_lags=10, plot=True,
            XTreg=0.005,L1reg=5e-3,MIreg=0.1,MSCreg=0.1,Greg=0.1,Mreg=1e-4,Cindx=Cxinds, base_mod=side2b, cids=cids)

#%%
DU.plot_3dfilters(side2c)
#%% if side2c is better
side2b = side2c.copy_model()
#%% compute likelihood surface
eyeAtFrameCentered = (eyeAtFrame-(640, 380))
centers5, locs, LLspace1 = ne.get_corr_grid(side2b, Stim, Robs, [NX,NY], cids,
        eyeAtFrameCentered, valdata, valid_eye_rad=5.2*ppd, Erad=0.5*ppd, Npos=25,
        crop_edge=3, plot=True, interpolation_steps=2, softmax=1000)

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

base_mod = side2b.copy_model()

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
cell_shift_mod.networks[core_net].layers[0].weights = deepcopy(base_mod.networks[core_net].layers[0].weights)
cell_shift_mod.networks[core_net].layers[0].biases = deepcopy(base_mod.networks[core_net].layers[0].biases)
cell_shift_mod.networks[core_net].layers[1].weights = deepcopy(base_mod.networks[core_net].layers[1].weights)
cell_shift_mod.networks[core_net].layers[1].biases = deepcopy(base_mod.networks[core_net].layers[1].biases)
cell_shift_mod.networks[side_net].layers[0].weights = deepcopy(base_mod.networks[side_net].layers[0].weights)
cell_shift_mod.networks[side_net].layers[0].biases = deepcopy(base_mod.networks[side_net].layers[0].biases)

num_space = np.prod(cell_shift_mod.input_sizes[scaff_net][:-1])

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
