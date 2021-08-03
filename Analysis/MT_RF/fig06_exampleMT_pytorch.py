#%% set paths
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import V1FreeViewingCode.Analysis.notebooks.gratings as gt

# from scipy.ndimage import gaussian_filter
from copy import deepcopy

import numpy as np

import matplotlib.pyplot as plt  # plotting

import torch
import torch.nn as nn

from V1FreeViewingCode.models.trainers import Trainer, EarlyStopping
from V1FreeViewingCode.models.datasets import generic_recording 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %% load example sessions

datadir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/MT_RF/'
# fname = 'Ellie_190120_0_0_30_30_1.mat'
fname = 'Ellie_190120_0_0_30_30_2.mat'

matdat = gt.loadmat(datadir+fname)

Robs = deepcopy(matdat['MoStimY']).T
X = deepcopy(matdat['MoStimX']).T
frameTime = X[:,0]

# stim is NT x (NX*NY). Any non-zero value is the drift direction (as an integer) of a dot (at that spatial location)
Stim = X[:,3:]

#%% convert direciton stimulus to dx and dy

# convert drift direction to degrees
dbin = 360/np.max(Stim)

# output will be x,y vectors
dx = np.zeros(Stim.shape, dtype='float32')
dy = np.zeros(Stim.shape, dtype='float32')

stimind = np.where(Stim!=0) # non-zero values of Stim

dx[stimind[0], stimind[1]]=np.cos(Stim[stimind[0], stimind[1]]*dbin/180*np.pi)
dy[stimind[0], stimind[1]]=np.sin(Stim[stimind[0], stimind[1]]*dbin/180*np.pi)

xax = np.arange(matdat['GRID']['box'][0], matdat['GRID']['box'][2], matdat['GRID']['div'][0])
yax = np.arange(matdat['GRID']['box'][1], matdat['GRID']['box'][3], matdat['GRID']['div'][0])

xx = np.meshgrid(xax, yax)

# plot single frame
plt.figure()
vframes = np.where(np.sum(dx**2 + dy**2, axis=1)==6)[0]
iframe = vframes[0]
plt.quiver(xx[0].flatten(), xx[1].flatten(), dx[iframe,:], dy[iframe,:], scale=10)
plt.title(iframe)

# concatenate dx/dy into one velocity stimulus
NT = Stim.shape[0]
NC = Robs.shape[1]
NX = len(xax)
NY = len(yax)

vel = np.concatenate((dx, dy), axis=1)
# weird python reshaping
v_reshape = np.reshape(vel,[NT, 2, NX*NY])
vel = np.transpose(v_reshape, (0,2,1)).reshape((NT, NX*NY*2))


#%% Build dataset
from V1FreeViewingCode.models.utils import create_time_embedding, generate_xv_folds
print("Creating Time Embedding for stimulus shape (%d,%d)" %(NT,NX*NY))
num_lags = 18
Xstim = create_time_embedding( vel, [num_lags, NX*2, NY], tent_spacing=1 )
Ui, Xi = generate_xv_folds(NT)

cids = [13, 16, 18, 20, 21, 22, 25, 27, 28, 29, 32, 33, 36, 47,61,64,65,66,68,78,79,80]
NC = len(cids)
R = Robs[:,cids]
train_ds = generic_recording(Xstim[Ui,:], R[Ui,:], device=device)
test_ds = generic_recording(Xstim[Ui,:], R[Ui,:], device=device)

#%%
print("Computing STA")
Y = Robs[Ui,:] - np.mean(Robs[Ui,:], axis=0)
sta = Xstim[Ui,:].T @ Y

print("Done")

#%%
from scipy.stats import median_absolute_deviation

sx = np.ceil(np.sqrt(NC)).astype(int)
sy = np.round(np.sqrt(NC)).astype(int)
plt.figure(figsize=(10,10))

NC = Robs.shape[1] # plot all neurons
mthresh = np.zeros(NC)
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)

    I = np.reshape(sta[:,cc], [NX*2*NY, num_lags])

    m = median_absolute_deviation(I.flatten())
    mthresh[cc] = np.sum(I.flatten() > m*4)

    tpower = np.std(I,axis=0)

    peak_lag = np.argmax(tpower)

    I = np.reshape(sta[:,cc], [NY, NX, 2, num_lags])
    dx = I[:,:,0,peak_lag]
    dy = I[:,:,1,peak_lag]

    plt.quiver(xx[0], xx[1], dx, dy, np.sqrt(dx**2 + dy**2), cmap=plt.cm.gray_r,
        pivot='tail',units='width',
        scale=250, headwidth=10, headlength=10)
    plt.axis("off")
    plt.title(cc)



#%% Select subset of units
cids = [13, 16, 18, 20, 21, 22, 25, 27, 28, 29, 32, 33, 36, 47,61,64,65,66,68,78,79,80]
NC = len(cids)
sx = np.ceil(np.sqrt(NC)).astype(int)
sy = np.round(np.sqrt(NC)).astype(int)
plt.figure(figsize=(10,10))

for cc in range(NC):
    plt.subplot(sx,sy,cc+1)

    I = np.reshape(sta[:,cids[cc]], [NX*2*NY, num_lags])

    tpower = np.std(I,axis=0)
    peak_lag = np.argmax(tpower)

    I = np.reshape(sta[:,cids[cc]], [NY, NX, 2, num_lags])
    dx = I[:,:,0,peak_lag]
    dy = I[:,:,1,peak_lag]
    
    plt.quiver(xx[0]-15, xx[1]-15, dx, dy, np.sqrt(dx**2 + dy**2), cmap=plt.cm.gray_r,
        pivot='tail',units='width',
        scale=500, headwidth=10, headlength=10)
    plt.axhline(0, color='r')
    plt.axvline(0, color='r')
    plt.axis("off")
    plt.title(cids[cc])


#%%

cc +=1
if cc >= NC:
    cc = 1


plt.figure(figsize=(10,2))

I = np.reshape(sta[:,cids[cc]], (NY, NX, 2, num_lags))
for lag in range(num_lags):
    plt.subplot(1,num_lags,lag+1)
    
    dx = I[:,:,0,lag]
    dy = I[:,:,1,lag]

    plt.quiver(xx[0].flatten()-15, xx[1].flatten()-15, dx, dy, np.sqrt(dx**2 + dy**2),
        cmap=plt.cm.jet, units='inches',
        pivot='tail',
        scale=100, headwidth=50, headlength=50,linewidth=100)

    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.axis("off")
    
plt.title(cids[cc])        

#%%
plt.figure(figsize=(10,10))
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)

    I = np.reshape(sta[:,cids[cc]], [NX*2*NY, num_lags])

    tpower = np.std(I,axis=0)
    peak_lag = np.argmax(tpower)
    
    print(peak_lag)
    I = np.reshape(sta[:,cids[cc]], [NY,NX,2, num_lags])

    dx = I[:,:,0,peak_lag]
    dy = I[:,:,1,peak_lag]
    I = np.concatenate( (dx, dy), axis=1)
    
    plt.imshow(I)
    plt.axvline(15/2, color='r')
    plt.axvline(45/2, color='r')
    plt.axhline(15/2, color='r')

    plt.axis("off")
    plt.title(cids[cc])


#%% GLM time


def ndGraphLap(nn):
    '''
        Graph Laplacian of a ND array to regularize smoothness
    '''
    if type(nn) is not list:
        nn = [nn]

    from scipy.sparse import spdiags, kron

    ndim = len(nn)
    L = []
    for j in range(ndim):
        if nn[j] <= 2:
            L.append(spdiags([np.ones(nn[j])], [0], nn[j], nn[j]))
        else:
            ectr = np.concatenate( [np.ones(1), 2*np.ones(nn[j]-2), np.ones(1)])
            eoff = -np.ones(nn[j])
            L.append(spdiags([eoff, ectr, eoff], [-1, 0, 1], nn[j], nn[j]))

    M = 0
    for j in range(ndim):
        
        nbefore = np.prod(nn[:j]).astype(int)
        nafter = np.prod(nn[j+1:]).astype(int)

        Mtmp = kron(L[j], np.eye(nbefore))
        Mtmp = kron(np.eye(nafter), Mtmp)
        M += Mtmp
    
    return M.toarray()

# specify models
class Poisson(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss = nn.PoissonNLLLoss(log_input=False)

    def regularlizer(self):
        return 0.0

    def training_step(self, batch):
        x = batch['stim']
        y = batch['robs']
        yhat = self(x)
        loss = self.loss(yhat, y)
        regpen = self.regularlizer()
        return {'loss': loss + regpen}

    def validation_step(self, batch):
        x = batch['stim']
        y = batch['robs']
        yhat = self(x)
        loss = self.loss(yhat, y)
        return {'loss': loss, 'val_loss': loss}

class GLM(Poisson):

    def __init__(self, dims, num_output=10, reg_amount = 0.01):
        super().__init__()

        # Set our init args as class attributes
        self.num_output = num_output

        self.dims = dims

        self.register_buffer("reg_mat", torch.tensor(ndGraphLap(dims), dtype=torch.float32))
        self.reg_amount = reg_amount

        # Define PyTorch model
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.dims), self.num_output),
            nn.Softplus()
        )
        
    def regularlizer(self):
        
        pen = self.net[1].weight@self.reg_mat@self.net[1].weight.T
        pen = pen.trace() + torch.sqrt(torch.sum(model.net[1].weight**2))
        return self.reg_amount * pen

    def forward(self, x):
        x = self.net(x)
        return x

class simpleNIM(Poisson):

    def __init__(self, dims, hidden_size=64, num_output=10):

        super().__init__()

        # Set our init args as class attributes
        self.hidden_size = hidden_size
        self.num_output = num_output

        self.dims = dims

        # Define PyTorch model
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.dims), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_output),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.net(x)

        return x


#%% make dataset

model = GLM([NY, NX, 2, num_lags], num_output=NC, reg_amount = .1)
model.to(device)


#%%
sample = train_ds[:10]
yhat = model.forward(sample['stim'])
print(yhat.shape, yhat.dtype)

#%%
import os
from torch.utils.data import DataLoader

# optimizer = torch.optim.AdamW(model.parameters(),
#                 lr=0.001,
#                 betas=[0.9, 0.999],
#                 weight_decay=10,
#                 amsgrad=True)

# optimizer = torch.optim.Adam(model.parameters(),
#                 lr=0.001,
#                 betas=[0.9, 0.999])

from V1FreeViewingCode.models.LBFGS import LBFGS, FullBatchLBFGS
optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search='Wolfe', debug=False)
# optimizer = FullBatchLBFGS(model.parameters(), lr=1., history_size=10, line_search='Wolfe', debug=True)
# # from torch.optim import LBFGS
# from NDN.LBFGSnew import LBFGS
# optimizer = LBFGS(model.parameters(), history_size=10, max_iter=4,line_search_fn=True,batch_mode=False)

batch_size = 8192
# batch_size = 1000
# from NDN.trainers import Trainer
traindl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
testdl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

earlystopping = EarlyStopping(patience=7,delta=0)

trainer = Trainer(model, optimizer, early_stopping=earlystopping)

#%% fit
num_epochs = 100
trainer.fit(num_epochs, traindl, testdl)

#%% plot weights
w = trainer.model.net[1].weight.data.cpu().numpy()
w1 = model.net[1].weight.data.cpu().numpy()

from scipy.stats import median_absolute_deviation
sta = w.T
NC = w.shape[0]
sx = np.ceil(np.sqrt(NC)).astype(int)
sy = np.round(np.sqrt(NC)).astype(int)
plt.figure(figsize=(10,10))

# NC = Robs.shape[1] # plot all neurons
mthresh = np.zeros(NC)
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)

    I = np.reshape(sta[:,cc], [NX*2*NY, num_lags])

    m = median_absolute_deviation(I.flatten())
    mthresh[cc] = np.sum(I.flatten() > m*4)

    tpower = np.std(I,axis=0)

    peak_lag = np.argmax(tpower)

    I = np.reshape(sta[:,cc], [NY, NX, 2, num_lags])
    dx = I[:,:,0,peak_lag]
    dy = I[:,:,1,peak_lag]

    plt.imshow(dx)
    # plt.quiver(xx[0], xx[1], dx, dy, np.sqrt(dx**2 + dy**2), cmap=plt.cm.gray_r,
    #     pivot='tail',units='width',
    #     scale=1, headwidth=10, headlength=10)
    plt.axis("off")
    plt.title(cc)


#%% Find best regularization

glmbest = glm0.copy_model()

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


#%% plot RFs

RF = []
plt.rcParams.update({'font.size': 7})

# for cc in range(NC):
#     plt.subplot(sx,sy,cc+1)
cc += 1
figname = '/home/jake/Data/Datasets/MitchellV1FreeViewing/FVmanuscript/' + 'Fig06_MTglm_' + str(cc) + '.pdf'
if cc >=NC:
    cc = 0

print("Get RF / Tuning Curve for unit %d" %cc)

if LLx0[cc] > LLx[cc]:
    wtsFull = deepcopy(glm0.networks[0].layers[0].weights[:,cc])
else:
    wtsFull = deepcopy(glmbest.networks[0].layers[0].weights[:,cc])


wts = np.reshape(wtsFull, (-1, num_lags))

tpower = np.std(wts, axis=0)
peak_lag = np.argmax(tpower)

I = np.reshape(wts[:,peak_lag], (NX,NY,2))

dx = I[:,:,0]
dy = I[:,:,1]

plt.figure(figsize=(4.48,2))
plt.tight_layout()

ax = plt.subplot(1,3,1)
amp = np.hypot(dx, dy)
peak_space = np.argmax(amp)
min_space = np.argmin(amp)
plt.quiver(xx[0]-np.mean(xx[0]), xx[1]-np.mean(xx[1]), dx/np.max(amp), dy/np.max(amp), # np.arctan2(dx, dy)/np.pi*180, cmap=plt.cm.v  ,
        pivot='tail',units='width', width=.008,
        scale=15, headwidth=5, headlength=5)


plt.axhline(0, color='gray', )
plt.axvline(0, color='gray')

plt.xlabel('Azimuth (d.v.a.)')
plt.ylabel('Elevation (d.v.a)')

plt.xticks(np.arange(-15,18,5))

plt.subplot(1,3,2)
w = np.reshape(wts, (NY*NX, 2, num_lags))

amp /= np.sum(amp)

muw = np.array( (dx.flatten() @ amp.flatten(), dy.flatten() @ amp.flatten()))
muw /= np.hypot(muw[0], muw[1])

tpeak = w[peak_space,0,:]*muw[0] + w[peak_space,1,:]*muw[1]
tmin = w[min_space,0,:]*muw[0] + w[min_space,1,:]*muw[1]
lags = np.arange(0, num_lags, 1)*1000/120
plt.plot(lags, tpeak, '-o', color='b', ms=3)
plt.plot(lags, tmin, '-o', color='r', ms=3)
plt.xlabel('Lags (ms)')
plt.ylabel('Power (along preferred direction)')

plt.axhline(0, color='gray')

plt.xticks(np.arange(0,200,50))
# get tuning

mask = ((amp/np.max(amp)) > .5).flatten()

sfilt = Stim * mask

inds = np.where(sfilt!=0)
ds = sfilt[inds[0],inds[1]]
dirs = np.unique(ds)

dstim = np.zeros( (NT, len(dirs)))
dstim[inds[0], (ds-1).astype(int)] = 1.0

dXstim = create_time_embedding(dstim, [num_lags, len(dirs)])

dsta = (dXstim.T@R[:,cc]) / np.sum(dXstim, axis=0) * 100

I = np.reshape(dsta, (-1, num_lags))

dirs = dirs * dbin
tpower = np.std(I,axis=0)
peak_lag = np.argmax(tpower)

# bootstrap error bars

# don't sum in STA somputation (all samples preserved)
dsta = (dXstim * np.expand_dims(R[:,cc], axis=1)) / np.sum(dXstim, axis=0) * 100

# resample and compute confidence intervals (memory inefficient)
nboot = 100
bootinds = np.random.randint(0, high=NT, size=(NT, nboot))
staboot = np.sum(dsta[bootinds,:], axis=0)
dboot = np.reshape(staboot, (nboot, len(dirs), num_lags))[:,:,peak_lag]

ci = np.percentile(dboot, (2.5, 97.5), axis=0)

# fit von mises
import scipy.optimize as opt

def von_mises(theta, thetaPref, Bandwidth, base, amplitude):

    y = base + amplitude * np.exp( Bandwidth * (np.cos(theta - thetaPref) - 1))
    return y

tuning_curve = I[:,peak_lag]



theta = np.linspace(0, 2*np.pi, 100)

w = tuning_curve / np.sum(tuning_curve)
th = dirs/180*np.pi
mu0 = np.arctan2(np.sin(th)@w, np.cos(th)@w)
bw0 = 1
initial_guess = (mu0, bw0, np.min(tuning_curve), np.max(tuning_curve)-np.min(tuning_curve))
popt, pcov = opt.curve_fit(von_mises, dirs/180*np.pi, tuning_curve, p0 = initial_guess)

plt.subplot(1,3,3)
plt.errorbar(dirs, tuning_curve, np.abs(ci-I[:,peak_lag]), marker='o', linestyle='none', markersize=3)
plt.plot(theta/np.pi*180, von_mises(theta, popt[0], popt[1], popt[2], popt[3]))
plt.xlabel('Direction')
plt.ylabel('Firing Rate (sp/s)')

plt.xticks(np.arange(0,365,90))
sns.despine(trim=True, offset=0)

plt.gcf().subplots_adjust(bottom=0.15)

plt.savefig(figname)

# store fit / 
RF_ = {'wts': wtsFull,
    'shape': (NY, NX, 2, num_lags), 
    'peak_lag': peak_lag,
    'dxsrf': dx,
    'dysrf': dy,
    'timelags': lags,
    'tpeak': tpeak,
    'tmin': tmin,
    'tuningCurveParams': ['Mu', 'Kappa', 'Base', 'Amplitude'],
    'popt': popt,
    'pcov': pcov}

RF.append(RF_)

#%% plots

figname = '/home/jake/Data/Datasets/MitchellV1FreeViewing/FVmanuscript/' + 'Fig04_glm_' + matdat['exname'] + '_' + str(cid) + '.mat'
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


#%% FUTURE: Try NIM

d2t = deepcopy(glmbest.networks[0].layers[0].reg.vals['d2t'])
d2x = deepcopy(glmbest.networks[0].layers[0].reg.vals['d2x'])
Greg = deepcopy(glmbest.networks[0].layers[0].reg.vals['glocal'])

par = NDNutils.ffnetwork_params( 
    input_dims=[2,NX,NY,num_lags],
    layer_sizes=[2, 1],
    ei_layers=[1],
    layer_types=['normal', 'normal'], normalization=[1, 0],
    act_funcs=['relu', 'softplus'], verbose=True,
    reg_list={'d2t': [d2t], 'd2x':[d2x], 'glocal':[Greg]})

# initialize GLM
nim0 = NDN.NDN([par],  noise_dist='poisson')

nim0.networks[0].layers[0].weights[:,0] = deepcopy(glmbest.networks[0].layers[0].weights[:,cc])
nim0.networks[0].layers[0].weights[:,1] = -deepcopy(glmbest.networks[0].layers[0].weights[:,cc])

v2f0 = nim0.fit_variables(fit_biases=True)

R = Robs[:,cids].astype('float32')
R = R[:,cc]

_ = nim0.train(input_data=[Xstim], output_data=R,
    train_indxs=Ui, test_indxs=Xi,
    learning_alg='adam', opt_params=adam_params,
    fit_variables=v2f0)     

# plot filters
DU.plot_3dfilters(nim0)


#%%

wts = deepcopy(nim0.networks[0].layers[0].weights[:,1])
wts = np.reshape(wts, (-1, num_lags))

tpower = np.std(wts, axis=0)
peak_lag = np.argmax(tpower)

I = np.reshape(wts[:,peak_lag], (NX,NY,2))

dx = I[:,:,0]
dy = I[:,:,1]

plt.figure(figsize=(10,5))
ax = plt.subplot(1,2,1)
amp = np.hypot(dx, dy)
peak_space = np.argmax(amp)
min_space = np.argmin(amp)
plt.quiver(xx[0]-np.mean(xx[0]), xx[1]-np.mean(xx[1]), dx, dy, # np.arctan2(dx, dy)/np.pi*180, cmap=plt.cm.v  ,
        pivot='tail',units='width',
        scale=1, headwidth=5, headlength=5)

plt.axhline(0, color='gray')
plt.axvline(0, color='gray')

plt.xlabel('Azimuth (d.v.a.)')
plt.ylabel('Elevation (d.v.a)')

plt.subplot(1,2,2)
w = np.reshape(wts, (NY*NX, 2, num_lags))

amp /= np.sum(amp)

muw = np.array( (dx.flatten() @ amp.flatten(), dy.flatten() @ amp.flatten()))
muw /= np.hypot(muw[0], muw[1])

tpeak = w[peak_space,0,:]*muw[0] + w[peak_space,1,:]*muw[1]
tmin = w[min_space,0,:]*muw[0] + w[min_space,1,:]*muw[1]
lags = np.arange(0, num_lags, 1)*1000/120
plt.plot(lags, tpeak, '-o', color='b')
plt.plot(lags, tmin, '-o', color='r')
plt.xlabel('Lags (ms)')
plt.ylabel('Power (along preferred direction)')

plt.axhline(0, color='gray')

# plt.figure()
# plt.imshow(np.flipud(amp))
# plt.plot(np.hypot(w[min_space,0,:], w[min_space,1,:]), color='r')

# plt.plot(np.hypot(w[peak_space,0,:], w[peak_space,1,:]), color='b')
# plt.plot(np.hypot(w[min_space,0,:], w[min_space,1,:]), color='r')
