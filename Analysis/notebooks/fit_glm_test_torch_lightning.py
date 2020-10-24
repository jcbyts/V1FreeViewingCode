# %% Import libraries
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

# import deepdish as dd
import Utils as U
import gratings as gt

# import NDN3.NDNutils as NDNutils

import numpy as np
# import tensorflow as tf
import torch
# import torch.nn.functional as F

import matplotlib.pyplot as plt  # plotting
import seaborn as sns


# %% list sessions
# sesslist = gt.list_sessions()
# sesslist = list(sesslist)
# for i in range(len(sesslist)):
#     print("%d %s" %(i, sesslist[i]))

# %% Load one session
# indexlist = [17]
# stim, _, _, Robs, _, basis, opts, sacbc, valid, eyepos = gt.load_and_setup(indexlist)

#%% load data

from V1FreeViewingCode.Analysis.notebooks.Datasets import GratingDataset
from torch.utils.data import Dataset, DataLoader, random_split

sessid = 18
gd = GratingDataset(sessid, num_lags=10)

# test set
gd_test = GratingDataset(sessid, num_lags=10,train=False)
# #%%
n_val = np.floor(len(gd)/5).astype(int)
n_train = (len(gd)-n_val).astype(int)

gd_train, gd_val = random_split(gd, lengths=[n_train, n_val])

bs = 1000
train_dl = DataLoader(gd_train, batch_size=bs)
valid_dl = DataLoader(gd_val, batch_size=bs)

#%% debugging


#%%
# print(X.shape)
# # from scipy.io import savemat

# # savemat('testdata.mat', opts)

# #%%
# # model
# from V1FreeViewingCode.models.LNP import LNP
# from pytorch_lightning import Trainer

# D_in = gd.NX*gd.NY*gd.num_lags
# NC = gd.NC
# model = LNP(input_dim=D_in, output_dim=NC)

# trainer = Trainer()

# lags = np.expand_dims(vind, axis=1) - range(gd.num_lags)
# gd.x[lags,:].permute((0,2,1)).reshape((n,-1))
# iix = lags[valid_rows,:]
#     Xstim = deepcopy(Stim[iix,:]).astype('float32')
#     Xstim = np.reshape(np.transpose(Xstim, (0,2,1)), (-1, np.prod(dims)))

#%%
# # %% build time-embedded stimulus
# num_saclags = 60
# back_shifts = 20
# num_lags = 15
# NX,NY = opts['NX'],opts['NY']
# NT,NC=Robs.shape
# # build time-embedded stimulus
# Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, NY], tent_spacing=1 )
# # XsacOn = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacon,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
# # XsacOff = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacoff,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
# # XsacOnCausal = NDNutils.create_time_embedding( sacon, [num_saclags, 1, 1], tent_spacing=1)
# # XsacOffCausal = NDNutils.create_time_embedding( sacoff, [num_saclags, 1, 1], tent_spacing=1)
# Robs = Robs.astype('float32')
# Xstim = Xstim.astype('float32')

# #%% setup data loader (Future: move to class)
# Ui = opts['Ui']
# Xi = opts['Xi']
# Ti = opts['Ti']


# from V1FreeViewingCode.models.LNP import LNP
# from torch.utils.data import TensorDataset, DataLoader
# from pytorch_lightning import Trainer


# bs = 10000

# x_train, y_train, x_valid, y_valid = map(
#     torch.tensor, (Xstim[Ui,:], Robs[Ui,:], Xstim[Ti,:], Robs[Ti,:])
# )

# x, y = x_train, y_train
# xt,yt = x_valid, y_valid

# # setup datasets
# train_ds = TensorDataset(x, y)
# train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

# valid_ds = TensorDataset(xt, yt)
# valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

#%%
from V1FreeViewingCode.models.basic import LNP
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping


D_in = gd.NX*gd.NY*gd.num_lags
model = LNP(input_dim=D_in, output_dim=gd.NC, learning_rate=1e-1,
    optimizer='AdamW')

optimizer = model.configure_optimizers()

#%%

dataloader = DataLoader(gd)
dataiter = iter(dataloader)
data = dataiter.next()
X,Y = data

model(X).shape
#%%    

early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.001)

seed_everything(42)

trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
    deterministic=False, progress_bar_refresh_rate=20,auto_lr_find=True)


trainer.tune(model, train_dl, valid_dl)

trainer.fit(model, train_dl, valid_dl)


#%%
cc = 0
m2 = model.cpu()
xt,yt=gd_test[:]


loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')

xtest = loss(m2(xt),yt).detach().cpu().numpy()

plt.hist(np.sum(xtest, axis=0))
# LNP.forward
#%%

ypred = model(xt)
cc += 1
if cc == gd.NC:
    cc = 0
a = ypred[:,cc].detach().cpu().numpy()
r0 = np.reshape(a, (gd.opts['num_repeats'],-1))
r = np.reshape(yt[:,cc], (gd.opts['num_repeats'],-1))

r = np.average(r, axis=0)
r0 = np.average(r0, axis=0)
plt.plot(r)
plt.plot(r0)
plt.title("cell %d" %cc)
U.r_squared(np.reshape(r, (-1,1)), np.reshape(r0, (-1,1)))


# %%
w = model.l1.weight.detach().cpu().numpy()


# %%
nfilt = w.shape[0]
sx,sy = U.get_subplot_dims(nfilt)
plt.figure(figsize=(10,10))
for cc in range(nfilt):
    plt.subplot(sx,sy,cc+1)
    wtmp = np.reshape(w[cc,:], (gd.num_lags, gd.NX*gd.NY))
    # plt.imshow(np.reshape(w[cc,:], (gd.NX*gd.NY, gd.num_lags)), aspect='auto')
    # plt.imshow(wtmp, aspect='auto')
    plt.plot(wtmp)

# %%
w.shape

# %%
