# %% Import libraries
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

# import deepdish as dd
import V1FreeViewingCode.Analysis.notebooks.Utils as U
import V1FreeViewingCode.Analysis.notebooks.gratings as gt

# import NDN3.NDNutils as NDNutils

import numpy as np
# import tensorflow as tf
import torch
# import torch.nn.functional as F

import matplotlib.pyplot as plt  # plotting
import seaborn as sns


#%% load data

from V1FreeViewingCode.Analysis.notebooks.Datasets import GratingDataset
from torch.utils.data import Dataset, DataLoader, random_split

sessid = 18
gd = GratingDataset(sessid, num_lags=10, augment=1)

# test set
gd_test = GratingDataset(sessid, num_lags=10,train=False)
# #%%
n_val = np.floor(len(gd)/5).astype(int)
n_train = (len(gd)-n_val).astype(int)

gd_train, gd_val = random_split(gd, lengths=[n_train, n_val])

bs = 10000
train_dl = DataLoader(gd_train, batch_size=bs)
valid_dl = DataLoader(gd_val, batch_size=bs)


#%%
from V1FreeViewingCode.models.basic import LNP
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


D_in = gd.NX*gd.NY*gd.num_lags
last_model = LNP.load_from_checkpoint(checkpoint_path="./checkpoints/no_augment.ckpt")
model = LNP(input_dim=D_in, output_dim=gd.NC, learning_rate=1e-1,
    optimizer='AdamW')


#%%

dataloader = DataLoader(gd)
dataiter = iter(dataloader)
data = dataiter.next()
X,Y = data

model(X).shape
#%%    

early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.001)
checkpoint_callback = ModelCheckpoint(monitor='val_loss')

seed_everything(42)


trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
    checkpoint_callback=checkpoint_callback,
    deterministic=False,
    default_root_dir='./checkpoints',
    progress_bar_refresh_rate=20,
    auto_lr_find=True)


trainer.tune(model, train_dl, valid_dl) # find learning rate

trainer.fit(model, train_dl, valid_dl)

#%%

trainer.save_checkpoint("./checkpoints/augment.ckpt")


#%%
cc = 0
m1 = last_model.cpu()
m2 = model.cpu()
xt,yt=gd_test[:]


loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')

l2 = loss(m2(xt),yt).detach().cpu().numpy().sum(axis=0)
l1 = loss(m1(xt),yt).detach().cpu().numpy().sum(axis=0)

l2/=np.sum(yt.numpy(), axis=0)
l1/=np.sum(yt.numpy(), axis=0)

plt.plot(l1, l2, '.')
plt.plot(plt.xlim(), plt.xlim())
# plt.hist(np.sum(xtest, axis=0)/np.sum(yt.numpy(), axis=0))
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

nfilt = w.shape[0]
sx,sy = U.get_subplot_dims(nfilt)
plt.figure(figsize=(10,10))
for cc in range(nfilt):
    plt.subplot(sx,sy,cc+1)
    wtmp = np.reshape(w[cc,:], (gd.num_lags, gd.NX*gd.NY))
    # plt.imshow(np.reshape(w[cc,:], (gd.NX*gd.NY, gd.num_lags)), aspect='auto')
    plt.imshow(wtmp, aspect='auto')
    # plt.plot(wtmp)

# %%
w.shape

# %%
