# %% Import libraries
sys.path.insert(0, '/home/jake/Data/Repos/')
# import deepdish as dd
import Utils as U
import gratings as gt


import NDN3.NDNutils as NDNutils


import numpy as np
# import tensorflow as tf
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt  # plotting
import seaborn as sns

# import NDN3.NDN as NDN
# import NDN3.Utils.DanUtils as DU

# %% list sessions
sesslist = gt.list_sessions()
sesslist = list(sesslist)
for i in range(len(sesslist)):
    print("%d %s" %(i, sesslist[i]))

# %% Load one session
indexlist = [17]
stim, _, _, Robs, _, basis, opts, sacbc, valid, eyepos = gt.load_and_setup(indexlist)

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
# XsacOn = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacon,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
# XsacOff = NDNutils.create_time_embedding( NDNutils.shift_mat_zpad(sacoff,-back_shifts,dim=0), [num_saclags, 1, 1], tent_spacing=1)
# XsacOnCausal = NDNutils.create_time_embedding( sacon, [num_saclags, 1, 1], tent_spacing=1)
# XsacOffCausal = NDNutils.create_time_embedding( sacoff, [num_saclags, 1, 1], tent_spacing=1)
Robs = Robs.astype('float32')
Xstim = Xstim.astype('float32')

#%% optimization params
Ui = opts['Ui']
Xi = opts['Xi']
Ti = opts['Ti']

#%%
class MyModel(torch.nn.Module):
    def __init__(self,D_in,D_out):
        super(MyModel, self).__init__()
        self.layer = torch.nn.Linear(D_in,D_out)
        self.spkNL = torch.nn.Softplus()
    def forward(self,x):
        out = self.spkNL(self.layer(x))
        return out

#%% setup model
from torch.utils.data import TensorDataset,DataLoader

bs = 500
device = torch.device("cuda:0") 

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (Xstim[Ui,:], Robs[Ui,:], Xstim[Ti,:], Robs[Ti,:])
)

x, y = x_train.to(device), y_train.to(device)
xt,yt = x_valid.to(device), y_valid.to(device)

# setup datasets
train_ds = TensorDataset(x, y)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(xt, yt)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

#%%

model = MyModel(NX*NY*num_lags, NC)
# model = torch.nn.Sequential(
#     torch.nn.Linear(NX*NY*num_lags, NC),
#     torch.nn.Softplus()
# )

model.to(device)

# loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn = torch.nn.PoissonNLLLoss()
#%%
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        if epoch % 10 ==0:
            print(epoch, val_loss)

#%%
learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1, amsgrad=True)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
fit(500, model, loss_fn,optimizer, train_dl, valid_dl)

#%%

ypred = model(xt)
cc = 1
a = ypred[:,cc].detach().cpu().numpy()
r0 = np.reshape(a, (opts['num_repeats'],-1))
r = np.reshape(Robs[Ti,cc], (opts['num_repeats'],-1))

r = np.average(r, axis=0)
r0 = np.average(r0, axis=0)
plt.plot(r)
plt.plot(r0)
plt.title("cell %d" %cc)
U.r_squared(np.reshape(r, (-1,1)), np.reshape(r0, (-1,1)))


# %%
w = model.layer.weight.detach().cpu().numpy()

# %%
sx,sy = U.get_subplot_dims(NC)
plt.figure(figsize=(10,10))
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    plt.imshow(np.reshape(w[cc,:], (NX*NY, num_lags)), aspect='auto')

# %%
w.shape

# %%
