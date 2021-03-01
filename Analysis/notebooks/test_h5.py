#%% import
import h5py
import numpy as np

#%%

f.close()
#%% open file for writing
f = h5py.File("mytestfile.hdf5", "w")

NT = 1000
x = np.random.randn(NT,1,10,20).astype('float32')
y = np.random.rand(NT,30).astype('float32')
grp = f.create_group("Gabor")

# add some meta data to the group
grp.attrs['SU'] = np.where(np.random.rand(30) > .5)[0]

grptrain = grp.create_group("Train")
grptest = grp.create_group("Test")

dset1 = grptrain.create_dataset("Stim", x.shape, dtype='f')
dset2 = grptrain.create_dataset("Robs", y.shape, dtype='f')
dset1[:,:,:,:] = x
dset2[:,:] = y


f.close()


# %%
f.close()

#%% test reading

fname = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/logan_20200304_-20_-10_50_60_0_19_0_1.hdf5'

f = h5py.File(fname, "r")

# %%
import matplotlib.pyplot as plt
stim = 'Gabor'
stimset = 'Train'
frame = 1

#%%
frame = np.arange(10,20)
if stim in f.keys():
    if stimset in f[stim].keys():
        I = f[stim][stimset]['Stim'][:,:,frame]
        Robs = f[stim][stimset]['Robs'][:,frame]

# plt.imshow(I)        
d = I.shape

I2 = I[:,:,range(1,10,2)]
plt.imshow(I2[:,:,0])

plt.figure()
plt.imshow(Robs)
#%%
grp = f['Gabor']
grp['Robs'][10:20,:]
#%%
f['Gabor'].attrs['SU']
# %%

f.close()