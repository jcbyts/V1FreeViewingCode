import V1FreeViewingCode.models.regularizers as regularizers
import matplotlib.pyplot as plt
import numpy as np
import torch

a = regularizers.gaussian2d(9, sigma=1.5)

plt.imshow(a)
#%%
a = torch.tensor(a)

#%%
a = a.repeat(5,1,1)
for i in range(a.shape[0]):
    a[i,:,:] = a[i,:,:]*i

#%%
    
plt.imshow(a[:,4,:])
#%%
a = a.repeat(20,1,1,1) # [Nsubs, Nlags, NY, NX]

plt.imshow(a[0,1,:,:])

#%%
plt.imshow(a[0,4,:,:])
#%%
flatten = nn.Flatten()
plt.plot(flatten(a)[0,:])
#%%
rr = regularizers.RegMats(dims=[5,9,9], type=['d2x'])
plt.imshow(rr.reg_mat.detach().cpu().numpy())
plt.colorbar()