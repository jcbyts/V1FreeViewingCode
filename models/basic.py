import os
from argparse import ArgumentParser
# from warnings import warn

import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

"""
This is early days
"""

class Poisson(LightningModule):
    def __init__(self,
        learning_rate=1e-3,
        batch_size=1000,
        num_workers=0,
        data_dir='',
        optimizer='AdamW',
        weight_decay=1e-2,
        amsgrad=False,
        betas=[.9,.999],
        max_iter=10000,
        **kwargs):

        super().__init__()
        self.save_hyperparameters()

        self.loss = nn.PoissonNLLLoss(log_input=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        self.log('val_loss', results['loss'])
        return results

    def validation_epoch_end(self, validation_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in validation_step_outputs]).mean()
        return {'val_loss': avg_val_loss}


    def configure_optimizers(self):
        
        if self.hparams.optimizer=='LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(),
                lr=self.hparams.learning_rate,
                max_iter=10000) #, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100)
        elif self.hparams.optimizer=='AdamW':
            optimizer = torch.optim.AdamW(self.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                weight_decay=self.hparams.weight_decay,
                amsgrad=self.hparams.amsgrad)
        elif self.hparams.optimizer=='Adam':
            optimizer = torch.optim.Adam(self.parameters(),
            lr=self.hparams.learning_rate)
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.max_iter, eta_min=1e-8)
        
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optimizer
        

class LNP(Poisson):
    def __init__(self, input_dim=(15, 8, 6),
        output_dim=128,
        **kwargs):

        super().__init__()
        self.save_hyperparameters()
        
        self.l0 = nn.Flatten()
        self.l1 = nn.Linear(np.prod(self.hparams.input_dim), self.hparams.output_dim ,bias=True)
        self.spikeNL = nn.Softplus()


    def forward(self, x):
        x = self.l0(x)
        x = self.spikeNL(self.l1(x))
        return x

class cNIM(Poisson):
    '''
    convolutional NIM w/ fit 
    '''
    def __init__(self, input_dim=128,
        n_temporal=2,
        n_hidden=10,
        output_dim=128,
        ei_split=0,
        ksize=9,
        normalization=0,
        l1reg=0,
        **kwargs):

        super().__init__()
        self.save_hyperparameters()

        self.temporal = temporalLayer(input_dim[0], self.hparams.n_temporal)
        self.conv1 = nn.Conv2d(in_channels=self.hparams.n_temporal, out_channels=n_hidden, kernel_size=ksize, bias=False)

        self.flatten = nn.Flatten()

        NY = input_dim[1]-ksize+1
        NX = input_dim[2]-ksize+1

        # build ei_mask
        if ei_split > 0:
            self.posconstraint = weightConstraint()
            ni = ei_split
            ne = n_hidden - ni
            eimask = torch.cat( (torch.ones((1,ne, NY, NX)),
                -1*torch.ones((1,ni, NY, NX))), axis=1)
            self.register_buffer("ei_mask", eimask)
        else:
            eimask = torch.ones((1,self.hparams.n_hidden, NY, NX))
            self.register_buffer("ei_mask", eimask)

        if self.hparams.normalization==1: # batch norm
            self.norm = nn.BatchNorm2d(n_hidden)
        elif self.hparams.normalization==2:
            self.norm = nn.GroupNorm(1, n_hidden) # layer norm

        self.hparams.readoutNX = NX
        self.hparams.readoutNY = NY
        # self.ei_mask = nn.Conv2d(in_channels=n_hidden, out_channels=1,kernel_size=1, bias=False)
        # if ei_split==0:
        #     # self.state_dict()['ei_mask.weight'][:] = torch.ones((n_hidden,1,1,1))
        #     self.state_dict()['ei_mask.weight'][:] = torch.ones((1,n_hidden,1,1))
        # else:
        #     ne = n_hidden - ei_split
        #     ni = ei_split
        #     # self.state_dict()['ei_mask.weight'][:] = torch.cat((torch.ones((ne,1,1,1)), -1*torch.ones((ni,1,1,1))), axis=0)
        #     self.state_dict()['ei_mask.weight'][:] = torch.cat((torch.ones((1,ne,1,1)), -1*torch.ones((1,ni,1,1))), axis=1)
        
        # self.ei_mask.weight.requires_grad = False # ignore gradient

        self.nl = nn.ReLU()
        self.readout = nn.Linear(self.hparams.n_hidden*NX*NY, self.hparams.output_dim , bias=True)
        
        self.spikeNL = nn.Softplus()

    def forward(self, x):
        x = self.temporal(x)
        x = self.nl(self.conv1(x))
        if self.hparams.normalization>0:
            x = self.norm(x)
        x = self.flatten(x*self.ei_mask)
        if self.hparams.ei_split>0:
            self.readout.apply(self.posconstraint)
        x = self.spikeNL(self.readout(x))
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if self.hparams.l1reg > 0:
            loss += self.hparams.l1reg * torch.norm(self.readout.weight, 1)
        self.log('train_loss', loss)
        return {'loss': loss}

    # learning rate warm-up
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        
    #     # update params
    #     optimizer.step()
    #     if self.hparams.ei_split>0:
    #         self.l2.apply(self.posconstraint)
    #     optimizer.zero_grad()
# end cNIM


class sGQM(Poisson):
    def __init__(self, input_dim=128,
        n_hidden=10,
        output_dim=128,
        ei_split=0,
        relu=False,
        filternorm=0,
        reduction=10,
        normalization=0,
        num_groups=2,
        l1reg=0,
        **kwargs):

        super().__init__()
        self.save_hyperparameters()

        self.flatten = nn.Flatten()
        if filternorm>0:
            self.linear1 = nn.utils.weight_norm(nn.Linear(self.hparams.input_dim, self.hparams.n_hidden ,bias=False), dim=0, name='weight')
        elif filternorm<0: # max norm
            self.linear1 = nn.utils.weight_norm(nn.Linear(self.hparams.input_dim, self.hparams.n_hidden ,bias=False), dim=None, name='weight')
        else:
            self.linear1 = nn.Linear(self.hparams.input_dim, self.hparams.n_hidden ,bias=True)
            

        if self.hparams.filternorm > 0:
            self.filternorm = filterNorm()
            self.linear1.apply(self.filternorm)

        self.relu = nn.ReLU()
        self.nl = Quad()
        if filternorm == 0:
            self.readout = nn.Linear(self.hparams.n_hidden, self.hparams.output_dim ,bias=True)
        else:
            self.readout = nn.utils.weight_norm(nn.Linear(self.hparams.n_hidden, self.hparams.output_dim ,bias=True), dim=None)    
            

        if self.hparams.normalization==1: # batch norm
            self.norm = nn.BatchNorm1d(n_hidden)
        elif self.hparams.normalization==2:
            self.norm = nn.GroupNorm(self.hparams.num_groups, n_hidden)
        
        # self.seLayer = SELayerLinear(self.hparams.n_hidden, reduction)
        self.spikeNL = nn.Softplus()

    def forward(self, x):
        x = self.flatten(x)
        if self.hparams.filternorm > 0:
            self.linear1.apply(self.filternorm)

        x = self.linear1(x)
        if self.hparams.relu:
            xlin = self.relu(x[:, :self.hparams.ei_split])
        else:
            xlin = x[:, :self.hparams.ei_split]

        x = torch.cat( (xlin, self.nl(x[:,self.hparams.ei_split:])), axis=1)
        if self.hparams.normalization>0:
            x = self.norm(x)
        x = self.spikeNL(self.readout(x))

        return x

    def plot_filters(self, gd, sort=False):
        import matplotlib.pyplot as plt  # plotting
        w = self.linear1.weight.detach().cpu().numpy()
        nfilt = w.shape[0]
        if sort:
            n = np.asarray([self.linear1.weight[i,:].abs().max().detach().numpy() for i in range(nfilt)])
            cinds = np.argsort(n)[::-1][-len(n):]
        else:
            cinds = np.arange(0, nfilt)
        
        sx = np.ceil(np.sqrt(nfilt*2))
        sy = np.round(np.sqrt(nfilt*2))
        # sx,sy = U.get_subplot_dims(nfilt*2)
        mod2 = sy % 2
        sy += mod2
        sx -= mod2

        plt.figure(figsize=(10,10))
        for cc,jj in zip(cinds, range(nfilt)):
            plt.subplot(sx,sy,jj*2+1)
            wtmp = np.reshape(w[cc,:], (gd.num_lags, gd.NX*gd.NY))
            bestlag = np.argmax(np.std(wtmp, axis=1))
            plt.imshow(np.reshape(wtmp[bestlag,:], (gd.NY, gd.NX)), interpolation=None)
            wmax = np.argmax(wtmp[bestlag,:])
            wmin = np.argmin(wtmp[bestlag,:])
            plt.axis("off")

            plt.subplot(sx,sy,jj*2+2)
            plt.plot(wtmp[:,wmax], 'b-')
            plt.plot(wtmp[:,wmin], 'r-')
            plt.axhline(0, color='k')
            plt.axvline(bestlag, color=(.5, .5, .5))
            plt.axis("off")


    def plot_readout(self, sort=False):
        w2 = self.readout.weight.detach().cpu().numpy()

        plt.imshow(w2)


class seGQM(sGQM):
    def __init(self, reduction=16, **kwargs):
        super().__init__()
        # super(seGQM, self).__init__()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.flatten(x)
        if self.hparams.filternorm > 0:
            self.linear1.apply(self.filternorm)

        x = self.linear1(x)
        if self.hparams.relu:
            xlin = self.relu(x[:, :self.hparams.ei_split])
        else:
            xlin = x[:, :self.hparams.ei_split]

        x = torch.cat( (xlin, self.nl(x[:,self.hparams.ei_split:])), axis=1)
        if self.hparams.normalization>0:
            x = self.norm(x)

        x = self.seLayer(x)
        x = self.spikeNL(self.readout(x))

        return x



class sNIM(Poisson):
    def __init__(self, input_dim=128,
        n_hidden=10,
        output_dim=128,
        ei_split=0,
        normalization=0,
        l1reg=0,
        **kwargs):

        super().__init__()
        self.save_hyperparameters()

        self.l0 = nn.Flatten()
        self.l1 = nn.Linear(self.hparams.input_dim, self.hparams.n_hidden ,bias=True)
        
        if ei_split>0:
            self.posconstraint = weightConstraint()
            ni = ei_split
            ne = self.hparams.n_hidden-ni
            self.register_buffer("ei_mask", torch.cat((torch.ones((1,ne)), -torch.ones((1,ni))), axis=1))
            self.nl = nn.ReLU()
            self.l2 = nn.Linear(self.hparams.n_hidden, self.hparams.output_dim ,bias=True)
            self.l2.apply(self.posconstraint)
        else:
            self.nl = nn.ReLU()
            self.register_buffer("ei_mask", torch.ones((1,self.hparams.n_hidden)))
            self.l2 = nn.Linear(self.hparams.n_hidden, self.hparams.output_dim ,bias=True)
        
        if self.hparams.normalization==1: # batch norm
            self.norm = nn.BatchNorm1d(n_hidden)
        elif self.hparams.normalization==2:
            self.norm = nn.GroupNorm(1, n_hidden)

        self.spikeNL = nn.Softplus()

    def forward(self, x):
        x = self.l0(x)
        x = self.nl(self.l1(x))
        if self.hparams.normalization>0:
            x = self.norm(x)
        x *= self.ei_mask
        if self.hparams.ei_split > 0:
            self.l2.apply(self.posconstraint)
        x = self.spikeNL(self.l2(x))
        return x

    
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):

    #     # update params
    #     optimizer.step()
    #     if self.hparams.ei_split > 0:
    #         self.l2.apply(self.posconstraint)
    #     optimizer.zero_grad()

class temporalLayer(nn.Module):
    '''
    Integrate out time with fitted kernels
    Input: [Nbatch, NT, NY, NX]
    0: permute (Nbatch, NY, NX, NT)
    1: reshape (Nbatch x NX x NY, NT)
    2: Linear (NT x Nkernel)
    3: reshape (Nbatch, NY, NX, Nkernel)
    4: permute (Nbatch, Nkernel, NY, NX)
    '''

    def __init__(self, numlags, numk):
        super(temporalLayer, self).__init__()
        self.numk = numk
        self.numlags = numlags
        self.linear = nn.Linear(numlags,numk,bias=True)
        self.posMaxConstraint = weightConstraintTemporal()

    def forward(self, x):
        sz = x.size()
        x = x.permute(0, 2, 3, 1) #[N, T, Y, X] --> [N, Y, X, T]
        x = x.reshape((-1, sz[1])) # [N, Y, X, T] --> [N*X*Y, T]
        self.linear.apply(self.posMaxConstraint)
        x = self.linear(x) # [N*X*Y, T] --> [N*X*Y, K]
        x = x.reshape(sz[0],sz[2],sz[3],self.numk) # [N*X*Y, K] --> [N, Y, X, K]
        x = x.permute(0,3,1,2) # [N, Y, X, K] --> [N, K, Y, X]
        return x

class SELayerLinear(nn.Module): # Squeeze and Expand Layer (acts as a gate)
    def __init__(self, channel, reduction=16):
        super(SELayerLinear, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False), Swish(),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid(),
        )
        
    def forward(self,x):
        # sz = list(x.size())
        # b = sz[0]
        # c = sz[1]
        y = self.fc(x)
        return x * y

class SELayerConv(nn.Module): # Squeeze and Expand Layer (acts as a gate)
    def __init__(self, channel, reduction=16):
        super(SELayerConv, self).__init__()
        self.avg_pool = nn.AvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False), Swish(),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid(),
        )
        
    def forward(self,x):
        sz = list(x.size())
        b = sz[0]
        c = sz[1]
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1,1)
        return x * y.expand_as(x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class Quad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x**2

class filterNorm(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            # print("Entered")
            w = module.weight.data
            s = w.sum(axis=1)
            for i in range(len(s)):
                w[i,:]/=s[i].abs().clamp(.001)

            module.weight.data = w

class weightConstraintTemporal(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            # print("Entered")
            w = module.weight.data
            t = w.abs().argmax(axis=1)
            s = w[:,t].diag().sign()
            for i in range(len(s)):
                w[i,:]*=s[i]
            module.weight.data=w

class weightConstraint(object):
    def __init__(self,minval=0.0):
        self.minval = minval
        # pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            # print("Entered")
            module.weight.data.clamp_(self.minval)