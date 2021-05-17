import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

# import regularizers
from V1FreeViewingCode.models.readouts import Readout
from V1FreeViewingCode.models.cores import Core

"""
Main encoder class
"""
class Encoder(LightningModule):
    def __init__(self,
        core=Core(),
        readout=Readout(),
        output_nl=nn.Softplus(),
        loss=nn.PoissonNLLLoss(log_input=False),
        val_loss=None,
        detach_core=False,
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
        self.core = core
        self.readout = readout
        self.detach_core = detach_core
        self.save_hyperparameters('learning_rate','batch_size',
            'num_workers', 'data_dir', 'optimizer', 'weight_decay', 'amsgrad', 'betas',
            'max_iter')          
        
        if val_loss is None:
            self.val_loss = loss
        else:
            self.val_loss = val_loss

        self.output_nl = output_nl
        self.loss = loss

    def forward(self, x, shifter=None, sample=None):
        x = self.core(x)
        if self.detach_core:
            x = x.detach()
        if "shifter" in dir(self.readout) and self.readout.shifter and shifter is not None:
            x = self.readout(x, shift=self.readout.shifter(shifter))
        else:
            x = self.readout(x)

        return self.output_nl(x)

    def training_step(self, batch, batch_idx):
        x = batch['stim']
        y = batch['robs']
        if "shifter" in dir(self.readout) and batch['eyepos'] is not None and self.readout.shifter:
            y_hat = self(x, shifter=batch['eyepos'])
        else:
            y_hat = self(x)

        loss = self.loss(y_hat, y)
        regularizers = int(not self.detach_core) * self.core.regularizer() + self.readout.regularizer()

        self.log('train_loss', loss + regularizers)
        return {'loss': loss + regularizers}

    def validation_step(self, batch, batch_idx):

        x = batch['stim']
        y = batch['robs']
        if "shifter" in dir(self.readout) and batch['eyepos'] is not None and self.readout.shifter:
            y_hat = self(x, shifter=batch['eyepos'])
        else:
            y_hat = self(x)
        loss = self.val_loss(y_hat, y)
        self.log('val_loss', loss)
        return {'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        # logging
        if(self.current_epoch==1):
            self.logger.experiment.add_text('core', str(dict(self.core.hparams)))
            self.logger.experiment.add_text('readout', str(dict(self.readout.hparams)))

        avg_val_loss = torch.tensor([x['loss'] for x in validation_step_outputs]).mean()

        self.log('val_loss', avg_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return


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
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    def on_save_checkpoint(self, checkpoint):
        # track the core, readout, shifter class and state_dicts
        checkpoint['core_type'] = type(self.core)
        checkpoint['core_hparams'] = self.core.hparams
        checkpoint['core_state_dict'] = self.core.state_dict()

        checkpoint['readout_type'] = type(self.readout)
        checkpoint['readout_hparams'] = self.readout.hparams
        checkpoint['readout_state_dict'] = self.readout.state_dict() # TODO: is this necessary or included in self state_dict?

        # checkpoint['shifter_type'] = type(self.shifter)
        # if checkpoint['shifter_type']!=type(None):
        #     checkpoint['shifter_hparams'] = self.shifter.hparams
        #     checkpoint['shifter_state_dict'] = self.shifter.state_dict() # TODO: is this necessary or included in model state_dict?

    def on_load_checkpoint(self, checkpoint):
        # properly handle core, readout, shifter state_dicts
        self.core = checkpoint['core_type'](**checkpoint['core_hparams'])
        self.readout = checkpoint['readout_type'](**checkpoint['readout_hparams'])
        # if checkpoint['shifter_type']!=type(None):
        #     self.shifter = checkpoint['shifter_type'](**checkpoint['shifter_hparams'])
        #     self.shifter.load_state_dict(checkpoint['shifter_state_dict'])
        self.core.load_state_dict(checkpoint['core_state_dict'])
        self.readout.load_state_dict(checkpoint['readout_state_dict'])

class EncoderMod(LightningModule): # IN PROGRESS
    def __init__(self,
        core=Core(),
        readout=Readout(),
        output_nl=nn.Softplus(),
        modifiers=None,
        gamma_mod=.1,
        loss=nn.PoissonNLLLoss(log_input=False),
        val_loss=None,
        detach_core=False,
        learning_rate=1e-3,
        batch_size=1000,
        num_workers=0,
        data_dir='',
        optimizer='AdamW',
        weight_decay=1e-2,
        amsgrad=False,
        betas=[.9,.999],
        max_iter=10000,
        lrscheduler=True,
        **kwargs):

        super().__init__()
        self.core = core
        self.readout = readout
        self.detach_core = detach_core
        self.save_hyperparameters('learning_rate','batch_size',
            'num_workers', 'data_dir', 'optimizer', 'weight_decay', 'amsgrad', 'betas',
            'max_iter','modifiers', 'gamma_mod', 'lrscheduler')
        
        # initialize variables for modifier: these all need to be here regardless of whether the modifiers are used so we can load the model checkpoints
        self.offsets = nn.ModuleList()
        self.gains = nn.ModuleList()
        self.offsetstims = []
        self.gainstims = []
        self.modify = False
        self.register_buffer("offval", torch.zeros(1))
        self.register_buffer("gainval", torch.ones(1))

        if self.hparams.modifiers is not None:
            """
            modifier is a hacky addition to the model to allow for offsets and gains at a certain stage in the model
            The default stage is after the readout
            example modifier input:
            modifier = {'stimlist': ['frametent', 'saccadeonset'],
            'gain': [40, None],
            'offset':[40,20],
            'stage': "readout",
            'outdims: gd.NC}
            """
            if type(self.hparams.modifiers)==dict:
                self.modify = True

                nmods = len(self.hparams.modifiers['stimlist'])
                assert nmods==len(self.hparams.modifiers["offset"]), "Encoder: modifier specified incorrectly"
                
                if 'stage' not in self.hparams.modifiers.keys():
                    self.hparams.modifiers['stage'] = "readout"
                
                # set the output dims (this hast to match either the readout output the whole core is modulated)
                if self.hparams.modifiers['stage']=="readout":
                    outdims = self.hparams.modifiers['outdims']
                elif self.hparams.modifiers['stage']=="core":
                    outdims = 1

                self.modifierstage = self.hparams.modifiers["stage"]
                for imod in range(nmods):
                    if self.hparams.modifiers["offset"][imod] is not None:
                        self.offsetstims.append(self.hparams.modifiers['stimlist'][imod])
                        self.offsets.append(nn.Linear(self.hparams.modifiers["offset"][imod], outdims, bias=False))
                    if self.hparams.modifiers["gain"][imod] is not None:
                        self.gainstims.append(self.hparams.modifiers['stimlist'][imod])
                        self.gains.append(nn.Linear(self.hparams.modifiers["gain"][imod], outdims, bias=False))
        else:
            self.modify = False

        if val_loss is None:
            self.val_loss = loss
        else:
            self.val_loss = val_loss

        self.output_nl = output_nl
        self.loss = loss

    def forward(self, x, shifter=None, sample=None):
        dlist = dir(self)
        if "offsets" in dlist:
            use_offsets = True
        else:
            use_offsets = False

        if "gains" in dlist:
            use_gains = True
        else:
            use_gains = False

        offset = self.offval
        if use_offsets:
            for offmod,stim in zip(self.offsets, self.offsetstims):
                offset = offset + offmod(sample[stim])

        gain = self.gainval
        if use_gains:
            for gainmod,stim in zip(self.gains, self.gainstims):
                gain = gain * (self.gainval + gainmod(sample[stim]))

        if self.modify and self.modifierstage=="stim":
            x *= gain
            x += offset

        x = self.core(x)

        if self.detach_core:
            x = x.detach()
        
        if self.modify and self.modifierstage=="core":
            x *= gain
            x += offset

        if "shifter" in dir(self.readout) and self.readout.shifter and shifter is not None:
            shift = self.readout.shifter(shifter)
        else:
            shift = None

        x = self.readout(x, shift=shift)

        if self.modify and self.modifierstage=="readout":
            x *= gain
            x += offset

        return self.output_nl(x)

    def training_step(self, batch, batch_idx):
        x = batch['stim']
        y = batch['robs']
        if "shifter" in dir(self.readout) and batch['eyepos'] is not None and self.readout.shifter:
            shift = batch['eyepos']
        else:
            shift = None

        if self.modify:
            y_hat = self(x, shifter=shift, sample=batch)
        else:
            y_hat = self(x, shifter=shift)
        

        loss = self.loss(y_hat, y)
        regularizers = int(not self.detach_core) * self.core.regularizer() + self.readout.regularizer()
        # regularizers for modifiers
        reg = 0
        if self.modify:
            for imod in range(len(self.offsets)):
                reg += self.offsets[imod].weight.pow(2).sum().sqrt()
            for imod in range(len(self.gains)):
                reg += self.offsets[imod].weight.pow(2).sum().sqrt()

        self.log('train_loss', loss, 'reg_pen', regularizers + self.hparams.gamma_mod * reg)
        return {'loss': loss + regularizers + self.hparams.gamma_mod * reg}
    
    def validation_step(self, batch, batch_idx):

        x = batch['stim']
        y = batch['robs']
        if "shifter" in dir(self.readout) and batch['eyepos'] is not None and self.readout.shifter:
            shift = batch['eyepos']
        else:
            shift = None

        if self.modify:
            y_hat = self(x, shifter=shift, sample=batch)
        else:
            y_hat = self(x, shifter=shift)
        

        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return {'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        # logging
        if(self.current_epoch==1):
            self.logger.experiment.add_text('core', str(dict(self.core.hparams)))
            self.logger.experiment.add_text('readout', str(dict(self.readout.hparams)))

        avg_val_loss = torch.tensor([x['loss'] for x in validation_step_outputs]).mean()

        self.log('val_loss', avg_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return


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
        
        if self.hparams.lrscheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            out = {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        else:
            out = optimizer
        
        return out
    
    def on_save_checkpoint(self, checkpoint):
        # track the core, readout, shifter class and state_dicts
        checkpoint['core_type'] = type(self.core)
        checkpoint['core_hparams'] = self.core.hparams
        checkpoint['core_state_dict'] = self.core.state_dict()

        checkpoint['readout_type'] = type(self.readout)
        checkpoint['readout_hparams'] = self.readout.hparams
        checkpoint['readout_state_dict'] = self.readout.state_dict() # TODO: is this necessary or included in self state_dict?

    def on_load_checkpoint(self, checkpoint):
        # properly handle core, readout, shifter state_dicts
        self.core = checkpoint['core_type'](**checkpoint['core_hparams'])
        self.readout = checkpoint['readout_type'](**checkpoint['readout_hparams'])
        self.core.load_state_dict(checkpoint['core_state_dict'])
        self.readout.load_state_dict(checkpoint['readout_state_dict'])

        # # check if there are modifiers and
        # keylist = list(checkpoint['state_dict'].keys())
        # offsetlist = [key if key[0:6]=='offset' for key in keylist]

import V1FreeViewingCode.models.layers as layers
import V1FreeViewingCode.models.regularizers as regularizers
class GLM(LightningModule):
    def __init__(self,
        input_size,
        output_size,
        bias = True,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        tik_reg_types=["d2x", "d2t"],
        tik_reg_amt=[.005,.001],
        output_nl=nn.Softplus(),
        loss=nn.PoissonNLLLoss(log_input=False),
        val_loss=None,
        learning_rate=1e-3,
        data_dir='',
        optimizer='AdamW',
        weight_decay=1e-2,
        amsgrad=False,
        betas=[.9,.999],
        max_iter=10000,
        **kwargs):

        super().__init__()
        
        self.save_hyperparameters()

        regularizer_config = {'dims': input_size,
                            'type': tik_reg_types, 'amount': tik_reg_amt}
        self._tik_weights_regularizer = regularizers.__dict__["RegMats"](**regularizer_config)
        
        
        if val_loss is None:
            self.val_loss = loss
        else:
            self.val_loss = val_loss

        self.linear = layers.ShapeLinear(input_size, output_size, bias=bias)
        self.output_nl = output_nl
        self.loss = loss

    def forward(self, x):
        x = self.linear(x)

        return self.output_nl(x)

    def training_step(self, batch, batch_idx):
        x = batch['stim']
        y = batch['robs']
        
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.hparams.l2_strength * l2_reg
        
        
        loss += self._tik_weights_regularizer(self.linear.weight)

        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        x = batch['stim']
        y = batch['robs']
        y_hat = self(x)
        loss = self.val_loss(y_hat, y)

        self.log('val_loss', loss)
        return {'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        # logging
        avg_val_loss = torch.tensor([x['loss'] for x in validation_step_outputs]).mean()
        self.log('val_loss', avg_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return


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
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}