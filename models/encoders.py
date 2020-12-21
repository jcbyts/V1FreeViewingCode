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

    def forward(self, x, shifter=None):
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
        tqdm_dict = {'val_loss': avg_val_loss}

        return {
                'progress_bar': tqdm_dict,
                'log': {'val_loss': avg_val_loss},
        }


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