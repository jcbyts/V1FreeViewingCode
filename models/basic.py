import os
from argparse import ArgumentParser
# from warnings import warn

import torch
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

class Poisson(LightningModule):
    def __init__(self,
        learning_rate=1e-3,
        batch_size=1000,
        num_workers=4,
        data_dir='',
        optimizer='AdamW',
        weight_decay=1e-2,
        max_iter=10000,
        **kwargs):

        super().__init__()
        self.save_hyperparameters()

        self.loss = torch.nn.PoissonNLLLoss(log_input=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return results
        # x, y = batch
        # y_hat = self(x)
        # loss = self.loss(y_hat, y)
        # self.log('val_loss', loss)

    def validation_epoch_end(self, validation_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in validation_step_outputs]).mean()
        return {'val_loss': avg_val_loss}
    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.loss(y_hat, y)
    #     self.log('train_loss', loss)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.loss(y_hat, y)
    #     self.log('val_loss', loss)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.poisson_nll_loss(y_hat, y)
    #     self.log('test_loss', loss)

    def configure_optimizers(self):
        
        if self.hparams.optimizer=='LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(),
                lr=self.hparams.learning_rate,
                max_iter=10000) #, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100)
        elif self.hparams.optimizer=='AdamW':
            optimizer = torch.optim.AdamW(self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                amsgrad=True)
        elif self.hparams.optimizer=='Adam':
            optimizer = torch.optim.Adam(self.parameters(),
            lr=self.hparams.learning_rate)

        return optimizer

class LNP(Poisson):
    def __init__(self, input_dim=(15, 8, 6),
        output_dim=128,
        **kwargs):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.save_hyperparameters()
        
        self.l0 = torch.nn.Flatten()
        self.l1 = torch.nn.Linear(np.prod(self.input_dim), self.output_dim ,bias=True)
        self.spikeNL = torch.nn.Softplus()


    def forward(self, x):
        x = self.l0(x)
        x = self.spikeNL(self.l1(x))
        return x

    

    # def prepare_data(self):

    #     dataset = GratingDataset
    #     # MNIST(self.hparams.data_dir, train=True, download=True, transform=transforms.ToTensor())

    # def train_dataloader(self):
    #     dataset = MNIST(self.hparams.data_dir, train=True, download=False, transform=transforms.ToTensor())
    #     mnist_train, _ = random_split(dataset, [55000, 5000])
    #     loader = DataLoader(mnist_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    #     return loader

    # def val_dataloader(self):
    #     dataset = MNIST(self.hparams.data_dir, train=True, download=False, transform=transforms.ToTensor())
    #     _, mnist_val = random_split(dataset, [55000, 5000])
    #     loader = DataLoader(mnist_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    #     return loader

    # def test_dataloader(self):
    #     test_dataset = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    #     loader = DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    #     return loader

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument('--batch_size', type=int, default=32)
    #     parser.add_argument('--num_workers', type=int, default=4)
    #     parser.add_argument('--hidden_dim', type=int, default=128)
    #     parser.add_argument('--data_dir', type=str, default='')
    #     parser.add_argument('--learning_rate', type=float, default=0.0001)
    #     return parser




class sNIM(LightningModule):
    def __init__(self, input_dim=128,
        n_hidden=10,
        output_dim=128,
        learning_rate=1e-3,
        batch_size=1000,
        num_workers=4,
        data_dir='',
        optimizer='AdamW',
        weight_decay=1e-1,
        max_iter=10000,
        **kwargs):

        super().__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim
        self.hparams.learning_rate = learning_rate
        self.hparams.batch_size = batch_size
        self.hparams.num_workers = 4
        self.hparams.data_dir = ''
        self.hparams.optimizer = optimizer
        self.hparams.weight_decay = weight_decay
        self.hparams.max_iter = max_iter


        self.l1 = torch.nn.Linear(self.input_dim, self.n_hidden ,bias=True)
        self.nl = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(self.n_hidden, self.output_dim ,bias=True)
        self.spikeNL = torch.nn.Softplus()

        self.loss = torch.nn.PoissonNLLLoss(log_input=False)

    def forward(self, x):
        x = self.nl(self.l1(x))
        x = self.spikeNL(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return results
        # x, y = batch
        # y_hat = self(x)
        # loss = self.loss(y_hat, y)
        # self.log('val_loss', loss)

    def validation_epoch_end(self, validation_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in validation_step_outputs]).mean()
        return {'val_loss': avg_val_loss}


    def test_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        self.log('test_loss', results['loss'])
        return results
        # x, y = batch
        # y_hat = self(x)
        # loss = F.poisson_nll_loss(y_hat, y)
        # self.log('test_loss', loss)

    def configure_optimizers(self):
        
        if self.hparams.optimizer=='LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(),
                lr=self.hparams.learning_rate,
                max_iter=10000) #, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100)
        elif self.hparams.optimizer=='AdamW':
            optimizer = torch.optim.AdamW(self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                amsgrad=True)
        elif self.hparams.optimizer=='Adam':
            optimizer = torch.optim.Adam(self.parameters(),
            lr=self.hparams.learning_rate)

        return optimizer