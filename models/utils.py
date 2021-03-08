import numpy as np
import torch

def get_trainer(dataset,
        version=1,
        save_dir='./checkpoints',
        name='jnkname',
        auto_lr=False,
        batchsize=1000,
        earlystopping=True,
        seed=None):
    """
    Returns a pytorch lightning trainer and splits the training set into "train" and "valid"
    """
    from torch.utils.data import Dataset, DataLoader, random_split
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TestTubeLogger
    from pathlib import Path

    
    save_dir = Path(save_dir)
    n_val = np.floor(len(dataset)/5).astype(int)
    n_train = (len(dataset)-n_val).astype(int)

    gd_train, gd_val = random_split(dataset, lengths=[n_train, n_val])

    # build dataloaders
    train_dl = DataLoader(gd_train, batch_size=batchsize)
    valid_dl = DataLoader(gd_val, batch_size=batchsize)

    # Train
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    logger = TestTubeLogger(
        save_dir=save_dir,
        name=name,
        version=version  # fixed to one to ensure checkpoint load
    )

    # ckpt_folder = save_dir / sessid / 'version_{}'.format(version) / 'checkpoints'
    if earlystopping:
        trainer = Trainer(gpus=1, callbacks=[early_stop_callback],
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            deterministic=False,
            gradient_clip_val=0,
            accumulate_grad_batches=1,
            progress_bar_refresh_rate=20,
            max_epochs=1000,
            auto_lr_find=auto_lr)
    else:
        trainer = Trainer(gpus=1,
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            deterministic=False,
            gradient_clip_val=0,
            accumulate_grad_batches=1,
            progress_bar_refresh_rate=20,
            max_epochs=300,
            auto_lr_find=auto_lr)

    if seed:
        seed_everything(seed)

    return trainer, train_dl, valid_dl

def find_best_epoch(ckpt_folder):
    # from os import listdir
    # import glob
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    try:
        # ckpt_files = listdir(ckpt_folder)  # list of strings
        ckpt_files = list(ckpt_folder.glob('*.ckpt'))
        epochs = [int(str(filename)[str(filename).find('=')+1:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
        out = max(epochs)
    except FileNotFoundError:
        out = None
    return out

def get_null_adjusted_ll(model, sample, bits=False):
    '''
    get null-adjusted log likelihood
    bits=True will return in units of bits/spike
    '''
    m0 = model.cpu()
    loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
    lnull = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy().sum(axis=0)
    yhat = m0(sample['stim'], shifter=sample['eyepos'])
    llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
    rbar = sample['robs'].sum(axis=0).numpy()
    ll = (llneuron - lnull)/rbar
    if bits:
        ll/=np.log(2)
    return ll