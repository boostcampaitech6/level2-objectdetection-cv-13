import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import DetaImageProcessor

from utils import set_seed, collate_fn
from datasets import PLCocoDataset
from models import DETA


if __name__ == '__main__':
    # set seed
    set_seed(2022)
    
    ## wandb
    wandb_logger = WandbLogger(
        project="Boost Camp Lv2-1",
        entity="frostings",
        name=f"DETA-swin-large",
    )
    
    if not os.path.exists("ckpt"):
        os.makedirs("ckpt")
        
    ## checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath="ckpt/",
        save_top_k=5,
        monitor="v_loss",
        mode='min',
        filename="DETA_{epoch}_{t_loss}_{v_loss}"
    )
    
    ## trainer
    trainer = pl.Trainer(
        strategy='ddp',
        accelerator="gpu",
        devices=1,
        max_epochs=20, 
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.1,
        precision='16'
    )
    
    dataset = PLCocoDataset()
    _processor = dataset.setup()
    pl_model = DETA(
        processor=_processor, 
        train_dataset=dataset.train_dataset, 
        val_dataset=dataset.val_dataset
    )
    trainer.fit(pl_model, datamodule=dataset)