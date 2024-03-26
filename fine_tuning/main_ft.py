import argparse
import sys
import os
import pytorch_lightning as pl
import soundfile as sf
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import summarize
from torch.utils.data import DataLoader
from pytorch_lightning.strategies.ddp import DDPStrategy
from ft_neural_branch import FinePLCModel
from CODE.config import CONFIG
from ft_dataset import TrainDataset
from CODE.neural_branch import PLCModel
from CODE.tblogger import TensorBoardLoggerExpanded
from CODE.utils import mkdir_p

pretrained_model = PLCModel.load_from_checkpoint('/nas/home/mviviani/nas/home/mviviani/tesi/CODE/lightning_logs/version_277/checkpoints/frn-epoch=66-val_loss=-8.5957.ckpt')
pretrained_model.train()
train_dataset = TrainDataset('train')
val_dataset = TrainDataset('val')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG.TRAIN.batch_size, shuffle=False, num_workers=CONFIG.TRAIN.workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG.TRAIN.batch_size, shuffle=False, num_workers=CONFIG.TRAIN.workers)

fine_tuning_model = FinePLCModel(train_dataset=train_dataset,
                         val_dataset=val_dataset,
                         window_size=CONFIG.DATA.window_size,
                         enc_layers=CONFIG.NN_MODEL.enc_layers,
                         enc_in_dim=CONFIG.NN_MODEL.enc_in_dim,
                         enc_dim=CONFIG.NN_MODEL.enc_dim,
                         pred_dim=CONFIG.NN_MODEL.pred_dim,
                         pred_layers=CONFIG.NN_MODEL.pred_layers,
                         model=pretrained_model)

checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=True,    
                                          filename='parcnet-{epoch:02d}-{val_loss:.4f}', save_weights_only=False, save_top_k=3)

gpus = CONFIG.gpus.split(',')
logger = TensorBoardLoggerExpanded(CONFIG.DATA.sr)

trainer = pl.Trainer(logger=logger,
                         gradient_clip_val=CONFIG.TRAIN.clipping_val,
                         devices=len(gpus),
                         max_epochs=CONFIG.TRAIN.epochs,
                         accelerator='auto',
                         callbacks=[checkpoint_callback])

trainer.fit(fine_tuning_model, train_loader, val_loader) 

trainer.save_checkpoint('/nas/home/mviviani/nas/home/mviviani/tesi/CODE/fine_tuning/fine_tuned_model.ckpt')


