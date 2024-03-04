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

# Caricamento del modello preaddestrato:

pretrained_model = PLCModel.load_from_checkpoint('/nas/home/mviviani/nas/home/mviviani/tesi/CODE/lightning_logs/version_271/checkpoints/frn-epoch=126-val_loss=-8.9949.ckpt')

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

checkpoint_callback = ModelCheckpoint(monitor='packet_val_loss', mode='min', verbose=True,    # monitor='val_loss'  # <--------
                                          filename='parcnet-{epoch:02d}-{packet_val_loss:.4f}', save_weights_only=False, save_top_k=8)

gpus = CONFIG.gpus.split(',')
logger = TensorBoardLoggerExpanded(CONFIG.DATA.sr)

trainer = pl.Trainer(logger=logger,
                         gradient_clip_val=CONFIG.TRAIN.clipping_val,
                         devices=len(gpus),
                         max_epochs=CONFIG.TRAIN.epochs,
                         accelerator='auto',
                         callbacks=[checkpoint_callback])

# trainer.fit(fine_tuned_model, train_dataloader)
# In questa fase, stai creando un'istanza del tuo modello di fine-tuning (FineTunedModel) 
# e lo stai addestrando con il tuo dataloader di training utilizzando PyTorch Lightning Trainer.
trainer.fit(fine_tuning_model, train_loader, val_loader) #, fine_tuning_model.train_dataloader())  # , train_dataloader)?

trainer.save_checkpoint('/nas/home/mviviani/nas/home/mviviani/tesi/CODE/fine_tuning/fine_tuned_model_4.ckpt')

