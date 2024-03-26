import os
import librosa
import pytorch_lightning as pl
import soundfile as sf
import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from CODE import metr
from CODE.config import CONFIG
from CODE.blocks import Encoder, Predictor
from CODE.utils import visualize, LSD
from CODE.ar_branch import ARModel
from CODE.parcnet_loss import SpectralConvergenceLoss, SingleResolutionSTFTLoss, LogMagnitudeSTFTLoss, _spectrogram, \
    MultiResolutionSTFTLoss
sys.path.append("/nas/home/mviviani/nas/home/mviviani/tesi")

class FinePLCModel(pl.LightningModule):
    def __init__(self, train_dataset=None, val_dataset=None, window_size=960, enc_layers=4, enc_in_dim=384, enc_dim=768,
                 pred_dim=512, pred_layers=1, model=None, pred_ckpt_path=None): 
        super(FinePLCModel, self).__init__()     
        self.pretrained_model = model
        self.val_predictions = []
        self.train_predictions = []
        self.window_size = window_size
        self.hop_size = window_size // 2
        self.learning_rate = CONFIG.TRAIN.lr
        self.hparams.batch_size = CONFIG.TRAIN.batch_size
        self.num_epochs = CONFIG.TRAIN.epochs
        self.ar_order = CONFIG.AR_MODEL.ar_order
        self.diagonal_load = CONFIG.AR_MODEL.diagonal_load
        self.ar_model = ARModel(self.ar_order, self.diagonal_load)
        self.p_size = CONFIG.DATA.TRAIN.packet_size  
        self.fadeout = CONFIG.DATA.TRAIN.fadeout
        self.padding = CONFIG.DATA.TRAIN.padding
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lmbda = 100.0

        self.window = torch.sqrt(torch.hann_window(self.window_size)).to('cuda:0')
        self.save_hyperparameters('window_size', 'enc_layers', 'enc_in_dim', 'enc_dim', 'pred_dim', 'pred_layers')
        self.mse_loss = F.mse_loss
        self.stft_loss = MultiResolutionSTFTLoss()
        
    def forward(self, x):
        output = self.pretrained_model(x)
        return output

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.val_dataset.previous_predictions = None
        self.train_dataset.current_epoch = self.current_epoch
        
    def training_step(self, batch, batch_idx):
        ar_past, nn_past, target = batch

        f_0 = nn_past[:, :, 0:1, :]
        nn_input = nn_past[:, :, 1:, :]
        nn_pred = self(nn_input)
        nn_pred = torch.cat([f_0, nn_pred], dim=2)
        nn_pred = torch.view_as_complex(nn_pred.permute(0, 2, 3, 1).contiguous())
        nn_pred = torch.istft(nn_pred, self.window_size, self.hop_size, window=self.window)

        ar_pred = []
        ar_past = ar_past.cpu().numpy()

        for i in range(len(target)):  #20
            prediction = self.ar_model.predict(ar_past[i], (self.p_size + self.fadeout + self.padding))
            tmp = np.concatenate((ar_past[i], prediction), axis=None)
            ar_pred.append(tmp)
 
        ar_pred = torch.tensor(np.array(ar_pred))

        pred = nn_pred.to('cuda:0') + ar_pred.to('cuda:0')
        output = pred[:, 7 * self.p_size: 8 * self.p_size]   # tensor(20,320)

        self.train_predictions.append(output) 

        pred = pred.to(torch.double)     
        target = target.to(torch.double)
        
        mse_loss = self.mse_loss(pred, target)  

        sc_loss, log_loss = self.stft_loss(y_pred=pred.squeeze(1), y_true=target.squeeze(1))
        spectral_loss = 0.5 * (sc_loss + log_loss)  
        tot_loss = self.lmbda * mse_loss + spectral_loss  

        self.log('mse_loss', mse_loss, prog_bar=True)
        self.log('sc_loss', sc_loss, prog_bar=False)
        self.log('log_mag_loss', log_loss, prog_bar=False)
        self.log('spectral_loss', spectral_loss, prog_bar=True)
        self.log('tot_loss', tot_loss, prog_bar=True)
        
        return tot_loss

    def on_train_epoch_end(self):
        if self.current_epoch % 8 == 0:   
            self.train_dataset.previous_predictions = torch.cat(self.train_predictions, dim=0).detach().cpu().numpy()
        elif self.current_epoch % 8 == 7:  
            self.train_dataset.previous_predictions = None
            self.train_dataset.audio_index = torch.randint(0, CONFIG.DATA.audio_chunk_len - self.p_size*17, (1,))[0]
        else:
            self.train_dataset.previous_predictions = np.concatenate((self.train_dataset.previous_predictions, torch.cat(self.train_predictions, dim=0).detach().cpu().numpy()), axis=1)
        self.train_predictions.clear()
        self.train_predictions = []

    def validation_step(self, val_batch, batch_idx):
        ar_past, nn_past, target = val_batch
        
        f_0 = nn_past[:, :, 0:1, :]
        nn_input = nn_past[:, :, 1:, :]              
        nn_pred = self(nn_input)                     
        nn_pred = torch.cat([f_0, nn_pred], dim=2)   
        nn_pred = torch.view_as_complex(nn_pred.permute(0, 2, 3, 1).contiguous())
        nn_pred = torch.istft(nn_pred, self.window_size, self.hop_size, window=self.window)               # torch.Size([20, 2880])

        ar_pred = []
        ar_past = ar_past.cpu().numpy()

        for i in range(len(target)):
            prediction = self.ar_model.predict(ar_past[i], (self.p_size + self.fadeout + self.padding))   # 2240 + 640
            tmp = np.concatenate((ar_past[i], prediction), axis=None)
            ar_pred.append(tmp)

        ar_pred = torch.tensor(np.array(ar_pred))            # 9

        pred = nn_pred.to('cuda:0') + ar_pred.to('cuda:0')   # 2880
        output = pred[:, 7 * self.p_size: 8 * self.p_size]   # 320   

        self.val_predictions.append(output)

        val_loss = metr.nmse(y_pred=pred, y_true=target)    # 2880
        packet_val_loss = metr.nmse(y_pred=pred[..., -(self.p_size + self.fadeout + self.padding):], y_true=target[..., -(self.p_size + self.fadeout + self.padding):])   # last 640

        self.log('val_loss', val_loss)  
        self.log('packet_val_loss', packet_val_loss)

        return packet_val_loss, val_loss   
    
    def on_validation_epoch_end(self):
        if self.current_epoch % 8 == 0:
            self.val_dataset.previous_predictions = torch.cat(self.val_predictions, dim=0).detach().cpu().numpy()
        elif self.current_epoch % 8 == 7:
            self.val_dataset.previous_predictions = None
            self.val_dataset.audio_index = torch.randint(0, CONFIG.DATA.audio_chunk_len - self.p_size * 17, (1,))[0]
        else:
            self.val_dataset.previous_predictions = np.concatenate((self.val_dataset.previous_predictions, torch.cat(self.val_predictions, dim=0).detach().cpu().numpy()), axis=1)
        self.val_predictions.clear()
        self.val_predictions = []

    def save_checkpoint(self, checkpoint_path='fine_tuned_model.ckpt'):
        torch.save(self.state_dict(), checkpoint_path)
        print(f"Fine-tuned model saved at: {checkpoint_path}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=CONFIG.TRAIN.patience,
                                                                  factor=CONFIG.TRAIN.factor, verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'packet_val_loss'
        }
        return [optimizer], [scheduler]

