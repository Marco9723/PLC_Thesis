import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("/nas/home/mviviani/nas/home/mviviani/tesi")

from CODE.config import CONFIG
import CODE.metr
from CODE.config import CONFIG
from CODE.blocks import Encoder, Predictor
from CODE.ar_branch import ARModel
from CODE.parcnet_loss import MultiResolutionSTFTLoss

'''import metr
from config import CONFIG
from blocks import Encoder, Predictor
from ar_branch import ARModel
from parcnet_loss import MultiResolutionSTFTLoss'''


class PLCModel(pl.LightningModule):
    def __init__(self, train_dataset=None, val_dataset=None, window_size=960, enc_layers=4, enc_in_dim=384, enc_dim=768,
                 pred_dim=512, pred_layers=1, pred_ckpt_path=None):
        super(PLCModel, self).__init__()
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

        self.enc_layers = enc_layers
        self.enc_in_dim = enc_in_dim
        self.enc_dim = enc_dim
        self.pred_dim = pred_dim  # NON CREDO ABBIA NULLA A CHE FARE CON I PACCHETTI
        self.pred_layers = pred_layers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.stoi = STOI(48000)
        self.pesq = PESQ(16000, 'wb')
        self.lmbda = 100.0

        self.predictor = Predictor(window_size=self.window_size, lstm_dim=self.pred_dim,
                                   lstm_layers=self.pred_layers)

        self.joiner = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(9, 1), stride=1, padding=(4, 0), padding_mode='reflect',
                      groups=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(48, 2, kernel_size=1, stride=1, padding=0, groups=2),
        )

        self.encoder = Encoder(in_dim=self.window_size, dim=self.enc_in_dim, depth=self.enc_layers,
                               mlp_dim=self.enc_dim)

        self.window = torch.sqrt(torch.hann_window(self.window_size))   # .to('cuda:0')  # <----------
        self.save_hyperparameters('window_size', 'enc_layers', 'enc_in_dim', 'enc_dim', 'pred_dim', 'pred_layers')
        self.mse_loss = F.mse_loss
        self.stft_loss = MultiResolutionSTFTLoss()

    def forward(self, x):
        """
        Input: real-imaginary; shape (B, F, T, 2); F = hop_size + 1
        Output: real-imaginary
        """

        B, C, F, T = x.shape

        x = x.permute(3, 0, 1, 2).unsqueeze(-1)
        prev_mag = torch.zeros((B, 1, F, 1), device=x.device)
        predictor_state = torch.zeros((2, self.predictor.lstm_layers, B, self.predictor.lstm_dim), device=x.device)
        mlp_state = torch.zeros((self.encoder.depth, 2, 1, B, self.encoder.dim), device=x.device)
        result = []
        for step in x:
            feat, mlp_state = self.encoder(step, mlp_state)
            prev_mag, predictor_state = self.predictor(prev_mag, predictor_state)
            feat = torch.cat((feat, prev_mag), 1)
            feat = self.joiner(feat)
            feat = feat + step
            result.append(feat)
            prev_mag = torch.linalg.norm(feat, dim=1, ord=1, keepdims=True)
        output = torch.cat(result, -1)

        return output

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=False, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, persistent_workers=True)

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

        for i in range(len(target)):
            prediction = self.ar_model.predict(ar_past[i], (self.p_size + self.fadeout + self.padding))
            tmp = np.concatenate((ar_past[i], prediction), axis=None)
            ar_pred.append(tmp)

        ar_pred = torch.tensor(np.array(ar_pred))    
        pred = nn_pred.to('cuda:0') + ar_pred.to('cuda:0')

        pred = pred.to(torch.double)     # per metrics
        target = target.to(torch.double)
        
        mse_loss = self.mse_loss(pred, target)  

        sc_loss, log_loss = self.stft_loss(y_pred=pred.squeeze(1), y_true=target.squeeze(1))  # NB
        spectral_loss = 0.5 * (sc_loss + log_loss)  
        tot_loss = self.lmbda * mse_loss + spectral_loss  

        self.log('mse_loss', mse_loss, prog_bar=True)
        self.log('sc_loss', sc_loss, prog_bar=False)
        self.log('log_mag_loss', log_loss, prog_bar=False)
        self.log('spectral_loss', spectral_loss, prog_bar=True)
        self.log('tot_loss', tot_loss, prog_bar=True)

        return tot_loss

    def validation_step(self, val_batch, batch_idx):
        ar_past, nn_past, target = val_batch

        f_0 = nn_past[:, :, 0:1, :]
        nn_input = nn_past[:, :, 1:, :]   # torch.Size([20, 2, 480, 7]) 
        nn_pred = self(nn_input)          # torch.Size([20, 2, 480, 7])
        nn_pred = torch.cat([f_0, nn_pred], dim=2)   # torch.Size([20, 2, 481, 7])
        nn_pred = torch.view_as_complex(nn_pred.permute(0, 2, 3, 1).contiguous())
        nn_pred = torch.istft(nn_pred, self.window_size, self.hop_size, window=self.window) # torch.Size([20, 2880])

        ar_pred = []
        ar_past = ar_past.cpu().numpy()

        for i in range(len(target)):
            prediction = self.ar_model.predict(ar_past[i], (self.p_size + self.fadeout + self.padding))   # 2240 + 640
            tmp = np.concatenate((ar_past[i], prediction), axis=None)
            ar_pred.append(tmp)

        ar_pred = torch.tensor(np.array(ar_pred))
        
        pred = nn_pred.to('cuda:0') + ar_pred.to('cuda:0')  

        val_loss = metr.nmse(y_pred=pred, y_true=target)  # 2880
        packet_val_loss = metr.nmse(y_pred=pred[..., -(self.p_size + self.fadeout + self.padding):], y_true=target[..., -(self.p_size + self.fadeout + self.padding):])   # ultimi 640

        self.log('val_loss', val_loss)
        self.log('packet_val_loss', packet_val_loss)

        return val_loss, packet_val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=CONFIG.TRAIN.patience,
                                                                  factor=CONFIG.TRAIN.factor, verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
