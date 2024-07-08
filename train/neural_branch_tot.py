import pytorch_lightning as pl
import torch
import sys
from torch import nn
import torch.nn.functional as F
import numpy as np
import metrics
from CODE.config import CONFIG
from CODE.blocks import Encoder, Predictor
from CODE.ar_branch import ARModel
from CODE.parcnet_loss import MultiResolutionSTFTLoss

class PLCModel(pl.LightningModule):
    def __init__(self, train_dataset=None, val_dataset=None, window_size=960, enc_layers=4, enc_in_dim=384, enc_dim=768,
                 pred_dim=512, pred_layers=1):
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
        self.val_predictions = []
        self.train_predictions = []
        self.enc_layers = enc_layers
        self.enc_in_dim = enc_in_dim
        self.enc_dim = enc_dim
        self.pred_dim = pred_dim
        self.pred_layers = pred_layers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lmbda = 100.0
        self.epsilon = 0.0

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

        self.window = torch.sqrt(torch.hann_window(self.window_size))  #.to('cuda:0')
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

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.val_dataset.previous_predictions = None
            self.train_dataset.previous_predictions = None

    def training_step(self, batch, batch_idx):
        ar_past, nn_past, target = batch

        use_teacher_forcing = torch.rand(1) < self.epsilon

        f_0 = nn_past[:, :, 0:1, :]
        nn_input = nn_past[:, :, 1:, :]
        nn_pred = self(nn_input)
        nn_pred = torch.cat([f_0, nn_pred], dim=2)
        nn_pred = torch.view_as_complex(nn_pred.permute(0, 2, 3, 1).contiguous())
        nn_pred = torch.istft(nn_pred, self.window_size, self.hop_size, window=self.window)

        ar_pred = []
        ar_past = ar_past.cpu().numpy()

        for i in range(len(target)):
            prediction = self.ar_model.predict(ar_past[i], (self.p_size + self.fadeout + self.padding)).astype(np.float32)
            tmp = np.concatenate((ar_past[i], prediction), axis=None)
            ar_pred.append(tmp)

        ar_pred = torch.tensor(np.array(ar_pred))

        pred = ar_pred.to('cuda:0') + nn_pred.to('cuda:0')

        if use_teacher_forcing:
            output = pred[:, 7 * self.p_size: 8 * self.p_size]
        else:
            output = target[:, 7 * self.p_size: 8 * self.p_size]

        self.train_predictions.append(output)

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
        if self.current_epoch % self.train_dataset.packets_for_signal == 0:
            self.train_dataset.previous_predictions = torch.cat(self.train_predictions, dim=0).detach().cpu().numpy()
        elif self.current_epoch % self.train_dataset.packets_for_signal == self.train_dataset.packets_for_signal - 1:
            print(
                f'Epoca {self.current_epoch}, Train dataset:{self.train_dataset.previous_predictions.shape[0]}, Train pred length:{self.train_dataset.previous_predictions.shape[1]}')
            self.train_dataset.previous_predictions = None
            self.train_dataset.audio_index = torch.randint(0, CONFIG.DATA.audio_chunk_len - self.p_size*17, (1,))[0]
        else:
            self.train_dataset.previous_predictions = np.concatenate((self.train_dataset.previous_predictions, torch.cat(self.train_predictions, dim=0).detach().cpu().numpy()), axis=1)
            print(
                f'Epoca {self.current_epoch}, Train dataset:{self.train_dataset.previous_predictions.shape[0]}, Train pred length:{self.train_dataset.previous_predictions.shape[1]}')

        self.train_predictions.clear()
        self.train_predictions = []

    def validation_step(self, val_batch, batch_idx):
        ar_past, nn_past, target = val_batch

        use_teacher_forcing = torch.rand(1) < self.epsilon

        f_0 = nn_past[:, :, 0:1, :]
        nn_input = nn_past[:, :, 1:, :]
        nn_pred = self(nn_input)
        nn_pred = torch.cat([f_0, nn_pred], dim=2)
        nn_pred = torch.view_as_complex(nn_pred.permute(0, 2, 3, 1).contiguous())
        nn_pred = torch.istft(nn_pred, self.window_size, self.hop_size, window=self.window)

        ar_pred = []
        ar_past = ar_past.cpu().numpy()

        for i in range(len(target)):
            prediction = self.ar_model.predict(ar_past[i], (self.p_size + self.fadeout + self.padding)).astype(np.float32)
            tmp = np.concatenate((ar_past[i], prediction), axis=None)
            ar_pred.append(tmp)

        ar_pred = torch.tensor(np.array(ar_pred))
        pred = ar_pred.to('cuda:0') + nn_pred.to('cuda:0')

        if use_teacher_forcing:
            output = pred[:, 7 * self.p_size: 8 * self.p_size]
        else:
            output = target[:, 7 * self.p_size: 8 * self.p_size]

        self.val_predictions.append(output)

        val_loss = metr.nmse(y_pred=pred, y_true=target)  
        packet_val_loss = metr.nmse(y_pred=pred[..., -(self.p_size + self.fadeout + self.padding):],
                                    y_true=target[..., -(self.p_size + self.fadeout + self.padding):])

        self.log('val_loss', val_loss)
        self.log('packet_val_loss', packet_val_loss)

        return val_loss, packet_val_loss

    def on_validation_epoch_end(self):
        if self.current_epoch % self.val_dataset.packets_for_signal == 0:
            self.val_dataset.previous_predictions = torch.cat(self.val_predictions, dim=0).detach().cpu().numpy()
        elif self.current_epoch % self.val_dataset.packets_for_signal == self.val_dataset.packets_for_signal - 1:
            print(
                f'Epoca {self.current_epoch}, Val dataset:{self.val_dataset.previous_predictions.shape[0]}, Val pred length:{self.val_dataset.previous_predictions.shape[1]}')
            self.val_dataset.previous_predictions = None
            self.val_dataset.audio_index = torch.randint(0, CONFIG.DATA.audio_chunk_len - self.p_size * 17, (1,))[0]
            self.epsilon = min(self.epsilon + (self.val_dataset.packets_for_signal / CONFIG.TRAIN.epochs), 1.0)
        else:
            self.val_dataset.previous_predictions = np.concatenate((self.val_dataset.previous_predictions, torch.cat(self.val_predictions, dim=0).detach().cpu().numpy()), axis=1)
            print(
                f'Epoca {self.current_epoch}, Val dataset:{self.val_dataset.previous_predictions.shape[0]}, Val pred length:{self.val_dataset.previous_predictions.shape[1]}')

        print(
            f'Epoca {self.current_epoch}, Packet Loss: {self.trainer.callback_metrics["packet_val_loss"]}, Val Loss: {self.trainer.callback_metrics["val_loss"]}')

        self.val_predictions.clear()
        self.val_predictions = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=CONFIG.TRAIN.patience,
                                                                  factor=CONFIG.TRAIN.factor, verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }

        optimizer_config = {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

        return [optimizer_config]
