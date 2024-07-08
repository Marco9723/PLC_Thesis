import torch
import sys
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from ar_branch import ARModel
from train_and_fine_tuning.neural_branch_tot import PLCModel
# sys.path.append("/nas/home/mviviani/tesi")
# from CODE.fine_tuning.ft_neural_branch import FinePLCModel  # <----------
from CODE.train_and_fine_tuning.neural_branch_tot import PLCModel
from config import CONFIG                                   # <----------
import metr


class PARCnet:

    def __init__(self,
                 packet_dim: int,      # 320
                 extra_pred_dim: int,  # 80 fadeout
                 padding: int,         # 240
                 ar_order: int,        # 128
                 ar_diagonal_load: float, #0.001
                 ar_context_dim: int,   #7
                 nn_context_dim: int,   #7
                 nn_fade_dim: int,      # xfade_len_in = 16
                 model_checkpoint: str,
                 device: str = 'cpu',
                 lite: bool = True,
                 ):

        self.packet_dim = packet_dim     # 320
        self.extra_dim = extra_pred_dim  # 80 per fadeout
        self.padding = padding           # 240

        # Define the prediction length, including the extra length
        self.pred_dim = packet_dim + extra_pred_dim  # 400

        # Define the AR and neural network contexts in sample
        self.ar_context_dim = ar_context_dim * packet_dim     # 7 * 320
        self.nn_context_dim = nn_context_dim * packet_dim     # 7 * 320

        # Define fade-in modulation vector (neural network contribution only)
        self.nn_fade_dim = nn_fade_dim                  # 16/24
        self.nn_fade = np.linspace(0, 1, nn_fade_dim)   # 16/24

        # Define fade-in and fade-out modulation vectors
        self.fade_in = np.linspace(0., 1., extra_pred_dim)  # 80  NON 16? <-----
        self.fade_out = np.linspace(1., 0., extra_pred_dim) # 80

        # Instantiate the linear predictor
        self.ar_model = ARModel(ar_order, ar_diagonal_load)

        # Load the pretrained neural network
        # -----> self.neural_net = PLCModel.load_from_checkpoint(model_checkpoint, channels=1, lite=lite, packet_dim=packet_dim, extra_pred_dim=extra_pred_dim,).to(device)
        self.neural_net = PLCModel.load_from_checkpoint(model_checkpoint, channels=1, lite=True).to(device)
        self.hann = torch.sqrt(torch.hann_window(960))

    def __call__(self, input_signal: np.ndarray, trace: np.ndarray, **kwargs) -> np.ndarray:
        # Neural estimator in eval mode
        self.neural_net.eval()

        # Instantiate the output signal
        output_signal = deepcopy(input_signal)
        output_signal = np.pad(output_signal, (0, self.extra_dim))

        # Initialize a flag keeping track of consecutive packet losses
        is_burst = False

        for i, is_lost in tqdm(enumerate(trace), total=len(trace)):
            if is_lost:
                # Start index of the ith packet
                idx = i * self.packet_dim

                # AR model prediction
                ar_context = output_signal[idx - self.ar_context_dim:idx]
                # print(len(ar_context))
                ar_context = np.pad(ar_context, (self.ar_context_dim - len(ar_context), 0))  # 2240, se perde pacchetto troppo presto
                ar_pred = self.ar_model.predict(valid=ar_context, steps=self.pred_dim)
                
                # NN model context
                nn_context = output_signal[idx - self.nn_context_dim: idx]
                # print(len(nn_context))
                nn_context = np.pad(nn_context, (self.nn_context_dim - len(nn_context), self.pred_dim + self.padding))  # <--- prima per avere 7 di context, dopo per input nn
                # nn_context = np.pad(nn_context, (0, self.pred_dim + self.padding))   # <----- messo noi, poi paddi con zeri
                # nn_context = torch.Tensor(nn_context[None, None, ...])    # <-------------- credo si possa togliere

                nn_context = torch.tensor(nn_context)  # torch.Size([2880]) come in dataset
                nn_context = torch.stft(nn_context, 960, 480, window=self.hann, return_complex=False).permute(2, 0, 1).float()
                nn_context = nn_context.unsqueeze(0)  # aggiungi una dimensione, in teoria giust perchè sono 3 e aggiungo batch size all'inizio

                with torch.no_grad():
                    # NN model inference
                    f_0 = nn_context[:, :, 0:1, :]
                    nn_input = nn_context[:, :, 1:, :]
                    nn_pred = self.neural_net(nn_input)   # <---------------- ORIGINALE: nn_pred = self.neural_net(nn_context)
                    nn_pred = torch.cat([f_0, nn_pred], dim=2)  # torch.Size([20, 2, 481, 7])
                    nn_pred = torch.view_as_complex(nn_pred.permute(0, 2, 3, 1).contiguous())
                    nn_pred = torch.istft(nn_pred, 960, 480,
                                          window=self.hann)  # 2240 + 320 + 320 = 2880   torch.Size([1, 2880])
                    # print(len(nn_pred))
                    nn_pred = nn_pred[..., -(self.pred_dim + self.padding):]  # prendi ultimi 320 + 320
                    nn_pred = nn_pred[..., :self.pred_dim]  # primi 400 (320 + 80)   torch.Size([1, 400])  # <---------------- ORIGINALE
                    nn_pred = nn_pred.squeeze().cpu().numpy()  # [400]    # <---------------- ORIGINALE

                # Apply fade-in to the neural network contribution (inbound fade-in)
                nn_pred[:self.nn_fade_dim] *= self.nn_fade

                # Combine the two predictions
                prediction = ar_pred + nn_pred
                # print(len(prediction))
                # prediction = np.clip(prediction, -1e15, 1e15)

                # Cross-fade the compound prediction (outbound fade-out)
                prediction[-self.extra_dim:] *= self.fade_out

                if is_burst:
                    # Cross-fade the prediction in case of consecutive packet losses (inbound fade-in)
                    prediction[:self.extra_dim] *= self.fade_in

                # Cross-fade the output signal (outbound fade-in)
                output_signal[idx + self.packet_dim:idx + self.pred_dim] *= self.fade_in

                # Conceal lost packet
                output_signal[idx: idx + self.pred_dim] += prediction

                # Keep track of consecutive packet losses
                is_burst = True

            else:
                # Reset burst loss indicator
                is_burst = False

        output_signal = output_signal[:len(input_signal)]

        return output_signal