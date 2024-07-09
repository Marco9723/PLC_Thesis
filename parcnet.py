import torch
import sys
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from ar_branch import ARModel
from train.neural_branch import PLCModel
from train.neural_branch import PLCModel
from metrics import _melspectrogram, mel_spectral_convergence, nmse
from config import CONFIG                                  


class PARCnet:

    def __init__(self,
                 packet_dim: int,     
                 extra_pred_dim: int,  
                 padding: int,         
                 ar_order: int,       
                 ar_diagonal_load: float, 
                 ar_context_dim: int,  
                 nn_context_dim: int,   
                 nn_fade_dim: int,     
                 model_checkpoint: str,
                 device: str = 'cpu',
                 lite: bool = True,
                 ):

        self.packet_dim = packet_dim    
        self.extra_dim = extra_pred_dim  
        self.padding = padding           

        # Define the prediction length, including the extra length
        self.pred_dim = packet_dim + extra_pred_dim  

        # Define the AR and neural network contexts in sample
        self.ar_context_dim = ar_context_dim * packet_dim     
        self.nn_context_dim = nn_context_dim * packet_dim    

        # Define fade-in modulation vector (neural network contribution only)
        self.nn_fade_dim = nn_fade_dim               
        self.nn_fade = np.linspace(0, 1, nn_fade_dim)  

        # Define fade-in and fade-out modulation vectors
        self.fade_in = np.linspace(0., 1., extra_pred_dim)  
        self.fade_out = np.linspace(1., 0., extra_pred_dim)

        # Instantiate the linear predictor
        self.ar_model = ARModel(ar_order, ar_diagonal_load)

        # Load the pretrained neural network
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
                ar_context = np.pad(ar_context, (self.ar_context_dim - len(ar_context), 0))  
                ar_pred = self.ar_model.predict(valid=ar_context, steps=self.pred_dim)
                
                # NN model context
                nn_context = output_signal[idx - self.nn_context_dim: idx]
                # print(len(nn_context))
                nn_context = np.pad(nn_context, (self.nn_context_dim - len(nn_context), self.pred_dim + self.padding)) 
                # nn_context = np.pad(nn_context, (0, self.pred_dim + self.padding))   
                # nn_context = torch.Tensor(nn_context[None, None, ...])   

                nn_context = torch.tensor(nn_context)  
                nn_context = torch.stft(nn_context, 960, 480, window=self.hann, return_complex=False).permute(2, 0, 1).float()
                nn_context = nn_context.unsqueeze(0)  

                with torch.no_grad():
                    # NN model inference
                    f_0 = nn_context[:, :, 0:1, :]
                    nn_input = nn_context[:, :, 1:, :]
                    nn_pred = self.neural_net(nn_input)   
                    nn_pred = torch.cat([f_0, nn_pred], dim=2)  
                    nn_pred = torch.view_as_complex(nn_pred.permute(0, 2, 3, 1).contiguous())
                    nn_pred = torch.istft(nn_pred, 960, 480,
                                          window=self.hann)  
                    nn_pred = nn_pred[..., -(self.pred_dim + self.padding):]  
                    nn_pred = nn_pred[..., :self.pred_dim]  # primi 400 (320 + 80)   
                    nn_pred = nn_pred.squeeze().cpu().numpy()  # [400]   

                # Apply fade-in to the neural network contribution (inbound fade-in)
                nn_pred[:self.nn_fade_dim] *= self.nn_fade

                # Combine the two predictions
                prediction = ar_pred + nn_pred

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
