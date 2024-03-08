import torch
import sys
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from ar_branch import ARModel
from neural_branch import PLCModel
from CODE.fine_tuning.ft_neural_branch import FinePLCModel 
from config import CONFIG 
sys.path.append("C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI")

class PARCnet:

    def __init__(self,
                 packet_dim: int,
                 fadeout: int,   # 80         
                 padding: int,   # 240
                 ar_order: int,
                 ar_diagonal_load: float,
                 num_valid_ar_packets: int,   # 7 packets
                 num_valid_nn_packets: int,
                 model_checkpoint: str,
                 xfade_len_in: int,
                 device: str = 'cpu',
                 ):

        self.packet_dim = packet_dim        # 320       
        self.f_out = fadeout                # 80
        self.extra_dim = padding            # 240 

        # Define the prediction length, including the extra length
        self.pred_dim = self.packet_dim + self.f_out    # 320 + 80

        # Define the AR and neural network contexts in sample
        num_valid_ar_packets = 7
        self.ar_context_len = num_valid_ar_packets * packet_dim
        self.nn_context_len = num_valid_nn_packets * packet_dim

        # Define fade-in modulation vector (neural network contribution only)
        self.fade_in = np.ones(self.pred_dim)   # 400 (320 + 80)
        self.fade_in[:xfade_len_in] = np.linspace(0, 1, xfade_len_in)

        # Define fade-out modulation vector
        self.fade_out = np.ones(self.pred_dim)  # 400 (320 + 80)
        self.fade_out[-(self.f_out):] = np.linspace(1, 0, self.f_out)  # 400

        # Instantiate the linear predictor
        self.ar_model = ARModel(ar_order, ar_diagonal_load)

        # Load the pretrained neural network
        self.neural_net = PLCModel.load_from_checkpoint(model_checkpoint, channels=1, lite=True).to(device) 
        
        self.hann = torch.sqrt(torch.hann_window(960)) 

    def __call__(self, input_signal: np.ndarray, trace: np.ndarray, **kwargs) -> np.ndarray:
        self.neural_net.eval()
        output_signal = deepcopy(input_signal)

        for i, loss in tqdm(enumerate(trace), total=len(trace)):
            if loss:
                # Start index of the ith packet
                idx = i * self.packet_dim   # 320

                # AR model context
                valid_ar_packets = output_signal[idx - self.ar_context_len:idx]   # 7 x 320

                # AR model inference
                # print(len(valid_ar_packets))  <--- 1440, 640, sempre 2240 in teoria
                ar_pred = self.ar_model.predict(valid=valid_ar_packets, steps=self.pred_dim)  # 400 

                # NN model context
                nn_context = output_signal[idx - self.nn_context_len: idx]    # 7 x 320 
                nn_context = np.pad(nn_context, (0, self.pred_dim + self.extra_dim))   # padding (400 + 240)
                nn_context = torch.tensor(nn_context)                                  # 2880       
                nn_context = torch.stft(nn_context, 960, 480, window=self.hann, return_complex=False).permute(2, 0, 1).float()
                nn_context = nn_context.unsqueeze(0)  
                
                with torch.no_grad():
                    # NN model inference
                    f_0 = nn_context[:, :, 0:1, :]   
                    nn_input = nn_context[:, :, 1:, :]  
                    nn_pred = self.neural_net(nn_input)   
                    nn_pred = torch.cat([f_0, nn_pred], dim=2)   
                    nn_pred = torch.view_as_complex(nn_pred.permute(0, 2, 3, 1).contiguous())
                    nn_pred = torch.istft(nn_pred, 960, 480, window=self.hann)  # 2240 + 320 + 320 = 2880  
                    nn_pred = nn_pred[..., -(self.pred_dim + self.extra_dim):]  # last 320 + 320 
                    nn_pred = nn_pred[..., :self.pred_dim]   # first 400 (320 + 80)   
                    nn_pred = nn_pred.squeeze().cpu().numpy()  # 400
                    
                # Apply fade-in to the neural network contribution (inbound fade-in)
                nn_pred *= self.fade_in  # 400

                # Combine the two predictions
                prediction = ar_pred + nn_pred  # 400
                # prediction = np.clip(prediction, -1e15, 1e15)

                # Cross-fade the compound prediction (outbound fade-out)
                prediction *= self.fade_out  # 400
                
                # Cross-fade the output signal (outbound fade-in)
                chunk_len = len(output_signal[idx:idx + self.pred_dim])  # 400
                output_signal[idx:idx + self.pred_dim] *= 1 - self.fade_out[:chunk_len]   
                
                # Conceal lost packet
                output_signal[idx: idx + self.pred_dim] += prediction[:chunk_len]

        return output_signal
