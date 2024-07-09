import os
import librosa
import numpy as np
import soundfile as sf
import torch
import sys
import numpy as np
from pathlib import Path
from parcnet import PARCnet
from utils import simulate_packet_loss
from config import CONFIG
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
from PLCmos.plc_mos import PLCMOSEstimator
from utils import LSD
from speechmos import plcmos

plc_mos = PLCMOSEstimator()

def main():

    model_checkpoint = ".../parcnet-epoch=184-val_loss=-10.8378.ckpt"   # AR 128
    # model_checkpoint = ".../parcnet-epoch=120-val_loss=-10.3744.ckpt"   # AR 256
    # model_checkpoint = ".../parcnet-epoch=168-val_loss=-10.5687.ckpt"   # AR 512
    # model_checkpoint = ".../parcnet-epoch=168-val_loss=-10.6563.ckpt"   # AR 1024
    
    audio_test_folder = Path("path/to/data/val_clean")
    trace_folder = Path("path/to/data/traces")
    output_folder = ".../parcnet_ar_128_"
    
    # Read global params from config file
    sr = CONFIG.DATA.sr                       
    packet_dim = CONFIG.DATA.EVAL.packet_size 
    fadeout = CONFIG.DATA.EVAL.fadeout         
    padding = CONFIG.DATA.EVAL.padding      

    # Read AR params from config file
    ar_order = CONFIG.AR_MODEL.ar_order                         
    diagonal_load = CONFIG.AR_MODEL.diagonal_load                
    num_valid_ar_packets = CONFIG.AR_MODEL.num_valid_ar_packets   

    # Read NN params from config file
    num_valid_nn_packets = CONFIG.NN_MODEL.num_valid_nn_packets  
    xfade_len_in = CONFIG.NN_MODEL.xfade_len_in                  

    # Instantiate PARCnet
    parcnet = PARCnet(
        packet_dim=packet_dim,                    
        extra_pred_dim = fadeout,                 
        padding = padding,                        
        ar_order=ar_order,                        
        ar_diagonal_load=diagonal_load,          
        ar_context_dim=num_valid_ar_packets,     
        nn_context_dim=num_valid_nn_packets, 
        model_checkpoint=model_checkpoint,
        nn_fade_dim=xfade_len_in,                
        device='cpu'
    )

    file_name = 'plcchallenge2024_val_FILEID.wav'
    print('File_name:', file_name)
    file_path = os.path.join(audio_test_folder, file_name)
    file_name= os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    
    # Load packet loss trace
    trace_path = trace_folder.joinpath(f"{file_name}.txt")
    trace = np.loadtxt(trace_path)
    trace = np.repeat(trace, 3)

    # Read audio file
    y_ref, sr = librosa.load(file_path, sr=sr, mono=True)
        
    # Simulate packet losses 
    y_lost = simulate_packet_loss(y_ref, trace, packet_dim)   
    
    # Predict using PARCnet
    y_pred = parcnet(y_lost, trace)
    
    
    stoi_metric = STOI(48000)
    pesq_metric = PESQ(16000, 'wb')
    
    # Compute STOI
    # stoi = stoi_metric(y_pred, y_ref) 
        
    # Compute LSD
    y_pred = y_pred # .numpy()
    y_ref = y_ref   # .numpy()
    # lsd, _ = LSD(y_ref, y_pred) 
    
    if sr != 16000:    
        y_pred_res = librosa.resample(y_pred, orig_sr=48000, target_sr=16000) 
        y_ref_res = librosa.resample(y_ref, orig_sr=48000, target_sr=16000, res_type='kaiser_fast')
        
    # Plcmos
    ret = plc_mos.run(y_pred_res, y_ref_res)   
        
    # Speechmos
    plcmos_v2=plcmos.run(y_pred_res, sr=16000)
    plcmos_v2_value = plcmos_v2['plcmos']
        
    # PESQ
    pesq = pesq_metric(torch.tensor(y_pred_res), torch.tensor(y_ref_res))    
    
    print('stoi:', stoi, 'lsd:', lsd, 'plcmos:', ret, 'speechmos:', plcmos_v2_value, 'pesq:', pesq)

    y_pred_save = y_pred.copy()
    
    # normalized_array = np.clip(y_pred_save, -1, 1)
    output_path = f'{output_folder}audio_{file_name}.wav'
    
    # sf.write(output_path, normalized_array, 48000)
    sf.write(output_path, y_pred_save, 48000)
    

if __name__ == "__main__":
    main()
