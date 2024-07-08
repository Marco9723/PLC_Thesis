import os
import librosa
import numpy as np
from pathlib import Path
from new_parcnet import PARCnet
from utils import simulate_packet_loss
from config import CONFIG
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
from utils import LSD
import torch
import sys
from speechmos import plcmos
from PLCmos.plc_mos import PLCMOSEstimator
import numpy as np

plc_mos = PLCMOSEstimator()

def main():

    # MIGLIOR FINE TUNING (somma errata), RIFORMATTATO CON FORMAT.PY, AR=NN=7, AR_ORDER=128:

    # model_checkpoint = ".../lightning_logs/version_249/checkpoints/parcnet-epoch=184-val_loss=-10.8378.ckpt"   # AR 128
    # model_checkpoint = ".../lightning_logs/version_267/checkpoints/parcnet-epoch=168-val_loss=-10.6563.ckpt"   # AR 1024

    audio_test_folder = Path("path/to/audio/folder")
    trace_folder = Path("path/to/trace/folder")

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
    
    stoi_metric = STOI(48000)
    pesq_metric = PESQ(16000, 'wb')   

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

    audio_format = "plcchallenge2024_val_*.wav"
    
    intrusive_list = []
    non_intrusive_list = []
    lsd_list = []
    stoi_list = []
    pesq_list = []
    speechmos_list = []
    index = 0
    
    for audio_test_path in audio_test_folder.glob(audio_format): 
        # if index < 497:
        #     continue
        file = os.path.basename(audio_test_path)
        file_name = os.path.splitext(file)[0]
        
        print('File_name:', file_name)
        
        # Load packet loss trace
        trace_path = trace_folder.joinpath(f"{file_name}.txt")

        # print('File_name:', file_name, trace_path)
        trace = np.loadtxt(trace_path)
        trace = np.repeat(trace, 3)

        # Read audio file
        y_ref, sr = librosa.load(audio_test_path, sr=sr, mono=True)
        # y_ref = librosa.util.normalize(y_ref)

        # Simulate packet losses
        y_lost = simulate_packet_loss(y_ref, trace, packet_dim)  

        # Predict using PARCnet
        y_pred = parcnet(y_lost, trace)  # parcnet
        # y_pred = y_lost                # zero  fill
        # y_pred = y_ref                 # upper bound

        y_pred = torch.tensor(y_pred)
        y_ref = torch.tensor(y_ref)

        # Compute STOI
        stoi = stoi_metric(y_pred, y_ref)

        # Compute LSD
        y_pred = y_pred.cpu().numpy()
        y_ref = y_ref.cpu().numpy()
        lsd, _ = LSD(y_ref, y_pred)

        # Resample for Plcmos, Speechmos, PESQ
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

        metrics = {
            "Intrusive": ret,
            "Non-intrusive": ret,
            'LSD': lsd,
            'STOI': stoi,
            'PESQ': pesq,
            'SPEECHMOS': plcmos_v2_value,
        }

        print(metrics)

        intrusive_list.append(ret)
        non_intrusive_list.append(ret)
        lsd_list.append(lsd)
        stoi_list.append(stoi)
        pesq_list.append(pesq)
        speechmos_list.append(plcmos_v2_value)

        if index % 100 == 0:
            mean = sum(intrusive_list) / len(intrusive_list) if len(intrusive_list) > 0 else 0
            print('PLCMOS:', mean)

        index = index + 1


    intrusive_mean = sum(intrusive_list) / len(intrusive_list) if len(intrusive_list) > 0 else 0
    lsd_mean = sum(lsd_list) / len(lsd_list) if len(lsd_list) > 0 else 0
    stoi_mean = sum(stoi_list) / len(stoi_list) if len(stoi_list) > 0 else 0
    pesq_mean = sum(pesq_list) / len(pesq_list) if len(pesq_list) > 0 else 0
    speechmos_mean = sum(speechmos_list) / len(speechmos_list) if len(speechmos_list) > 0 else 0

    intrusive_stdev = np.std(intrusive_list)
    lsd_stdev = np.std(lsd_list)
    stoi_stdev = np.std(stoi_list)
    pesq_stdev = np.std(pesq_list)
    speechmos_stdev = np.std(speechmos_list)

    print('plcmos_mean:',intrusive_mean, "intrusive_stdev:", intrusive_stdev)
    print('lsd_mean:', lsd_mean, "lsd_stdev:", lsd_stdev)
    print('stoi_mean:', stoi_mean, "stoi_stdev:", stoi_stdev)
    print('pesq_mean:', pesq_mean, "pesq_stdev:", pesq_stdev)
    print('speechmos_mean:', speechmos_mean, "speechmos_stdev:", speechmos_stdev)
    
if __name__ == "__main__":
    main()
