import os
import librosa
import numpy as np
from pathlib import Path
from parcnet import PARCnet
from utils import simulate_packet_loss
from config import CONFIG
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
from utils import LSD
import torch
import sys
from speechmos import plcmos
sys.path.append("C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI")
from CODE.PLCmos.plc_mos import PLCMOSEstimator
import numpy as np

plc_mos = PLCMOSEstimator()

def main():

    # MIGLIOR FINE TUNING (somma errata), RIFORMATTATO CON FORMAT.PY, AR=NN=7, AR_ORDER=128:
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/fine_tuning/lightning_logs/version_86/frn-epoch=89-val_loss=-9.6633.ckpt"
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/fine_tuning/lightning_logs/version_69/checkpoints/format-parcnet-epoch=64-val_loss=-9.3397.ckpt"
    model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/fine_tuning/format_fine_tuned.ckpt"
    # MIGLIOR TRAINING, AR=NN=7, AR_ORDER=128:
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/lightning_logs/version_271/checkpoints/frn-epoch=126-val_loss=-8.9949.ckpt"
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/lightning_logs/version_277/checkpoints/frn-epoch=66-val_loss=-8.5957.ckpt"

    audio_test_folder = Path("C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/data/val_clean_v2")
    trace_folder = Path("C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/data/traces_v2")
    
    # Read global params from config file
    sr = CONFIG.DATA.sr                        # 48000
    packet_dim = CONFIG.DATA.EVAL.packet_size  # 320
    fadeout = CONFIG.DATA.EVAL.fadeout         # 80   
    padding = CONFIG.DATA.EVAL.padding         # 240

    # Read AR params from config file
    ar_order = CONFIG.AR_MODEL.ar_order                           # 128 
    diagonal_load = CONFIG.AR_MODEL.diagonal_load                 # 0.001
    num_valid_ar_packets = CONFIG.AR_MODEL.num_valid_ar_packets   # 10  

    # Read NN params from config file
    num_valid_nn_packets = CONFIG.NN_MODEL.num_valid_nn_packets   # 7
    xfade_len_in = CONFIG.NN_MODEL.xfade_len_in                   # 16   
    
    stoi_metric = STOI(48000)
    pesq_metric = PESQ(16000, 'wb')   

    # Instantiate PARCnet
    parcnet = PARCnet(
        packet_dim=packet_dim,                      # 320
        fadeout = fadeout,                          # 80
        padding = padding,                          # 240
        ar_order=ar_order,                          # 256 
        ar_diagonal_load=diagonal_load,             # 0.001
        num_valid_ar_packets=num_valid_ar_packets,  # 7    
        num_valid_nn_packets=num_valid_nn_packets,  # 7
        model_checkpoint=model_checkpoint,
        xfade_len_in=xfade_len_in,                  # 16
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
    
    # for index, audio_test_path in enumerate(audio_test_folder.glob(audio_format)):
    for audio_test_path in audio_test_folder.glob(audio_format): 
        # if index < 497:
        #     continue
        file = os.path.basename(audio_test_path)
        file_name = os.path.splitext(file)[0]
        
        print('File_name:', file_name)
        
        # Load packet loss trace
        trace_path = trace_folder.joinpath(f"{file_name}.txt")
        trace = np.loadtxt(trace_path)
        trace = np.repeat(trace, 3)

        # Read audio file
        y_ref, sr = librosa.load(audio_test_path, sr=sr, mono=True)

        # Simulate packet losses 
        y_lost = simulate_packet_loss(y_ref, trace, packet_dim)   # packet dim = 320, il + 80 fatto dopo con fadeout (prima era extra_dim)

        # Predict using PARCnet
        y_pred = parcnet(y_lost, trace)
        
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
    # non_intrusive_mean = sum(non_intrusive_list) / len(non_intrusive_list) if len(non_intrusive_list) > 0 else 0
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