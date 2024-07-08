import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from parcnet import PARCnet
from utils import simulate_packet_loss
from config import CONFIG
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
import torch
import sys
sys.path.append("C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI")
from CODE.PLCmos.plc_mos import PLCMOSEstimator
import numpy as np
from utils import LSD
from speechmos import plcmos

plc_mos = PLCMOSEstimator()

def main():

    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/lightning_logs/version_271/checkpoints/frn-epoch=126-val_loss=-8.9949.ckpt" POCO ZIP
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/fine_tuning/lightning_logs/version_25/checkpoints/format-parcnet-epoch=399-val_loss=-10.3031.ckpt" ZIP
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/lightning_logs/version_277/checkpoints/frn-epoch=66-val_loss=-8.5957.ckpt"
    model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/train_and_fine_tuning/lightning_logs/version_249/checkpoints/parcnet-epoch=184-val_loss=-10.8378.ckpt" # SMOOTH
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/train_and_fine_tuning/lightning_logs/version_45/checkpoints/parcnet-epoch=64-val_loss=-10.4818.ckpt" ZIP
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/train_and_fine_tuning/lightning_logs/version_32/checkpoints/frn-epoch=48-val_loss=-11.6293.ckpt" ZIP
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/train_and_fine_tuning/lightning_logs/version_42/checkpoints/parcnet-epoch=64-val_loss=-11.0226.ckpt" ZIP
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/train_and_fine_tuning/lightning_logs/version_249/checkpoints/parcnet-epoch=08-val_loss=-10.6032.ckpt"
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/train_and_fine_tuning/lightning_logs/version_251/checkpoints/parcnet-epoch=23-val_loss=-6.6515.ckpt" # ABBASTANZA SMOOTH, POCO ZIP
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/train_and_fine_tuning/lightning_logs/version_259/checkpoints/parcnet-epoch=01-val_loss=-13.9202.ckpt"
    # model_checkpoint = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/train_and_fine_tuning/lightning_logs/version_263/checkpoints/parcnet-epoch=60-val_loss=-14.6845.ckpt"
    
    audio_test_folder = Path("C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/data/val_clean_v2")
    trace_folder = Path("C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/data/traces_v2")
    output_folder = "C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/long_training_"
    
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

    # Instantiate PARCnet
    parcnet = PARCnet(
        packet_dim=packet_dim,                      # 320
        fadeout = fadeout,                          # 80
        padding = padding,                          # 240
        ar_order=ar_order,                          # 128 
        ar_diagonal_load=diagonal_load,             # 0.001
        num_valid_ar_packets=num_valid_ar_packets,  # 7    
        num_valid_nn_packets=num_valid_nn_packets,  # 7
        model_checkpoint=model_checkpoint,
        xfade_len_in=xfade_len_in,                  # 16
        device='cpu'
    )

    # file 23 {'Intrusive': 1.6867953538894653, 'Non-intrusive': 1.6867953538894653, 'LSD': 2.100722, 'STOI': tensor(0.8870), 'PESQ': tensor(1.5858), 'SPEECHMOS': 2.4181840737660725}
    file_name = 'plcchallenge2024_val_.wav'
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
    # y_ref = librosa.util.normalize(y_ref)
    # with sf.SoundFile(file_path) as f:
    #     sr = f.samplerate
    #     y_ref = f.read(always_2d=True, dtype="float32")
    #     y_ref = y_ref.reshape(-1).astype(np.float32)
    # print(len(y_ref), len(trace))
        
    # Simulate packet losses 
    y_lost = simulate_packet_loss(y_ref, trace, packet_dim)   # packet dim = 320, il + 80 fatto dopo con fadeout (prima era extra_dim)
    # Predict using PARCnet
    y_pred = parcnet(y_lost, trace)
    #y_pred = y_lost  # se vuoi fare lower bound
    
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
    
    # print('stoi:', stoi, 'lsd:', lsd, 'plcmos:', ret, 'speechmos:', plcmos_v2_value, 'pesq:', pesq)
    print( 'plcmos:', ret, 'speechmos:', plcmos_v2_value, 'pesq:', pesq)

    y_pred_save = y_pred.copy()
    # normalized_array = np.clip(y_pred_save, -1, 1)
    output_path = f'{output_folder}audio_{file_name}.wav'
    # sf.write(output_path, normalized_array, 48000)
    sf.write(output_path, y_pred_save, 48000)
    

if __name__ == "__main__":
    main()