import librosa
# import numpy as np
from pathlib import Path
from config import CONFIG
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
from utils import visualize, LSD
import torch
import sys
from speechmos import plcmos
sys.path.append("C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI")
from CODE.PLCmos.plc_mos import PLCMOSEstimator

plc_mos = PLCMOSEstimator()

def main():

    # Read paths from config file
    audio_test_folder = Path("C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/data/val_clean_v2")
    audio_format = "plcchallenge2024_val_*.wav"
    sr = CONFIG.DATA.sr 

    stoi_metric = STOI(48000)
    pesq_metric = PESQ(16000, 'wb')

    intrusive_list = []
    non_intrusive_list = []
    lsd_list = []
    stoi_list = []
    pesq_list = []
    speechmos_list = [] 

    # sono già ordinati nella cartella
    for audio_test_path in audio_test_folder.glob(audio_format):
        
        print('File_id:',audio_test_path)

        # Read audio file
        y_ref, sr = librosa.load(audio_test_path, sr=sr, mono=True)
        y_pred = y_ref.copy()

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
            y_pred_res = librosa.resample(y_pred, orig_sr=48000, target_sr=16000)  # res_type????
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
        
    intrusive_mean = sum(intrusive_list) / len(intrusive_list) if len(intrusive_list) > 0 else 0
    non_intrusive_mean = sum(non_intrusive_list) / len(non_intrusive_list) if len(non_intrusive_list) > 0 else 0
    lsd_list_mean = sum(lsd_list) / len(lsd_list) if len(lsd_list) > 0 else 0
    stoi_list_mean = sum(stoi_list) / len(stoi_list) if len(stoi_list) > 0 else 0
    pesq_list_mean = sum(pesq_list) / len(pesq_list) if len(pesq_list) > 0 else 0
    speechmos_mean = sum(speechmos_list) / len(speechmos_list) if len(speechmos_list) > 0 else 0

    print('intrusive_mean:',intrusive_mean)
    print('non_intrusive_mean:', non_intrusive_mean)
    print('lsd_mean:', lsd_list_mean)
    print('stoi_mean:', stoi_list_mean)
    print('pesq_mean:', pesq_list_mean)
    print('speechmos_mean:', speechmos_mean)
    
if __name__ == "__main__":
    main()