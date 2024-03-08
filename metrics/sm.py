import librosa
from speechmos import plcmos 
import os

def load_file_list(file_path):
    file_list = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                file_list.append(line.strip())
        #print(file_list)
    except FileNotFoundError:
        print(f"Il file '{file_path}' non è stato trovato.")
    
    return file_list

def calculate_speechmos_for_files(file_list):
    for file_path in file_list:
        path=os.path.join("C:/Users/marco/Documents/GitHub/FNR/data/plcchallenge/", file_path)
        plcmos_v2=plcmos.run(path, sr=16000)  
        print('Speechmos':,plcmos_v2) 

# 'plcchallenge2024_val/val_clean/plcchallenge2024_val_0758.wav'         
path_to_txt='C:/Users/marco/Documents/GitHub/FNR/data/test.txt'
data_list = load_file_list(path_to_txt)
calculate_speechmos_for_files(data_list)


        
