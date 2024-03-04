import librosa
from speechmos import plcmos 
#from config import CONFIG
import os

'''def load_txt(txt_list):
    target = []
    with open(txt_list) as f:
        for line in f:
            target.append(os.path.join('C:/Users/marco/Documents/GitHub/FNR/data/', line.strip('\n')))
    target = list(set(target))
    target.sort()
    return target'''

def load_file_list(file_path):
    file_list = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Rimuove spazi bianchi e newline dalla fine e aggiunge il nome del file alla lista
                file_list.append(line.strip())
        #print(file_list)
    except FileNotFoundError:
        print(f"Il file '{file_path}' non è stato trovato.")
    
    return file_list

def calculate_speechmos_for_files(file_list):
    for file_path in file_list:
        path=os.path.join("C:/Users/marco/Documents/GitHub/FNR/data/plcchallenge/", file_path)
        #print(path)
        plcmos_v2=plcmos.run(path, sr=16000)  
        print(plcmos_v2) 
        #print(f"File: {path} - Speechmos: {plcmos_v2:.2f} ")   #<-----

#data_list contiene cose tipo: 'plcchallenge2024_val/val_clean/plcchallenge2024_val_0758.wav'         
path_to_txt='C:/Users/marco/Documents/GitHub/FNR/data/test.txt'
data_list = load_file_list(path_to_txt)
calculate_speechmos_for_files(data_list)

#Lui funziona:
#plc_mos=plcmos.run("C:/Users/marco/Documents/GitHub/FNR/data/plcchallenge/plcchallenge2024_val/val_lossy/plcchallenge2024_val_0000.wav", sr=16000) 
#print(plc_mos)

        
