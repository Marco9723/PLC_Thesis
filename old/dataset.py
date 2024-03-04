import glob
import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch
from numpy.random import default_rng
from pydtmc import MarkovChain
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import CONFIG

np.random.seed(0)
rng = default_rng()

def load_audio(
        path,
        sample_rate: int = 16000,
        chunk_len=None,
):
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        audio_len = f.frames

        if chunk_len is not None and chunk_len < audio_len:
            start_index = torch.randint(0, audio_len - chunk_len, (1,))[0]

            frames = f._prepare_read(start_index, start_index + chunk_len, -1)
            audio = f.read(frames, always_2d=True, dtype="float32")

        else:
            audio = f.read(always_2d=True, dtype="float32")

    if sr != sample_rate:
        audio = librosa.resample(np.squeeze(audio), orig_sr=sr, target_sr=sample_rate)[:, np.newaxis]

    return audio.T


class TestLoader(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return len(self.data_list)

    def load_txt(self, txt_list):
        return 

    def __getitem__(self, index):
        return 


class BlindTestLoader(Dataset):
    def __init__(self, test_dir):
        return

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return 

# LA CLASSE TRAINDATASET CARICA E PREPARA I DATI DA PASSARE POI NEL MAIN train_dataset = TrainDataset('train')
# MA SOPRATTUTTO IN test_loader = DataLoader(testset, batch_size=1, num_workers=4) OPPURE def train_dataloader(self): IN NEURAL BRANCH

# self.previous_predictions = torch.tensor([[1, 2, 3],   vorrei fosse così
#                                           [4, 5, 6],
#                                           [7, 8, 9]])

class TrainDataset(Dataset):
    def __init__(self, mode='train'):                                                                                                   # forse devi mettere un previous_predictions qui e nel mein quando lo chiami
        dataset_name = CONFIG.DATA.dataset                                # nome del dataset
        self.target_root = CONFIG.DATA.data_dir[dataset_name]['root']     # nome della cartella root

        txt_list = CONFIG.DATA.data_dir[dataset_name]['train']            # setting del file txt con i nomi degli audio
        self.data_list = self.load_txt(txt_list)                          # recupera i files e li mette in data_list
        
        if mode == 'train':                                               # splitta il dataset in training set e validation set
            self.data_list, _ = train_test_split(self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)

        elif mode == 'val':
            _, self.data_list = train_test_split(self.val_list, CONFIG.TRAIN.val_split, random_state=0)

        self.p_sizes = CONFIG.DATA.TRAIN.packet_sizes                     # possibili dimensioni dei pacchetti: [256, 512, 768, 960, 1024, 1536]
        self.mode = mode                                                  # o 'train' o 'val'
        self.sr = CONFIG.DATA.sr                                          # sample rate (48 kHz)
        self.window = CONFIG.DATA.audio_chunk_len                         # lunghezza audio estratto in termini di samples: 30720
        self.stride = CONFIG.DATA.stride                                  # stride of the STFT operation: 480
        self.chunk_len = CONFIG.DATA.window_size                          # window size of the STFT operation, equivalent to packet size: 960
        self.hann = torch.sqrt(torch.hann_window(self.chunk_len))         # hanning window
        self.context_lenght = CONFIG.DATA.TRAIN.context_lenght
        # self.previous_predictions = None # None perchè prima in dataset  # solo una volta, max 10 pacchetti
        self.packet_sizes = None                                          # DEVE PERO' ESSERE ORDINATO !!!
        
    def __len__(self):
        return len(self.data_list)

    def load_txt(self, txt_list):
        target = []
        with open(txt_list) as f:
            for line in f:
                target.append(os.path.join(self.target_root, line.strip('\n')))
        target = list(set(target))
        target.sort()
        return target
    
    def convert_to_mono(stereo_signal):
        if stereo_signal.ndim == 2 and stereo_signal.shape[1] == 2:
            mono_signal = np.mean(stereo_signal, axis=1)
            return mono_signal
        else:
            return stereo_signal

    def fetch_audio(self, index):
        sig = load_audio(self.data_list[index], sample_rate=self.sr, chunk_len=self.window) # ricava i files audio come numpy array
        while sig.shape[1] < self.window:                                  
            idx = torch.randint(0, len(self.data_list), (1,))[0]          
            pad_len = self.window - sig.shape[1]                           
                                                                           
            if pad_len < 0.02 * self.sr:                                   # se pad_len è minore di 960 samples (considerato trascurabile) metti soltanto zeri
                padding = np.zeros((1, pad_len), dtype=np.float)           
            else:                                                         
                padding = load_audio(self.data_list[idx], sample_rate=self.sr, chunk_len=pad_len)   # fai padding con un altro segmento audio
            sig = np.hstack((sig, padding))                                
        return sig

    def __getitem__(self, index):
                                                                                                # vd questione + 80, dimensioni da non sforare, controlli, channels (aggiungi unsqueeze lungo dimensione desiderata), assicurati che self.fetch_audio(index) sia nello stesso ordine! (in teoria con shuffle=false dovrebbe caricare tutto allo stesso modo), vedi se serve squeeze o unsqueeze                                                                                      
        target = self.fetch_audio(index)         # (num_channels, num_samples) + NumPy                                                                    
        target = self.convert_to_mono(target)    # (num_samples,) + NumPy                             
        target = target.astype(np.float32)       # (num_samples,) + astype + NumPy                                                                                        
        # target = target.reshape(-1).astype(np.float32)                                      
        # con reshape sarebbe stato  (num_samples*numchannels,) + astype
        
        if self.previous_predictions is None:                                                   
            p_size = random.choice(self.p_sizes)                                                
            self.packet_sizes = torch.cat(p_size, dim=1)  # torch tensor                                      
            target = target[:(self.context_lenght+1)*p_size]  # (num_samples,) + astype + NumPy    # primi 8           
            nn_input = target.copy()  # (num_samples,) + astype + NumPy 
            nn_input[-p_size:] = 0.0   # (num_samples,) + astype + NumPy dovrebbe essere così       # 7 + 1 a zero                                                  
            ar_input = target[:-p_size]    # (num_samples,) + astype + NumPy          non dovrebbe esserci differenza tra (num_samples,) e (num_samples)                                      
            target = target[-p_size:]      # (num_samples,) + astype + NumPy          array_2d_column = array_1d.reshape((3, 1)) + al contrario array_1d_again = array_2d_column.reshape((3,))                                          
            
            target = torch.tensor(target.copy())                                           
            nn_input = torch.tensor(nn_input.copy())   # dovrebbe essere corretto, anche frn fa così
            nn_input = torch.stft(nn_input, self.chunk_len, self.stride, window=self.hann, return_complex=False)
            nn_input = nn_input.permute(2, 0, 1).float()       
            # nn_input = (num_frames, num_bins, num_channels)   7+1    anche frn
            # p_size tensore con dimensioni pacchetti
            # ar_input se mono (num_samples,) + astype + NumPy  7     (batch_size, num_samples)  
            # target = tensore 1D (num_samples,) 1 , convertito come frn      (batch_size, num_samples) 
            
        elif self.previous_predictions is not None:                                             
            p_size = self.packet_sizes[index] 
            target = target[len(self.previous_predictions[index,:]) : len(self.previous_predictions[index,:])+(self.context_lenght+1)*p_size]  # nuovo target da 8, array (num_samples,) + astype + NumPy  # l'estremo sx è escluso
            nn_input = target.copy()    # array (num_samples,) + astype + NumPy                                                                
            tmp = torch.cat([self.previous_predictions[index,:], torch.zeros(p_size)]) # tensore (num_samples) la concatenazione dovrebbe essere giusta, torche zeroes dovrebbe essere float
            tmp= tmp.numpy()
            tmp = tmp.reshape(-1).astype(np.float32)  # cosi target e tmp sono  array (num_samples,) + astype + NumPy
            replacement_length = min(len(target), len(tmp))                             # to(torch.float32)  può essere utile
            nn_input[-replacement_length:] = tmp[-replacement_length:]   # array (num_samples,) + astype + NumPy
            nn_input = torch.tensor(nn_input.copy())   # tensore proveniente da array (num_samples,) + astype + NumPy, corretto
            nn_input = torch.stft(nn_input, self.chunk_len, self.stride, window=self.hann, return_complex=False)  
            nn_input = nn_input.permute(2, 0, 1).float() 
            ar_input = target[:-p_size]  # array (num_samples,) + astype + NumPy 
            target = torch.tensor(target[-p_size:].copy())    
            # target = tensore 1D (num_samples,) 1  , convertito come frn                                     
            # ar_input se mono (num_samples,) + astype + NumPy  7
            # nn_input = (num_frames, num_bins, num_channels)
        return ar_input, nn_input, target, p_size





















 
    
    # PROBLEMA IS NOT NONE, FA SOLO LA PRIMA BATCH

   
    # target = torch.stft(target, self.chunk_len, self.stride, window=self.hann,
    #                         return_complex=False).permute(2, 0, 1).float()
    # input = target.copy()
    # input[-p_size:] = 0.
    # input = torch.tensor(input.copy())
    # torch.stft(input, self.chunk_len, self.stride, window=self.hann,
    #                         return_complex=False).permute(2, 0, 1).float()
    

    # dataset_hybrid, l'altro dataset che usava era dataset.py in parcnet per estrarre i 7 pacchetti per AR offline
    # noi possiamo rendere net_input semplicemente input e in training step predirre il residuo e AR packet 
        # truth = wav[:,-(self.n_packets * self.packet_dim + 80):].clone()
        # past = wav[:,-(self.packet_dim * self.n_packets+80):-(self.packet_dim+80)].clone()
        # net_input = torch.cat((past, torch.zeros(1,self.packet_dim+80)), dim=1)  # past + zeros