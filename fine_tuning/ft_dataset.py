import glob
import os
import random
import sys
sys.path.append("/nas/home/mviviani/nas/home/mviviani/tesi")
from CODE.config import CONFIG
import librosa
import numpy as np
import soundfile as sf
import torch
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
from CODE.config import CONFIG

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
            # start_index = torch.randint(0, audio_len - chunk_len, (1,))[0]
            start_index = 0
            frames = f._prepare_read(start_index, start_index + chunk_len, -1)
            audio = f.read(frames, always_2d=True, dtype="float32")

        else:
            audio = f.read(always_2d=True, dtype="float32")

    if sr != sample_rate:
        audio = librosa.resample(np.squeeze(audio), orig_sr=sr, target_sr=sample_rate)[:, np.newaxis]

    return audio.T


class TrainDataset(Dataset):
    def __init__(self, mode='train'):
        dataset_name = CONFIG.DATA.dataset                               
        self.target_root = CONFIG.DATA.data_dir[dataset_name]['root']    

        txt_list = CONFIG.DATA.data_dir[dataset_name]['train']           
        self.data_list = self.load_txt(txt_list)                          
        
        if mode == 'train':                                               
            self.data_list, _ = train_test_split(self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)

        elif mode == 'val':
            _, self.data_list = train_test_split(self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)

        self.p_size = CONFIG.DATA.TRAIN.packet_size                    
        self.mode = mode                                               
        self.sr = CONFIG.DATA.sr                                         
        self.window = CONFIG.DATA.audio_chunk_len                         
        self.stride = CONFIG.DATA.stride                                  
        self.chunk_len = CONFIG.DATA.window_size                          
        self.hann = torch.sqrt(torch.hann_window(self.chunk_len))
        self.context_length = CONFIG.DATA.TRAIN.context_length
        self.signal_packets = CONFIG.DATA.TRAIN.signal_packets
        self.fadeout = CONFIG.DATA.TRAIN.fadeout
        self.padding = CONFIG.DATA.TRAIN.padding
        self.previous_predictions = None
        self.current_epoch = 0
        np.random.seed(0)
        torch.manual_seed(0)
        
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

    def fetch_audio(self, index):
        sig = load_audio(self.data_list[index], sample_rate=self.sr, chunk_len=self.window) 
        while sig.shape[1] < self.window:
            idx = torch.randint(0, len(self.data_list), (1,))[0]
            pad_len = self.window - sig.shape[1]                           
                                                                           
            if pad_len < 0.02 * self.sr:                                   
                padding = np.zeros((1, pad_len), dtype=float)
            else:                                                         
                padding = load_audio(self.data_list[idx], sample_rate=self.sr, chunk_len=pad_len)  
            sig = np.hstack((sig, padding))                                
        return sig

    def __getitem__(self, index):

        if self.previous_predictions is None:  # fino ad ora ar e nn da 7 pacchetti
            sig = self.fetch_audio(index)
            sig = sig.reshape(-1).astype(np.float32)
            # if index == 1:
            #     print(sig)
            sig = sig[:int((self.signal_packets*self.p_size) + self.fadeout + self.padding)].copy()   # 7 context + 1 da predirre + 1 di padding, all'inizio del segnale
            target = torch.tensor(sig.copy()).float()     # .float() non necessario  ORIGINALE
            # target = torch.tensor(sig, dtype=torch.float32)
            nn_input = torch.tensor(sig[:int(self.p_size*self.context_length)].copy())  # 7  torch.Size([2240])
            nn_input = torch.cat((nn_input, torch.zeros(int(self.p_size+(self.fadeout+self.padding)))))    # 7 + 1 + 1 torch.Size([2880])
            nn_input = torch.stft(nn_input, self.chunk_len, self.stride, window=self.hann, return_complex=False).permute(2, 0, 1).float()
            ar_input = sig[:int(self.p_size*self.context_length)].copy()
            ar_input = torch.tensor(ar_input.copy()).float()   # .float() necessario????

        else:
            sig = self.fetch_audio(index)
            sig = sig.reshape(-1).astype(np.float32)
            # if index == 1:
            #     print(sig)
            # print(self.previous_predictions[index, :].dtype, sig.dtype)
            pred_length = len(self.previous_predictions[index, :])   # 320, 640, 960, ...
            sig = sig[int(pred_length):int(pred_length + (self.signal_packets*self.p_size) + self.fadeout + self.padding)].copy()
            target = torch.tensor(sig.copy()).float() # lungo 9
            # BISOGNA METTERE IL FADEOUT ?????
            sig = sig[:-(int(self.p_size+(self.fadeout+self.padding)))]  # 7
            sig[-min(len(sig), pred_length):] = self.previous_predictions[index,:][-min(len(sig), pred_length):]
            nn_input = torch.tensor(sig.copy())  # 7
            nn_input = torch.cat((nn_input, torch.zeros(int(self.p_size+(self.fadeout+self.padding)))))    # 7 + 1 + 1 torch.Size([2880])
            nn_input = torch.stft(nn_input, self.chunk_len, self.stride, window=self.hann, return_complex=False).permute(2, 0, 1).float()
            ar_input = sig.copy()
            ar_input = torch.tensor(ar_input.copy()).float()
            # print("getitem, dimensioni:", target.size(), nn_input.size(), ar_input.size())

        return ar_input, nn_input, target