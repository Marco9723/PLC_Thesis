import glob
import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F

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
        # x non creare discontinuità chuck_len<48000 ?
        sig = self.fetch_audio(index)                           # (1,122880)
        sig = sig.reshape(-1).astype(np.float32)                # (122880)
        start_index = random.randint(0, len(sig) - (self.signal_packets*self.p_size) - self.fadeout - self.padding)  
        sig = sig[start_index:int(start_index + (self.signal_packets*self.p_size) + self.fadeout + self.padding)].copy()   # 8 + 1 da 320
        target = torch.tensor(sig.copy()).float() 
        nn_input = torch.tensor(sig[:int(self.p_size*self.context_length)].copy())  # 7  torch.Size([2240])
        nn_input = torch.cat((nn_input, torch.zeros(int(self.p_size+(self.fadeout+self.padding)))))    # 7 + 1 + 1 torch.Size([2880])
        # MULTIPLI DI 960, se no taglia
        nn_input = torch.stft(nn_input, self.chunk_len, self.stride, window=self.hann, return_complex=False).permute(2, 0, 1).float()
        ar_input = sig[:int(self.p_size*self.context_length)].copy()   # 7, poi ne predice 2
        ar_input = torch.tensor(ar_input.copy()).float()
        # DIMENSIONI: torch.Size([2240]) torch.Size([2, 481, 7]) torch.Size([2880])
        return ar_input, nn_input, target


    








        
'''original_length_tensor1 = len(ar_input)
        padded_tensor1 = F.pad(ar_input, (0, max(0, 10752 - original_length_tensor1)), value=0)   # 10752

        original_length_target = len(target)
        padded_target = F.pad(target, (0, max(0, 1920 - original_length_target)), value=0)        # 1920

        original_length_tensor2 = nn_input.size(2)
        if nn_input.size(-1) < 27:    # <---------- <original_length_tensor2
            padding_size = 27 - nn_input.size(-1)
            padded_tensor2 = F.pad(nn_input, (0, padding_size), value=0)
        else:
            padded_tensor2 = nn_input    # (2, 481, 27)
        # print(padded_tensor1.size(), padded_tensor2.size(), padded_target.size())
        # print(original_length_tensor1, original_length_tensor2, original_length_target)
        return padded_tensor1, original_length_tensor1, padded_tensor2, original_length_tensor2, padded_target, original_length_target, p_size'''

