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


class TrainDataset(Dataset):
    def __init__(self, mode='train'):                                                                                                 
        dataset_name = CONFIG.DATA.dataset                                # dataset
        self.target_root = CONFIG.DATA.data_dir[dataset_name]['root']     # root folder

        txt_list = CONFIG.DATA.data_dir[dataset_name]['train']            # setting txt file 
        self.data_list = self.load_txt(txt_list)                          # data_list
        
        if mode == 'train':                                               # training set and validation set
            self.data_list, _ = train_test_split(self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)

        elif mode == 'val':
            _, self.data_list = train_test_split(self.val_list, CONFIG.TRAIN.val_split, random_state=0)

        self.p_sizes = CONFIG.DATA.TRAIN.packet_sizes                     # [256, 512, 768, 960, 1024, 1536]
        self.mode = mode                                                  # 'train' or 'val'
        self.sr = CONFIG.DATA.sr                                          # sample rate (48 kHz)
        self.window = CONFIG.DATA.audio_chunk_len                        
        self.stride = CONFIG.DATA.stride                                  # stride of the STFT operation: 480
        self.chunk_len = CONFIG.DATA.window_size                          # window size of the STFT operation, equivalent to packet size: 960
        self.hann = torch.sqrt(torch.hann_window(self.chunk_len))         # hanning window
        self.context_lenght = CONFIG.DATA.TRAIN.context_lenght
        # self.previous_predictions = None
        self.packet_sizes = None                                       
        
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
        sig = load_audio(self.data_list[index], sample_rate=self.sr, chunk_len=self.window)
        while sig.shape[1] < self.window:                                  
            idx = torch.randint(0, len(self.data_list), (1,))[0]          
            pad_len = self.window - sig.shape[1]                           
                                                                           
            if pad_len < 0.02 * self.sr:                                  
                padding = np.zeros((1, pad_len), dtype=np.float)           
            else:                                                         
                padding = load_audio(self.data_list[idx], sample_rate=self.sr, chunk_len=pad_len)   
            sig = np.hstack((sig, padding))                                
        return sig

    def __getitem__(self, index):
                                                                                                                                                                                     
        target = self.fetch_audio(index)         # (num_channels, num_samples) + NumPy                                                                    
        target = self.convert_to_mono(target)    # (num_samples,) + NumPy                             
        target = target.astype(np.float32)       # (num_samples,) + astype + NumPy                                                                                        
        # target = target.reshape(-1).astype(np.float32)                                      
        # con reshape sarebbe stato  (num_samples*numchannels,) + astype
        
        if self.previous_predictions is None:                                                   
            p_size = random.choice(self.p_sizes)                                                
            self.packet_sizes = torch.cat(p_size, dim=1)                                      
            target = target[:(self.context_lenght+1)*p_size]             
            nn_input = target.copy()   
            nn_input[-p_size:] = 0.0                                                 
            ar_input = target[:-p_size]                                         
            target = target[-p_size:]                                               
            
            target = torch.tensor(target.copy())                                           
            nn_input = torch.tensor(nn_input.copy())  
            nn_input = torch.stft(nn_input, self.chunk_len, self.stride, window=self.hann, return_complex=False)
            nn_input = nn_input.permute(2, 0, 1).float()       
            
        elif self.previous_predictions is not None:                                             
            p_size = self.packet_sizes[index] 
            target = target[len(self.previous_predictions[index,:]) : len(self.previous_predictions[index,:])+(self.context_lenght+1)*p_size]  # new target, 8 packets 
            nn_input = target.copy()                                                                    
            tmp = torch.cat([self.previous_predictions[index,:], torch.zeros(p_size)]) # tensor
            tmp= tmp.numpy()
            tmp = tmp.reshape(-1).astype(np.float32)  #  target and tmp array 
            replacement_length = min(len(target), len(tmp))                            
            nn_input[-replacement_length:] = tmp[-replacement_length:]   # array
            nn_input = torch.tensor(nn_input.copy())   # tensor
            nn_input = torch.stft(nn_input, self.chunk_len, self.stride, window=self.hann, return_complex=False)  
            nn_input = nn_input.permute(2, 0, 1).float() 
            ar_input = target[:-p_size]  # array 
            target = torch.tensor(target[-p_size:].copy())    

        return ar_input, nn_input, target, p_size



