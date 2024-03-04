import os
import librosa
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI

from PLCMOS.plc_mos import PLCMOSEstimator
from config import CONFIG
from loss import Loss
from models.blocks import Encoder, Predictor
from CODE.utils import visualize, LSD
from ar_branch import ARModel

plcmos = PLCMOSEstimator()

class PLCModel(pl.LightningModule):
    def __init__(self, train_dataset=None, val_dataset=None, window_size=960, enc_layers=4, enc_in_dim=384, enc_dim=768,
                 pred_dim=512, pred_layers=1, pred_ckpt_path=None):
        super(PLCModel, self).__init__()
        self.window_size = window_size
        self.hop_size = window_size // 2
        self.learning_rate = CONFIG.TRAIN.lr
        self.hparams.batch_size = CONFIG.TRAIN.batch_size
        self.num_epochs = CONFIG.TRAIN.epochs
        self.ar_order = CONFIG.AR_MODEL.ar_order
        self.diagonal_load = CONFIG.AR_MODEL.diagonal_load
        self.ar_model = ARModel(self.ar_order, self.diagonal_load)  #nell'init direi

        self.enc_layers = enc_layers
        self.enc_in_dim = enc_in_dim
        self.enc_dim = enc_dim
        self.pred_dim = pred_dim
        self.pred_layers = pred_layers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.stoi = STOI(48000)
        self.pesq = PESQ(16000, 'wb')
        # usare num_epochs per capire quante predizioni future fare?
        

        self.predictor = Predictor(window_size=self.window_size, lstm_dim=self.pred_dim,
                                       lstm_layers=self.pred_layers)
        
        self.joiner = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(9, 1), stride=1, padding=(4, 0), padding_mode='reflect',
                      groups=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(48, 2, kernel_size=1, stride=1, padding=0, groups=2),
        )

        self.encoder = Encoder(in_dim=self.window_size, dim=self.enc_in_dim, depth=self.enc_layers,
                               mlp_dim=self.enc_dim)

        self.loss = Loss()
        self.window = torch.sqrt(torch.hann_window(self.window_size))
        self.save_hyperparameters('window_size', 'enc_layers', 'enc_in_dim', 'enc_dim', 'pred_dim', 'pred_layers')     

    def forward(self, x):
        """
        Input: real-imaginary; shape (B, F, T, 2); F = hop_size + 1
        Output: real-imaginary
        """

        B, C, F, T = x.shape

        x = x.permute(3, 0, 1, 2).unsqueeze(-1)
        prev_mag = torch.zeros((B, 1, F, 1), device=x.device)
        predictor_state = torch.zeros((2, self.predictor.lstm_layers, B, self.predictor.lstm_dim), device=x.device)
        mlp_state = torch.zeros((self.encoder.depth, 2, 1, B, self.encoder.dim), device=x.device)
        result = []
        for step in x:
            feat, mlp_state = self.encoder(step, mlp_state)
            prev_mag, predictor_state = self.predictor(prev_mag, predictor_state)
            feat = torch.cat((feat, prev_mag), 1)
            feat = self.joiner(feat)
            feat = feat + step
            result.append(feat)
            prev_mag = torch.linalg.norm(feat, dim=1, ord=1, keepdims=True)  # compute magnitude
        output = torch.cat(result, -1)
        
        return output
    
    def train_dataloader(self):               
        return DataLoader(self.train_dataset, shuffle=False, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, persistent_workers=True)                            # previous_predictions= self.previous_predictions)    # oppure basta passargli solo self.previous_predictions
                                                                                                                # self.train_dataset.update_data(self.previous_predictions)    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, persistent_workers=True)        

    def training_step(self, batch, batch_idx):
                                                                                                                # automaticamente tutti tensori quelli provenienti da batch # ogni elemento della batch contiene cose relative allo stesso dato
        ar_past, nn_past, target, packet_size = batch                                                           # target 1 packet in time formato tensore, ar_past 7 Numpy array, nn_past self.forward(past) = 7 + 1 a zero da predirre (torch stft format)
                                                                                                                # dimensioni [batch_size, channels, ...]
                                                                                                                # in realtà target potrebbe essere 1 solo pacchetto nel nostro caso    <-----
                                                                                                                # ar_past[batch_size, context_lenght]
        teacher_forcing_prob = max(0, 1 - self.trainer.current_epoch/self.num_epochs)                           # teacher forcing probability
        
        f_0 = nn_past[:, :, 0:1, :]                                                                             # nn_past torch tensor stft
        nn_input = nn_past[:, :, 1:, :]
        nn_pred = self(nn_input)                                                                                # predizione
        nn_pred = torch.cat([f_0, nn_pred], dim=2)  
        nn_pred = torch.view_as_complex(nn_pred.permute(0, 2, 3, 1).contiguous())                               # riporta in time
        nn_pred = torch.istft(nn_pred, self.window_size, self.hop_size, window=self.window)                     # 8 packets, ricostruisce tutto il segnale residuo
        nn_pred = nn_pred[:,:,-(packet_size+80):]       #no                                                        # estrai residuo dell'ultimo pacchetto predetto (vedi questione + 80)
        nn_pred = nn_pred.squeeze(dim=1)                                                                        # La forma risultante non cambierà per gli esempi in cui channels è già 1                                    # mantieni batch_size, channel = 1
         
        ar_pred=[]                                          # nn_pred tensore [batch_size, channels, time] in teoria con channels = 1 (funziona solo se channel = 1), dopo riga precedente [batch_size, time], ma vedi se channel serve oppure no, ho preso però l'ultimo pezzo                                                
        for past in ar_past.cpu().numpy():                                                                                                         # se ar_past è un tensore con dimensioni [batch_size, ar_past_length] dovrebbe funzionare
            ar_pred.append(self.ar_model.predict(past, packet_size+80))                                        # la predizione è l'ultimo pacchetto lungo PACKET_DIM+80, formato NumPy array
                                                                                                                # ar_past dovrebbe essere un tensore [batch_size, ar_past_length]
        ar_pred=torch.tensor(np.array(ar_pred))                                                                                                        # ar_past.numpy(), converte in una matrice numpy [batch_size, ar_past_length]
                                                                                             # NECESSARIA, per riconvertire in tensore
                                                                                                                # ar_pred dovrebbe avere le stesse dimensioni di nn_pred a questo punto  (se no al max fai un ciclo for)
        pred = nn_pred + ar_pred                                                              
        target = target.squeeze(dim=1)                                                                          # dipende se channel = 1 o se non c'è   # TARGET DOVREBBE CONTENERE IL TARGET GIUSTO
                                                                                                                # target meglio che sia lungo solo 1 pacchetto direi, target tensore in time di dimensioni [batch_size, channel=1, time]
                                                                                                                # per ogni batch salverei il tensore [batch_size, pred_length]
        if torch.rand(1) < teacher_forcing_prob:    
            output = target                                                                                     # così non salvi durante il training!
        else:
            output = pred                                                                                       # QUELLO CHE SALVI è UN TENSORE [batch_size, predizione]
            
                                                                                                                # METTERE QUELLO FATTO SOPRA NEL BRANCH ELSE?
                                                                                                                # se serve .numpy() x convertire da tensore a matrice numpy nel nostro caso 2D
        
                                                                                              # parcnet loss ha bisogno di squeeze lungo la dimensione di channel ()= 1): nn_pred fatto, ar_pred dovrebbe avere stesse dimensioni
                                                                                              # target anche in teoria, assicuratene  -->  dovrebbe essere giusto avere tutto di dimensione [batch_size, predizione]
        mse_loss = self.mse_loss(output, target)   
        sc_loss, log_loss = self.stft_loss(y_pred=output.squeeze(1), y_true=target.squeeze(1))
        spectral_loss = 0.5 * (sc_loss + log_loss)
        tot_loss = self.lmbda * mse_loss + spectral_loss

        self.log('mse_loss', mse_loss, prog_bar=True)
        self.log('sc_loss', sc_loss, prog_bar=False)
        self.log('log_mag_loss', log_loss, prog_bar=False)
        self.log('spectral_loss', spectral_loss, prog_bar=True)
        self.log('tot_loss', tot_loss, prog_bar=True)

        return {'loss': tot_loss, 'training_output': output}     
 
    def training_epoch_end(self, outputs):                                                    # così lo fai alla fine dopo tutti i training step e non vai a influire sul training!
        predictions = [output['training_output'] for output in outputs]   

        if self.train_dataset.previous_predictions is None:
            self.train_dataset.previous_predictions = torch.cat(predictions, dim=0) 
        else:
            # SE SEI ALLA DECIMA EPOCA RESETTA A NONE
            self.train_dataset.previous_predictions = torch.cat([self.train_dataset.previous_predictions, predictions[-1]], dim=0)

    def validation_step(self, val_batch, batch_idx):
        # valutazione della performance, forward pass, calcolo loss, calcolare metriche come accuratezza e precisione (opzionale), ma no backward pass e ottimizzazione
        # si esegue solo il forward pass per ottenere le predizioni e calcolare la loss
        # vedi val_dataset
        ar_past, nn_past, target, packet_size = val_batch                                        
                                                                                                                                                                          
        teacher_forcing_prob = max(0, 1 - self.trainer.current_epoch/self.num_epochs)         
        
        f_0 = nn_past[:, :, 0:1, :]          
        nn_input = nn_past[:, :, 1:, :]
        nn_pred = self(nn_input)                                                             
        nn_pred = torch.cat([f_0, nn_pred], dim=2)  
        nn_pred = torch.view_as_complex(nn_pred.permute(0, 2, 3, 1).contiguous())            
        nn_pred = torch.istft(nn_pred, self.window_size, self.hop_size, window=self.window)   
        nn_pred = nn_pred[:,:,-(packet_size+80):]                                             
        nn_pred = nn_pred.squeeze(dim=1)                                                         
                                                                                                                                                                                                                       
        ar_pred = self.ar_model.predict(ar_past.numpy(), packet_size+80)   
                                                                                                                                                                               
        ar_pred = torch.from_numpy(ar_pred)                                                  
                                                                                    
        pred = nn_pred + ar_pred                                                              
        target = target.squeeze(dim=1)                                                                                                                                
                                                                                        
        if torch.rand(1) < teacher_forcing_prob:   
            output = target                                                              
        else:
            output = pred                                       

        val_loss = 
        packet_val_loss = 

        self.log('val_loss', val_loss)
        self.log('packet_val_loss', packet_val_loss)

        return
    
    
    def test_step(self, test_batch, batch_idx):

           return
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            # parcnet= PARCnet(...)
            # y_pred = parcnet(y_lost, trace)
            return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=CONFIG.TRAIN.patience,
                                                                  factor=CONFIG.TRAIN.factor, verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

