import torch

def remove_prefix(state_dict, prefix='pretrained_model.'):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# Carica il tuo file di checkpoint
checkpoint_path = 'C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/fine_tuning/lightning_logs/version_0/checkpoints/parcnet-epoch=96-packet_val_loss=-0.8142.ckpt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Estrai solo lo stato del modello
model_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Rimuovi il prefisso 'pretrained_model.' dai nomi dei parametri
new_state_dict = remove_prefix(model_state_dict)

# Sovrascrivi il model_state_dict nel checkpoint con il nuovo stato modificato
if 'state_dict' in checkpoint:
    checkpoint['state_dict'] = new_state_dict
else:
    checkpoint = new_state_dict

# Crea un nuovo checkpoint con lo stato modificato
new_checkpoint_path = 'C:/Users/marco/Documents/GitHub/Packet_Loss_Concealment_Thesis/TESI/CODE/fine_tuning/lightning_logs/version_230/checkpoints/format-parcnet-epoch=96-packet_val_loss=-0.8142.ckpt'
torch.save(checkpoint, new_checkpoint_path)

print(f"Nuovo checkpoint salvato in: {new_checkpoint_path}")



