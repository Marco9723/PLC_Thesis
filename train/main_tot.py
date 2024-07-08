import argparse
import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset_tot import TrainDataset
from neural_branch_tot import PLCModel
from CODE.config import CONFIG
from CODE.tblogger import TensorBoardLoggerExpanded

parser = argparse.ArgumentParser()

parser.add_argument('--version', default=None,
                    help='version to resume')
parser.add_argument('--mode', default='train',
                    help='training mode')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.gpus)
assert args.mode in ['train']

def resume(train_dataset, val_dataset, version):
    print("Version", version)
    model_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
    config_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/'.format(str(version)) + 'hparams.yaml')
    model_name = [x for x in os.listdir(model_path) if x.endswith(".ckpt")][0]
    ckpt_path = model_path + model_name
    checkpoint = PLCModel.load_from_checkpoint(ckpt_path,
                                               strict=True,
                                               hparams_file=config_path,
                                               train_dataset=train_dataset,
                                               val_dataset=val_dataset,
                                               window_size=CONFIG.DATA.window_size)

    return checkpoint

def train():
    train_dataset = TrainDataset('train')
    val_dataset = TrainDataset('val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG.TRAIN.batch_size, shuffle=False,
                                               num_workers=CONFIG.TRAIN.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG.TRAIN.batch_size, shuffle=False,
                                             num_workers=CONFIG.TRAIN.workers)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=True,
                                          filename='parcnet-{epoch:02d}-{val_loss:.4f}', save_weights_only=False, save_top_k=10, save_last=True)
    gpus = CONFIG.gpus.split(',')
    logger = TensorBoardLoggerExpanded(CONFIG.DATA.sr)
    if args.version is not None:
        model = resume(train_dataset, val_dataset, args.version)
    else:
        model = PLCModel(train_dataset,
                         val_dataset,
                         window_size=CONFIG.DATA.window_size,
                         enc_layers=CONFIG.NN_MODEL.enc_layers,
                         enc_in_dim=CONFIG.NN_MODEL.enc_in_dim,
                         enc_dim=CONFIG.NN_MODEL.enc_dim,
                         pred_dim=CONFIG.NN_MODEL.pred_dim,
                         pred_layers=CONFIG.NN_MODEL.pred_layers)

    trainer = pl.Trainer(logger=logger,
                         gradient_clip_val=CONFIG.TRAIN.clipping_val,
                         devices=len(gpus),
                         max_epochs=CONFIG.TRAIN.epochs,
                         accelerator='auto',
                         callbacks=[checkpoint_callback]
                         )

    print(model.hparams)
    print(
        'Dataset: {}, Train files: {}, Val files {}'.format(CONFIG.DATA.dataset, len(train_dataset), len(val_dataset)))
    trainer.fit(model, train_loader, val_loader)



if __name__ == '__main__':

    if args.mode == 'train':
        train()

