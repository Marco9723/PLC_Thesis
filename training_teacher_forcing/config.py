class CONFIG:
    gpus = "0"  # gpu devices

    class TRAIN:
        batch_size = 20  # number of audio files per batch
        lr = 1e-4  # learning rate  1e-4
        epochs = 500  # training epochs
        workers = 8  # number of dataloader workers
        val_split = 0.1  # validation set percentage
        clipping_val = 1.0 # gradient clipping value  1.0
        patience = 3  # learning rate scheduler's patience
        factor = 0.5  # learning rate reduction factor
        packets_for_signal = 8 # 0,1,2,3,4,5,6,7

    class NN_MODEL:
        enc_layers = 4  # number of MLP blocks in the encoder
        enc_in_dim = 384  # dimension of the input projection layer in the encoder
        enc_dim = 768  # dimension of the MLP blocks
        pred_dim = 512  # dimension of the LSTM in the predictor  
        pred_layers = 1  # number of LSTM layers in the predictor
        num_valid_nn_packets = 7 # SOLO IN PARCNET.PY
        gradient_clip = 2.0
        xfade_len_in = 16
        
    class AR_MODEL:
        ar_order = 128 # 128
        diagonal_load = 0.001
        num_valid_ar_packets = 7   # SOLO IN PARCNET.PY

    class DATA:
        dataset = 'vctk'  
        data_dir = {'vctk': {'root': '/nas/home/mviviani/nas/home/mviviani/tesi/CODE/data/vctk/wav48',
                             'train': "/nas/home/mviviani/nas/home/mviviani/tesi/data/vctk/train.txt"},
                    }

        assert dataset in data_dir.keys(), 'Unknown dataset.'
        sr = 48000  
        audio_chunk_len = 48000  # 48000   122880  # size of chunk taken in each audio files
        window_size = 960          # window size of the STFT operation, equivalent to packet size
        stride = 480               # stride of the STFT operation

        class TRAIN:
            packet_size = 320      # packet sizes for training. All sizes should be divisible by 'audio_chunk_len'
            context_length = 7
            signal_packets = 8
            fadeout = 80
            padding = 240

        class EVAL:
            packet_size = 320
            fadeout = 80
            padding = 240

    class LOG:
        log_dir = 'lightning_logs'  # checkpoint and log directory
        sample_path = 'audio_samples'  # path to save generated audio samples in evaluation.



