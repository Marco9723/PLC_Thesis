class CONFIG:
    gpus = "0"  # gpu devices

    class TRAIN:
        batch_size = 32  # number of audio files per batch
        lr = 1e-4  # learning rate 
        epochs = 200  # training epochs
        workers = 8  # number of dataloader workers
        val_split = 0.1  # validation set percentage
        clipping_val = 1.0 # gradient clipping value  1.0
        patience = 3  # learning rate scheduler's patience
        factor = 0.5  # learning rate reduction factor
        packets_for_signal = 8 # autoregressive training loop duration, 8 epochs

    class NN_MODEL:
        enc_layers = 4  # number of MLP blocks in the encoder
        enc_in_dim = 384  # dimension of the input projection layer in the encoder
        enc_dim = 768  # dimension of the MLP blocks
        pred_dim = 512  # dimension of the LSTM in the predictor  
        pred_layers = 1  # number of LSTM layers in the predictor
        num_valid_nn_packets = 7  # number of valid NN packets for parcnet's inference
        gradient_clip = 2.0
        xfade_len_in = 16
        
    class AR_MODEL:
        ar_order = 128 # 256, 512 or 1024
        diagonal_load = 0.001
        num_valid_ar_packets = 7 # number of valid AR packets for parcnet's inference   

    class DATA:
        dataset = 'vctk'  
        data_dir = {'vctk': {'root': '.../path/to/data/vctk/wav48',
                             'train': ".../path/to/data/vctk/train.txt"},
                    }

        assert dataset in data_dir.keys(), 'Unknown dataset.'
        sr = 48000  
        audio_chunk_len = 48000    # size of chunk taken for each audio files
        window_size = 960          # window size of the STFT operation, equivalent to packet size
        stride = 480               # stride of the STFT operation

        class TRAIN:
            packet_size = 320      # packet sizes for training. All sizes should be divisible by 'audio_chunk_len'
            context_length = 7     # AR and NN context's length during training
            signal_packets = 8     # total length of the final prediction
            fadeout = 80           # extra prediction length for fadeout during training
            padding = 240          # extra prediction length to avoid truncation during training

        class EVAL:
            packet_size = 320
            fadeout = 80           # extra prediction length for fadeout during testing
            padding = 240          # extra prediction length to avoid truncation during testing

    class LOG:
        log_dir = 'lightning_logs'  # checkpoint and log directory
        sample_path = 'audio_samples'  # path to save generated audio samples in evaluation.
