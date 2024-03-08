class CONFIG:
    gpus = "0"  # gpu devices

    class TRAIN:
        batch_size = 20  # number of audio files per batch
        lr = 1e-4  # learning rate
        epochs = 150  # training epochs   
        workers = 8  # number of dataloader workers
        val_split = 0.1  # validation set percentage
        clipping_val = 1.0  # gradient clipping value
        patience = 3  # learning rate scheduler's patience
        factor = 0.5  # learning rate reduction factor

    class NN_MODEL:
        enc_layers = 4  # number of MLP blocks in the encoder
        enc_in_dim = 384  # dimension of the input projection layer in the encoder
        enc_dim = 768  # dimension of the MLP blocks
        pred_dim = 512  # dimension of the LSTM in the predictor  
        pred_layers = 1  # number of LSTM layers in the predictor
        num_valid_nn_packets = 7 
        gradient_clip = 2.0
        max_epochs = 150
        xfade_len_in = 16
        
    class AR_MODEL:
        ar_order = 128
        diagonal_load = 0.001
        num_valid_ar_packets = 7  

    class DATA:
        dataset = 'vctk'  
        data_dir = {'vctk': {'root': '/nas/home/mviviani/nas/home/mviviani/tesi/CODE/data/vctk/wav48',
                             'train': "/nas/home/mviviani/nas/home/mviviani/tesi/data/vctk/train.txt"},
                    }

        assert dataset in data_dir.keys(), 'Unknown dataset.'
        sr = 48000  
        audio_chunk_len = 122880   # size of chunk taken in each audio files
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
        log_dir = 'lightning_logs'
        sample_path = 'audio_samples' 
        log_dir = 'lightning_logs'


