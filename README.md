# PARCnet: a new hybrid packet loss concealment technique for speech applications based on LPC and deep residual learning

# 1. Results

We present an hybrid technique where a feed-forward neural network and an autoregressive model cooperate in order to reconstruct the lost packets. The model under consideration is PARCnet, a recent PLC technique which has demonstrated excellent results in the field of realtime music networks in terms of auditory perception and time inference. The main goal of this work is to move from the musical domain to that of speech, trying to replicate the excellent results of the original model thanks to new implementation of the two branches and training strategies. Our new model has been tested with state-of-the-art objective metrics and with a perceptual test on a sample of listeners. Results confirm that such method outperforms the baselines taken into consideration and it charts a new course for addressing the PLC problem.

We also provide several audio samples for comparison at !public repository index.html!

# 2. Installation

## Setup

### Clone the repo

```
$ git clone https://github.com/Marco9723/PLC_Thesis.git
$ cd PLC_Thesis
```

### Create environment and install requirements

* Install `libsndfile` libraries for the Python packages `soundfile`
    ```
    $ apt-get update && apt-get install libsndfile-dev
    ```
* Create a Python 3.9 environment with Conda:
   ```
   $ conda create -n frn python=3.9
   $ conda activate frn
   ```
* Install requirements:
    ```
    $ pip install -r requirements.txt 
    ```

# 3. Data preparation

* Download and extract the datasets. In our paper, we conduct experiments on the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset.
    ```
    $ wget http://www.udialogue.org/download/VCTK-Corpus.tar.gz -O data/vctk/VCTK-Corpus.tar.gz
    $ tar -zxvf data/vctk/VCTK-Corpus.tar.gz -C data/vctk/ --strip-components=1
    ```

  After extracting the datasets, `./data` directory should look like this:

    ```
    .
    |--data
        |--vctk
            |--wav48
                |--p225
                    |--p225_001.wav
                    ...
            |--train.txt   
            |--test.txt
    ```
* In order to load the datasets, text files that contain training and testing audio paths are required. We have
  prepared `train.txt` and `test.txt` files in `./data/vctk` directory.

* In `./data` it is recommended to also save the files and traces of the testing set: 

    ```
    .
    |--data
        |--vctk
        |--traces
        |--testing
    ```


# 4. Run the code

## Configuration

* Configure your experiments in `config.py` 

## Training

* Adjust training hyperparameters in `config.py`.

* Modify paths where needed. If necessary, recheck the imports if they cause problems.

* The train folder contains the main files of the autoregressive training with teacher forcing.

* Run `main.py`:
    ```
    $ python main.py --mode train
    ```
* In case you want to continue training from one of these versions, just set the argument `--version` of the above command to your desired version number. For example:
    ```
    # resume from version 0
    $ python main.py --mode train --version 0
    ```

## Evaluation  

* Get your evaluation set and modify paths where needed. If necessary, recheck the imports if they cause problems.
  
* Modify `config.py` to change evaluation setup if necessary.
  
* Run `testing.py`:
    ```
    $ python testing.py 
    ```
    
* To reconstruct a lossy file run "reconstruct.py"
    ```
    $ python reconstruct.py 
    ```

* You will probably need to modify `CONFIG.TEST.in_dir` to your input directory.
  

## Configure a new dataset   

The implementation can be easily extended to a new one.

* Prepare `train.txt` and `test.txt`. See `./data/vctk/train.txt` and `./data/vctk/test.txt` for example.

* Add a new dictionary to `CONFIG.DATA.data_dir`:
    ```
    {
    'root': 'path/to/data/directory',
    'train': 'path/to/train.txt',
    'test': 'path/to/test.txt'
    }
    ```
  **Important:** Make sure each line in `train.txt` and `test.txt` joining with `'root'` is a valid path to its
  corresponding audio file.




