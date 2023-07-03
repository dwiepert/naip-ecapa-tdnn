# ECAPA-TDNN for Mayo Data
The command line usable, start-to-finish implementation of ECAPA-TDNN is available with [run.py](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/run.py). A notebook tutorial version is also available at [run.ipynb](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/run.ipynb). 

This implementation trains an ECAPA-TDNN model and a classification head from scratch and can extract embeddings from the trained model. 

This implementation uses wrapper classes over an [ECAPA-TDNN model](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.ECAPA_TDNN.html) available with Speechbrain. The `ECAPA_TDNNForSpeechClassification` is the wrapped model, which includes an added classification head with a Dense layer, ReLU activation, dropout, and a final linear projection layer (this class is defined as `ClassificationHead` in [speech_utils.py](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/utilities/speech_utils.py)) as well as a function for embedding extraction. See [ecapa_tdnn_models.py](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/models/ecapa_tdnn_models.py) for information on intialization arguments.

## Running Environment
The environment must include the following packages, all of which can be dowloaded with pip or conda:
* albumentations
* librosa
* torch, torchvision, torchaudio
* tqdm (this is essentially enumerate(dataloader) except it prints out a nice progress bar for you)
* speechbrain 
* pyarrow

If running on your local machine and not in a GCP environment, you will also need to install:
* google-cloud-storage

To access data stored in GCS on your local machine, you will need to additionally run

```gcloud auth application-default login```

```gcloud auth application-defaul set-quota-project PROJECT_NAME```

Please note that if using GCS, the model expects arguments like model paths or directories to start with `gs://BUCKET_NAME/...` with the exception of defining an output cloud directory which should just be the prefix to save within a bucket. 

## Model checkpoints
Unlike some of our other models like [SSAST](https://github.com/dwiepert/mayo-ssast/main) and [W2V2](https://github.com/dwiepert/mayo-w2v2/main), we are training a model from scratch. As such, we only ever load in fully trained models, which is a required step for running only evaluation (`--mode` set to 'eval') or embedding extraction (`--mode` set to extraction). If running in one of those modes, specify a path to a trained model with the `--trained_mdl_path` argument. It expects that this argument will contain a full file path pointing to a single model, and that the directory this model is contained in (i.e., `os.path.dirname(args.trained_mdl_path)` contains an `arg.pkl` file. 

## Data structure
This code will only function with the following data structure.

SPEECH DATA DIR

    |

    -- UID 

        |

        -- waveform.EXT (extension can be any audio file extension)

        -- metadata.json (containing the key 'encoding' (with the extension in capital letters, i.e. mp3 as MP3), also containing the key 'sample_rate_hz' with the full sample rate)

and for the data splits

DATA SPLIT DIR

    |

    -- train.csv

    -- test.csv

## Audio Configuration
Data is loaded using an `ECAPA_TDNNDataset` class, where you pass a dataframe of the file names (UIDs) along with columns containing label data, a list of the target labels (columns to select from the df), specify audio configuration, method of loading, and initialize transforms on the raw waveform and spectrogram (see [dataloader.py](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/dataloader.py)). 

To specify audio loading method, you can alter the `bucket` variable and `librosa` variable. As a default, `bucket` is set to None, which will force loading from the local machine. If using GCS, pass a fully initialized bucket. Setting the `librosa` value to 'True' will cause the audio to be loaded using librosa rather than torchaudio. 

The audio configuration parameters should be given as a dictionary (which can be seen in [run.py](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/run.py) and [run.ipynb](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/run.ipynb). Most configuration values are for initializing audio and spectrogram transforms. The transform will only be initialized if the value is not 0. If you have a further desire to add transforms, see [speech_utils.py](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/utilities/speech_utils.py)) and alter [dataloader.py](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/dataloader.py) accordingly. 

The following parameters are accepted (`--` indicates the command line argument to alter to set it):

*Audio Transform Information*
* `resample_rate`: an integer value for resampling. Set with `--resample_rate`
* `reduce`: a boolean indicating whether to reduce audio to monochannel. Set with `--reduce`
* `clip_length`: float specifying how many seconds the audio should be. Will work with the 'sample_rate' of the audio to get # of frames. Set with `--clip_length`
* `tshift`: Time shifting parameter (between 0 and 1). Set with `--tshift`
* `speed`: Speed tuning parameter (between 0 and 1). Set with `--speed`
* `gauss_noise`: amount of gaussian noise to add (between 0 and 1). Set with `--gauss`
* `pshift`: pitch shifting parameter (between 0 and 1). Set with `--pshift`
* `pshiftn`: number of steps for pitch shifting. Set with `--pshiftn`
* `gain`: gain parameter (between 0 and 1).Set with `--gain`
* `stretch`: audio stretching parameter (between 0 and 1). Set with `--stretch`
* `mixup`: parameter for file mixup (between 0 and 1). Set with `--mixup`

*Spectrogram Transform Information*
* `n_mfcc`: integer for number of MFCCs to extract
* `n_fft`: integer for number of frequency bins
* `n_mels`: integer for number of mels
* `fbank`: boolean to indicate whether to extract MFCC based spectrogram or fbank, if selected, target length is `n_fft` and num mel bins is `n_mels`. For consistency, set `n_fft` to 1024. 
* `freqm`: frequency mask paramenter for use only with `fbank` version. Set with `--freqm`
* `timem`: time mask parameter for use only with `fbank` version. Set with `--timem`
* `noise`: add default noise to spectrogram for use only with `fbank` version. Set with `--noise`
* `mean`: dataset mean (float) for use only with `fbank` version. Set with `--dataset_mean`
* `std`: dataset standard deviation (float) for use only with `fbank` version. Set with `--dataset_std`

# Arguments
There are many possible arguments to set, including all the parameters associated with audio configuration. The main run function describes most of these, and you can alter defaults as required. 

### Loading data
* `-i, --prefix`: sets the `prefix` or input directory. Compatible with both local and GCS bucket directories containing audio files, though do not include 'gs://'
* `-s, --study`: optionally set the study. You can either include a full path to the study in the `prefix` arg or specify some parent directory in the `prefix` arg containing more than one study and further specify which study to select here.
* `-d, --data_split_root`: sets the `data_split_root` directory or a full path to a single csv file. For classification, it must be  a directory containing a train.csv and test.csv of file names. If runnning embedding extraction, it should be a csv file. Running evaluation only can accept either a directory or a csv file. This path should include 'gs://' if it is located in a bucket. 
* `-l, --label_txt`: sets the `label_txt` path. This is a full file path to a .txt file contain a list of the target labels for selection (see [labels.txt](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/labels.txt). Features in same classifier group should be split by ',', each feature classifier group should be split by '/n'). If stored in a bucket it If it is empty, it will require that embedding extraction be running.
* `--lib`: : specifies whether to load using librosa (True) or torchaudio (False), default=False
* `--trained_mdl_path`: if running eval-only or extraction, you must specify a trained model model to load in. This can either be a local path of a 'gs://' path, that latter of which will trigger the code to download the specified model path to the local machine. 

### Google cloud storage
* `-b, --bucket_name`: sets the `bucket_name` for GCS loading. Required if loading from cloud.
* `-p, --project_name`: sets the `project_name` for GCS loading. Required if loading from cloud. 
* `--cloud`: this specifies whether to save everything to GCS bucket. It is set as True as default.

### Saving data
* `--dataset`: Specify the name of the dataset you are using. When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root. Default is None. 
* `-o, --exp_dir`: sets the `exp_dir`, the LOCAL directory to save all outputs to. 
* `--cloud_dir`: if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket. Do not include the bucket_name or 'gs://' in this path.

### Run mode
* `-m, --mode`: Specify the mode you are running, i.e., whether to run fine-tuning for classification ('finetune'), evaluation only ('eval-only'), or embedding extraction ('extraction'). Default is 'finetune'.
* `--shared_dense`: specify whether to include a shared dense layer
* `--sd_bottleneck`: specify bottleneck for shared dense layer
* `--embedding_type`: specify whether embeddings should be extracted from classification head (ft), base pretrained model (pt), or shared dense layer (st)
* `--pooling_mode`: specify how to pool embeddings if there are multiple classification heads, should be either 'mean' or 'sum'

### Audio transforms
see the audio configurations section for which arguments to set

### Spectrogram transforms
see the audio configurations section for which arguments to set

### Model parameters
* `-bs, --batch_size`: set the batch size (default 8)
* `-nw, --num_workers`: set number of workers for dataloader (default 0)
* `-lr, --learning_rate`: you can manually change the learning rate (default 0.0003)
* `-e, --epochs`: set number of training epochs (default 1)
* `--optim`: specify the training optimizer. Default is `adam`.
* `--weight_decay`: specify weight decay for AdamW optimizer
* `--loss`: specify the loss function. Can be 'BCE' or 'MSE'. Default is 'BCE'.
* `--scheduler`: specify a lr scheduler. If None, no lr scheduler will be use. The only scheduler option is 'onecycle', which initializes `torch.optim.lr_scheduler.OneCycleLR`
* `--max_lr`: specify the max learning rate for an lr scheduler. Default is 0.01.

### Classification Head parameters
* `--activation`: specify activation function to use for classification head
* `--final_dropout`: specify dropout probability for final dropout layer in classification head
* `--layernorm`: specify whether to include the LayerNorm in classification head
* `--clf_bottleneck`: specify bottleneck for classifier initial dense layer

For more information on arguments, you can also run `python run.py -h`. 

## Functionality
This implementation contains functionality options as listed below:

### 1. Training from scratch
You can train an ECAPA-TDNN model from scratch for classifying speech features using the `ECAPA_TDNNForSpeechClassification` class in [ecapa_tdnn_models.py](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/models/ecapa_tdnn_models.py) and the `train(...)` function in [loops.py](https://github.com/dwiepert/mayo-ecapa-tdnn/blob/main/src/loops.py).

This mode is triggered by setting `-m, --mode` to 'train'. 

You can add a shared dense layer prior to the classification head(s) by specifying  `--shared_dense` along with `--sd_bottleneck` to designate the output size for the shared dense layer. 

Classification head(s) can be implemented in the following manner:
1. Specify `--clf_bottleneck` to designate output for initial linear layer 
2. Give `label_dims` as a list of dimensions or a single int. If given as a list, it will make a number of classifiers equal to the number of dimensions given, with each dimension indicating the output size of the classifier (e.g. [2, 1] will make a classifier with an output of (batch_size, 2) and one with an output of (batch_size, 1). The outputs then need to be stacked by columns to make one combined prediction). In order to do this in `run.py`, you must give a label_txt in the following format: split labels with a ',' to specify a group of features to be fed to one classifier; split with a new line '/n' to specify a new classifier. Note that `args.target_labels` should be a flat list of features, but `args.label_groups` should be a list of lists. 

There are data augmentation transforms available for finetuning, such as time shift, speed tuning, adding noise, pitch shift, gain, stretching audio, and audio mixup. 

Note we include two different spectrogram input options. 
1. `mfcc`: extracts the mel spectrogram based on MFCCs. Trigger by setting `fbank` to False. This version uses default values of: `n_mfcc = 80`, `n_fft = 400`, `n_mels = 128`.
2. `fbank`: extracts the fbank from a waveform. Trigger by setting `fbank` to True. This version uses `n_fft` and `n_mels`. We recommend setting `n_fft` to 1024 for consistency with SSAST and timm model extraction. This version also has additional transform parameters for frequency masking, time masking, normalization, and adding noise. Please see more in the Audio Configurations section. 

### 2. Evaluation only
If you have a trained model and want to evaluate it on a new data set, you can do so by setting `-m, --mode` to 'eval'. You must then also specify a `--trained_mdl_path` to load in. 

It is expected that there is an `args.pkl` file in the same directory as the model to indicate which arguments were used to initialize the model. This implementation will load the arguments and initialize/load the  model with these arguments. If no such file exists, it will use the arguments from the current run, which could be incompatible if you are not careful. 

### 3. Embedding extraction
We implemented multiple embedding extraction methods for use with the SSAST model. The implementation is a function within `ECAPA_TDNNForSpeechClassification` called `extract_embedding(x, embedding_type)`, which is called on batches instead of the forward function. 

Embedding extraction is triggered by setting `-m, --mode` to 'extraction'. 

You must also consider where you want the embeddings to be extracted from. The options are as follows:
1. From the output of the base ECAPA-TDNN model? Set `embedding_type` to 'pt'. 
2. From a layer in the classification head? Set `embedding_type` to 'ft'. This version requires specification of `pooling_mode` to merge embeddings if there are multiple classifiers. It only accepts "mean" or "sum" for merging, and if nothing is specified it will use the pooling_mode set with the model. It will always return the output from the first dense layer in the classification head, prior to any activation function or normalization. 
3. After a shared dense layer? Set `embedding_type` to 'st'. This version requires that the model was initially trained with `shared_dense` set to True.

Brief note on target labels:
Embedding extraction is the only mode where target labels are not required. You can give None or an empty list or np.array and it will still function and extract embeddings.
