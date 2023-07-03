'''
ECAPA-TDNN run function 

Last modified: 07/2023
Author: Daniela Wiepert, Sampath Gogineni
Email: wiepert.daniela@mayo.edu
File: run.py
'''

#IMPORTS
#built-in
import argparse
import ast
import itertools
import os
import pickle

#third-party
import numpy as np
import torch
import pandas as pd
import pyarrow

from google.cloud import storage
from torch.utils.data import  DataLoader

#local
from dataloader import ECAPA_TDNNDataset
from utilities import *
from models import *
from loops import *

def train_ecapa_tdnn(args):
    """
    Run finetuning from start to finish
    :param args: dict with all the argument values
    """
    print('Running training: ')
    # (1) load data
    assert '.csv' not in args.data_split_root, f'May have given a full file path, please confirm this is a directory: {args.data_split_root}'
    train_df, val_df, test_df = load_data(args.data_split_root, args.target_labels, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    if args.debug:
        train_df = train_df.iloc[0:8,:]
        val_df = val_df.iloc[0:8,:]
        test_df = test_df.iloc[0:8,:]

    # (2) set up audio configuration for transforms
    train_audio_conf = {'resample_rate':args.resample_rate, 'reduce': args.reduce,'clip_length': args.clip_length, 
                        'tshift':args.tshift, 'speed':args.speed, 'gauss_noise':args.gauss, 'pshift':args.pshift, 'pshiftn':args.pshiftn, 'gain':args.gain, 'stretch': args.stretch, 'mixup':args.mixup,
                        'n_mfcc':args.n_mfcc, 'n_fft': args.n_fft, 'n_mels': args.n_mels, 
                        'fbank':args.fbank, 'freqm': args.freqm, 'timem': args.timem,  'noise':args.noise,
                        'mean':args.dataset_mean, 'std':args.dataset_std}
    
    eval_audio_conf = {'resample_rate':args.resample_rate, 'reduce': args.reduce,'clip_length': args.clip_length, 
                        'tshift':0, 'speed':0, 'gauss_noise':0, 'pshift':0, 'pshiftn':0, 'gain':0, 'stretch': 0, 'mixup':0,
                        'n_mfcc':args.n_mfcc, 'n_fft': args.n_fft, 'n_mels': args.n_mels, 
                        'fbank':args.fbank, 'freqm': 0, 'timem': 0,  'noise':False,
                        'mean':args.dataset_mean, 'std':args.dataset_std}

    # (3) set up datasets and dataloaders
    dataset_train = ECAPA_TDNNDataset(train_df, target_labels=args.target_labels, audio_conf=train_audio_conf,
                                      prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    dataset_val = ECAPA_TDNNDataset(val_df, target_labels=args.target_labels, audio_conf=eval_audio_conf,
                                      prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    dataset_test = ECAPA_TDNNDataset(test_df, target_labels=args.target_labels, audio_conf=eval_audio_conf,
                                      prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    
    dataloader_train = DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    dataloader_val= DataLoader(dataset_val, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    dataloader_test= DataLoader(dataset_test, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    #dataloader_test = DataLoader(dataset_test, batch_size = len(diag_test), shuffle = False, num_workers = args.num_workers)

    # (4) initialize model
    if args.fbank:
        n_size = args.n_mels
    else:
        n_size = args.n_mfcc

    model = ECAPA_TDNNForSpeechClassification(n_size=n_size, label_dim=args.n_class, lin_neurons=args.lin_neurons,
                                              shared_dense=args.shared_dense, sd_bottleneck=args.sd_bottleneck, clf_bottleneck=args.clf_bottleneck,
                                              activation=args.activation, final_dropout=args.final_dropout, layernorm=args.layernorm)
    
    # (5) start fine-tuning classification
    model = train(model, dataloader_train, dataloader_val, 
                  args.optim, args.learning_rate, args.weight_decay,
                  args.loss, args.scheduler, args.max_lr, args.epochs,
                  args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    print('Saving final model')
    if args.fbank:
        e_type = 'fbank'
    else:
        e_type = 'mfcc'
    path = os.path.join(args.exp_dir, '{}_{}_{}_epoch{}_ecapa_tdnn_{}'.format(args.dataset, np.sum(args.n_class), args.optim, args.epochs, e_type))

    mdl_path = path + '_mdl.pt'
    torch.save(model.state_dict(), mdl_path)

    if args.cloud:
        upload(args.cloud_dir, mdl_path, args.bucket)

    
    # (6) start evaluating
    preds, targets = evaluation(model, dataloader_test)
    
    print('Saving predictions and targets')
    pred_path = path + '_predictions.pt'
    target_path = path + '_targets.pt'
    torch.save(preds, pred_path)
    torch.save(targets, target_path)

    if args.cloud:
        upload(args.cloud_dir, pred_path, args.bucket)
        upload(args.cloud_dir, target_path, args.bucket)

    print('Training finished')


def eval_only(args):
    """
    Run only evaluation of a pre-existing model
    :param args: dict with all the argument values
    """
    assert args.trained_mdl_path is not None, 'must give a model to load'
    # get original model args (or if no finetuned model, uses your original args)
    model_args = load_args(args, args.trained_mdl_path)
    
   # (1) load data
    if '.csv' in args.data_split_root: 
        eval_df = pd.read_csv(args.data_split_root, index_col = 'uid')

        if 'distortions' in args.target_labels and 'distortions' not in eval_df.columns:
            eval_df["distortions"]=((eval_df["distorted Cs"]+eval_df["distorted V"])>0).astype(int)
        
        eval_df = eval_df.dropna(subset=args.target_labels)
    else:
        train_df, val_df, eval_df = load_data(args.data_split_root, args.target_labels, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)
    
    if args.debug:
        eval_df = eval_df.iloc[0:8,:]

   # (2) set up audio configuration for transforms
    audio_conf = {'resample_rate':model_args.resample_rate, 'reduce': model_args.reduce,'clip_length': model_args.clip_length, 
                        'tshift':0, 'speed':0, 'gauss_noise':0, 'pshift':0, 'pshiftn':0, 'gain':0, 'stretch': 0, 'mixup':0,
                        'n_mfcc':model_args.n_mfcc, 'n_fft': model_args.n_fft, 'n_mels': model_args.n_mels, 
                        'fbank':model_args.fbank, 'freqm': 0, 'timem': 0,  'noise':False,
                        'mean':model_args.dataset_mean, 'std':model_args.dataset_std}
    #think that any configuration args that alter the input dims must be the SAME as the model you are loading in.
    #model

    # (3) set up datasets and dataloaders
    dataset_eval = ECAPA_TDNNDataset(eval_df, target_labels=model_args.target_labels, audio_conf=audio_conf,
                                      prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    
    dataloader_eval= DataLoader(dataset_eval, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    
    # (4) initialize model
    #set up n_size (different if fbank or mfcc)
    if model_args.fbank:
        n_size = model_args.n_mels
    else:
        n_size = model_args.n_mfcc

    model = ECAPA_TDNNForSpeechClassification(n_size=n_size, label_dim=model_args.n_class, lin_neurons=model_args.lin_neurons, 
                                              shared_dense=model_args.shared_dense, sd_bottleneck=model_args.sd_bottleneck, clf_bottleneck=model_args.clf_bottleneck,
                                              activation=model_args.activation, final_dropout=model_args.final_dropout, layernorm=model_args.layernorm)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.trained_mdl_path, map_location=device)
    model.load_state_dict(sd, strict=False)

     # (6) start evaluating
    preds, targets = evaluation(model, dataloader_eval)
    
    print('Saving predictions and targets')
    pred_path = os.path.join(args.exp_dir, '{}_predictions.pt'.format(args.dataset))
    target_path = os.path.join(args.exp_dir, '{}_targets.pt'.format(args.dataset))
    torch.save(preds, pred_path)
    torch.save(targets, target_path)

    if args.cloud:
        upload(args.cloud_dir, pred_path, args.bucket)
        upload(args.cloud_dir, target_path, args.bucket)

   
    print('Evaluation finished')

def get_embeddings(args):
    """
    Run embedding extraction from start to finish
    :param args: dict with all the argument values
    """
    print('Running Embedding Extraction: ')
    assert args.trained_mdl_path is not None, 'must give a model to load for embedding extraction. '
    # Get original 
    model_args = load_args(args, args.trained_mdl_path)

    # (1) load data to get embeddings for
    assert '.csv' in args.data_split_root, f'A csv file is necessary for embedding extraction. Please make sure this is a full file path: {args.data_split_root}'
    annotations_df = pd.read_csv(args.data_split_root, index_col = 'uid') #data_split_root should use the CURRENT arguments regardless of the finetuned model

    if 'distortions' in args.target_labels and 'distortions' not in annotations_df.columns:
        annotations_df["distortions"]=((annotations_df["distorted Cs"]+annotations_df["distorted V"])>0).astype(int)

    if args.debug:
        annotations_df = annotations_df.iloc[0:8,:]

    # (2) set up audio configuration for transforms
    audio_conf = {'resample_rate':args.resample_rate, 'reduce': args.reduce,'clip_length': args.clip_length, 
                        'tshift':0, 'speed':0, 'gauss_noise':0, 'pshift':0, 'pshiftn':0, 'gain':0, 'stretch': 0, 'mixup':0,
                        'n_mfcc':model_args.n_mfcc, 'n_fft': model_args.n_fft, 'n_mels': model_args.n_mels, 
                        'fbank':model_args.fbank, 'freqm': 0, 'timem': 0,  'noise':False,
                        'mean':model_args.dataset_mean, 'std':model_args.dataset_std}
    
    #think that any configuration args that alter the input dims must follow the loaded model (model_args)
    
    # (3) set up dataloaders
    waveform_dataset = ECAPA_TDNNDataset(annotations_df, target_labels=args.target_labels, audio_conf=audio_conf,
                                      prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
     #not super important for embeddings, but the dataset should be selecting targets based on the FINETUNED model
    dataloader = DataLoader(waveform_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # (4) initialize model
    if model_args.fbank:
        n_size = model_args.n_mels
    else:
        n_size = model_args.n_mfcc

    model = ECAPA_TDNNForSpeechClassification(n_size=n_size, label_dim=model_args.n_class, lin_neurons=model_args.lin_neurons, 
                                            shared_dense=model_args.shared_dense, sd_bottleneck=model_args.sd_bottleneck, clf_bottleneck=model_args.clf_bottleneck,
                                            activation=model_args.activation, final_dropout=model_args.final_dropout, layernorm=model_args.layernorm)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.trained_mdl_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    
    # (5) get embeddings
    embeddings = embedding_extraction(model, dataloader, args.embedding_type, args.pooling_mode)
        
    df_embed = pd.DataFrame([[r] for r in embeddings], columns = ['embedding'], index=annotations_df.index)

    path= '{}/{}_{}_embeddings'.format(args.exp_dir, args.dataset, args.embedding_type)

    try:
        pqt_path = path + '.pqt'
        
        df_embed.to_parquet(path=pqt_path, index=True, engine='pyarrow') 

        if args.cloud:
            upload(args.cloud_dir, pqt_path, args.bucket)
    except:
        print('Unable to save as pqt, saving instead as csv')
        csv_path = path + '.csv'
        df_embed.to_csv(csv_path, index=True)

        if args.cloud:
            upload(args.cloud_dir, csv_path, args.bucket)

    print('Embedding extraction finished')
    return df_embed

def main():
    parser = argparse.ArgumentParser()
    #Inputs
    parser.add_argument('-i','--prefix',default='/', help='Input directory or location in google cloud storage bucket containing files to load')
    parser.add_argument("-s", "--study", default=None, help="specify study name")
    parser.add_argument("-d", "--data_split_root", default='', help="specify file path where datasplit is located. If you give a full file path to classification, an error will be thrown. On the other hand, evaluation and embedding expects a single .csv file.")
    parser.add_argument('-l','--label_txt', default='src/labels.txt')
    parser.add_argument('--lib', default=False, type=ast.literal_eval, help="Specify whether to load using librosa as compared to torch audio")
    parser.add_argument("--trained_mdl_path", default=None, help="specify path to a trained model")
    #GCS
    parser.add_argument('-b','--bucket_name', default=None, help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default=None, help='google cloud platform project name')
    parser.add_argument('--cloud', default=False, type=ast.literal_eval, help="Specify whether to save everything to cloud")
    #output
    parser.add_argument("--dataset", default=None,type=str, help="When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root")
    parser.add_argument("-o", "--exp_dir", default="./experiments", help='specify LOCAL output directory')
    parser.add_argument('--cloud_dir', default='', type=str, help="if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket")
    #Mode specific
    parser.add_argument("-m", "--mode", choices=['train','eval','extraction'], default='extraction')
    parser.add_argument("--shared_dense", type=ast.literal_eval, default=False, help="specify whether to add an additional shared dense layer before the classifier(s)")
    parser.add_argument("--sd_bottleneck", type=int, default=768, help="specify whether to decrease when using shared_dense layer")
    parser.add_argument('--embedding_type', type=str, default=None, help='specify whether embeddings should be extracted from classification head (ft), base pretrained model (pt), or shared dense layer (st)', choices=['ft','pt', 'st', None])
    parser.add_argument('--pooling_mode', default='mean', choices=['mean','sum'],help='specify how to pool embeddings (mean, sum)')
    #Audio transforms
    parser.add_argument("--resample_rate", default=16000,type=int, help='resample rate for audio files')
    parser.add_argument("--reduce", default=True, type=ast.literal_eval, help="Specify whether to reduce to monochannel")
    parser.add_argument("--clip_length", default=10.0, type=float, help="If truncating audio, specify clip length in seconds. 0 = no truncation")
    parser.add_argument("--tshift", default=0, type=float, help="Specify p for time shift transformation")
    parser.add_argument("--speed", default=0, type=float, help="Specify p for speed tuning")
    parser.add_argument("--gauss", default=0, type=float, help="Specify p for adding gaussian noise")
    parser.add_argument("--pshift", default=0, type=float, help="Specify p for pitch shifting")
    parser.add_argument("--pshiftn", default=0, type=float, help="Specify number of steps for pitch shifting")
    parser.add_argument("--gain", default=0, type=float, help="Specify p for gain")
    parser.add_argument("--stretch", default=0, type=float, help="Specify p for audio stretching")
    parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    #ecapa-tdnn specific
    parser.add_argument("--lin_neurons", default=192, type=int, help='specify # of neurons for ECAPA_TDNN')
    parser.add_argument("--n_mfcc", default=80, type=int, help='specify number of MFCCs for spectrogram')
    parser.add_argument("--n_fft", default=400, type=int, help="specify number of frequency bins in spectrogram")
    parser.add_argument("--n_mels", default=128, type=int, help="specify number of mels for spectrogram")
    #if using fbank instead of mfcc
    parser.add_argument("--fbank", default=False, type=ast.literal_eval)
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)
    parser.add_argument("--noise", type=ast.literal_eval, default=False, help="specify if augment noise in finetuning")
    parser.add_argument("--dataset_mean", default=0, type=float, help="the dataset mean, used for input normalization")
    parser.add_argument("--dataset_std", default=0, type=float, help="the dataset std, used for input normalization")
    #Model parameters
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="specify batch size")
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="specify number of parallel jobs to run for data loader")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="specify learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="specify number of training epochs")
    parser.add_argument("--optim", type=str, default="adamw", help="training optimizer", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=.0001, help='specify weight decay for adamw')
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["MSE", "BCE"])
    parser.add_argument("--scheduler", type=str, default=None, help="specify lr scheduler", choices=["onecycle", None])
    parser.add_argument("--max_lr", type=float, default=0.01, help="specify max lr for lr scheduler")
    #classification head parameters
    parser.add_argument("--activation", type=str, default='relu', help="specify activation function to use for classification head")
    parser.add_argument("--final_dropout", type=float, default=0.3, help="specify dropout probability for final dropout layer in classification head")
    parser.add_argument("--layernorm", type=ast.literal_eval, default=False, help="specify whether to include the LayerNorm in classification head")
    parser.add_argument("--clf_bottleneck", type=int, default=768, help="specify whether to apply a bottleneck to initial classifier dense layer")
    #OTHER
    parser.add_argument("--debug", default=False, type=ast.literal_eval)
    args = parser.parse_args()
    
    print('Torch version: ',torch.__version__)
    print('Cuda availability: ', torch.cuda.is_available())
    print('Cuda version: ', torch.version.cuda)
    
    #brief check
    if args.fbank and args.n_fft != 1024:
        print(f'Target length is {args.n_fft}. We recommend setting to 1024 but it is not required')
    #variables
    # (1) Set up GCS
    if args.bucket_name is not None:
        storage_client = storage.Client(project=args.project_name)
        bucket = storage_client.bucket(args.bucket_name)
    else:
        bucket = None

    # (2), check if given study or if the prefix is the full prefix.
    if args.study is not None:
        args.prefix = os.path.join(args.prefix, args.study)
    
    # (3) get dataset name
    if args.dataset is None:
        if args.trained_mdl_path is None or args.mode == 'train':
            if '.csv' in args.data_split_root:
                args.dataset = '{}_{}'.format(os.path.basename(os.path.dirname(args.data_split_root)), os.path.basename(args.data_split_root[:-4]))
            else:
                args.dataset = os.path.basename(args.data_split_root)
        else:
            args.dataset = os.path.basename(args.trained_mdl_path)[:-7]
    
    # (4) get target labels
     #get list of target labels
    if args.label_txt is None:
        assert args.mode == 'extraction', 'Must give a txt with target labels for training or evaluating.'
        args.target_labels = None
        args.label_groups = None
        args.n_class = []
    else:
        if args.label_txt[:5] =='gs://':
            label_txt = args.label_txt[5:].replace(args.bucket_name,'')[1:]
            bn = os.path.basename(label_txt)
            blob = bucket.blob(label_txt)
            blob.download_to_filename(bn)
            label_txt = bn
        else:
            label_txt = args.label_txt

        with open(label_txt) as f:
            target_labels = f.readlines()
        target_labels = [l.strip().split(sep=",") for l in target_labels]
        args.label_groups = target_labels 
        args.target_labels = list(itertools.chain.from_iterable(target_labels))
        args.n_class = [len(l) for l in args.label_groups]

        if args.n_class == []:
            assert args.mode == 'extraction', 'Target labels must be given for training or evaluating. Txt file was empty.'


    # (5) check if output directory exists, SHOULD NOT BE A GS:// path
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # (6) check that clip length has been set
    if args.clip_length == 0:
        try: 
            assert args.batch_size == 1, 'Not currently compatible with different length wav files unless batch size has been set to 1'
        except:
            args.batch_size = 1
    
    # (7) dump arguments
    if args.mode=='train':
        #only save args if training a model. 
        args_path = "%s/args.pkl" % args.exp_dir
        assert not os.path.exists(args_path), 'Current experiment directory already has an args.pkl file. Please change experiment directory or rename the args.pkl to avoid overwriting the file.'

        with open(args_path, "wb") as f:
            pickle.dump(args, f)
        
        #in case of error, everything is immediately uploaded to the bucket
        if args.cloud:
            upload(args.cloud_dir, args_path, bucket)

    # (8) check if trained model is stored in gcs bucket or confirm it exists on local machine 
    if args.mode != 'train' and args.trained_mdl_path is not None:
        args.trained_mdl_path = gcs_model_exists(args.trained_mdl_path, args.bucket_name, args.exp_dir, bucket)

    #(9) add bucket to args
    args.bucket = bucket

    # (10) run model
    if args.mode == "train":
        train_ecapa_tdnn(args)

    elif args.mode == 'eval':
        eval_only(args)
              
    elif args.mode == "extraction":
        df_embed = get_embeddings(args)
    
if __name__ == "__main__":
    main()