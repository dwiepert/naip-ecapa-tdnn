'''
ECAPA-TDNN run function 

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: run.py
'''

#IMPORTS
#built-in
import argparse
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
    print('Running finetuning: ')
    # (1) load data
    assert '.csv' not in args.data_split_root, f'May have given a full file path, please confirm this is a directory: {args.data_split_root}'
    train_df, val_df, test_df = load_data(args.data_split_root, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    if args.debug:
        train_df = train_df.iloc[0:8,:]
        val_df = val_df.iloc[0:8,:]
        test_df = test_df.iloc[0:8,:]

    # (2) set up audio configuration for transforms
    audio_conf = {'checkpoint': args.checkpoint, 'resample_rate':args.resample_rate, 'reduce': args.reduce,
                  'trim': args.trim, 'clip_length': args.clip_length, 'n_mfcc':args.n_mfcc, 'n_fft': args.n_fft, 'n_mels': args.n_mels}

    # (3) set up datasets and dataloaders
    dataset_train = ECAPA_TDNNDataset(train_df, target_labels=args.target_labels, audio_conf=audio_conf,
                                      prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    dataset_val = ECAPA_TDNNDataset(val_df, target_labels=args.target_labels, audio_conf=audio_conf,
                                      prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    dataset_test = ECAPA_TDNNDataset(test_df, target_labels=args.target_labels, audio_conf=audio_conf,
                                      prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    
    dataloader_train = DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    dataloader_val= DataLoader(dataset_val, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    dataloader_test= DataLoader(dataset_test, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    #dataloader_test = DataLoader(dataset_test, batch_size = len(diag_test), shuffle = False, num_workers = args.num_workers)

    # (4) initialize model
    model = ECAPA_TDNNForSpeechClassification(n_size=args.n_mfcc, label_dim=args.n_class, lin_neurons=192,
                                              activation=args.activation, final_dropout=args.final_dropout, layernorm=args.layernorm)
    
    # (5) start fine-tuning classification
    model = train(model, dataloader_train, dataloader_val, 
                  args.optim, args.learning_rate, args.loss,
                  args.scheduler, args.max_lr, args.epochs,
                  args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    print('Saving final model')
    mdl_path = os.path.join(args.exp_dir, '{}_{}_{}_epoch{}_ecapa_tdnn_mdl.pt'.format(args.dataset, args.n_class, args.optim, args.epochs))
    torch.save(model.state_dict(), mdl_path)

    if args.cloud:
        upload(args.cloud_dir, mdl_path, args.bucket)

    
    # (6) start evaluating
    preds, targets = evaluation(model, dataloader_test, 
                                args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    print('Training finished')


def eval_only(args):
    """
    Run only evaluation of a pre-existing model
    :param args: dict with all the argument values
    """
    assert args.checkpoint is not None, 'must give a model checkpoint to load'
    # get original model args (or if no finetuned model, uses your original args)
    model_args = load_args(args, args.checkpoint)
    
   # (1) load data
    if '.csv' in args.data_split_root: 
        eval_df = pd.read_csv(args.data_split_root, index_col = 'uid')

        if 'distortions' not in eval_df.columns:
            eval_df["distortions"]=((eval_df["distorted Cs"]+eval_df["distorted V"])>0).astype(int)
    else:
        train_df, val_df, eval_df = load_data(args.data_split_root, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)
    
    if args.debug:
        eval_df = eval_df.iloc[0:8,:]

   # (2) set up audio configuration for transforms
    audio_conf = {'checkpoint': args.checkpoint, 'resample_rate':args.resample_rate, 'reduce': args.reduce,
                  'trim': args.trim, 'clip_length': args.clip_length, 'n_mfcc':args.n_mfcc, 'n_fft': args.n_fft, 'n_mels': args.n_mels}


    # (3) set up datasets and dataloaders
    dataset_eval = ECAPA_TDNNDataset(eval_df, target_labels=model_args.target_labels, audio_conf=audio_conf,
                                      prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    
    dataloader_eval= DataLoader(dataset_eval, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    
    # (4) initialize model
    model = ECAPA_TDNNForSpeechClassification(n_size=model_args.n_mfcc, label_dim=model_args.n_class, lin_neurons=192,
                                              activation=model_args.activation, final_dropout=model_args.final_dropout, layernorm=model_args.layernorm)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(sd, strict=False)

     # (6) start evaluating
    preds, targets = evaluation(model, dataloader_eval, 
                                args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

   
    print('Evaluation finished')

def get_embeddings(args):
    """
    Run embedding extraction from start to finish
    :param args: dict with all the argument values
    """
    print('Running Embedding Extraction: ')
    assert args.checkpoint is not None, 'must give a model checkpoint to load for embedding extraction. '
    # Get original 
    model_args = load_args(args, args.checkpoint)

    # (1) load data to get embeddings for
    assert '.csv' in args.data_split_root, f'A csv file is necessary for embedding extraction. Please make sure this is a full file path: {args.data_split_root}'
    annotations_df = pd.read_csv(args.data_split_root, index_col = 'uid') #data_split_root should use the CURRENT arguments regardless of the finetuned model

    if 'distortions' not in annotations_df.columns:
        annotations_df["distortions"]=((annotations_df["distorted Cs"]+annotations_df["distorted V"])>0).astype(int)

    if args.debug:
        annotations_df = annotations_df.iloc[0:8,:]

    # (2) set up audio configuration for transforms
    audio_conf = {'checkpoint': args.checkpoint, 'resample_rate':args.resample_rate, 'reduce': args.reduce,
                  'trim': args.trim, 'clip_length': args.clip_length, 'n_mfcc':args.n_mfcc, 'n_fft': args.n_fft, 'n_mels': args.n_mels}

    
    # (3) set up dataloaders
    waveform_dataset = ECAPA_TDNNDataset(annotations_df, target_labels=model_args.target_labels, audio_conf=audio_conf,
                                      prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
     #not super important for embeddings, but the dataset should be selecting targets based on the FINETUNED model
    dataloader = DataLoader(waveform_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # (4) initialize model
    model = ECAPA_TDNNForSpeechClassification(n_size=model_args.n_mfcc, label_dim=model_args.n_class, lin_neurons=192,
                                              activation=model_args.activation, final_dropout=model_args.final_dropout, layernorm=model_args.layernorm)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(sd, strict=False)
    
    # (5) get embeddings
    embeddings = embedding_extraction(model, dataloader, args.embedding_type)
        
    df_embed = pd.DataFrame([[r] for r in embeddings], columns = ['embedding'], index=annotations_df.index)

    try:
        pqt_path = '{}/{}_ecapa_tdnn_{}_embeddings.pqt'.format(args.exp_dir, args.dataset, args.embedding_type)
        
        df_embed.to_parquet(path=pqt_path, index=True, engine='pyarrow') 

        if args.cloud:
            upload(args.cloud_dir, pqt_path, args.bucket)
    except:
        print('Unable to save as pqt, saving instead as csv')
        csv_path = '{}/{}_ecapa_tdnn_{}_embeddings.csv'.format(args.exp_dir, args.dataset, args.embedding_type)
        df_embed.to_csv(csv_path, index=True)

        if args.cloud:
            upload(args.cloud_dir, csv_path, args.bucket)

    print('Embedding extraction finished')
    return df_embed

def main():
    parser = argparse.ArgumentParser()
    #Inputs
    parser.add_argument('-i','--prefix',default='speech_ai/speech_lake/', help='Input directory or location in google cloud storage bucket containing files to load')
    parser.add_argument("-s", "--study", choices = ['r01_prelim','speech_poc_freeze_1', None], default='speech_poc_freeze_1', help="specify study name")
    parser.add_argument("-d", "--data_split_root", default='gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620/test.csv', help="specify file path where datasplit is located. If you give a full file path to classification, an error will be thrown. On the other hand, evaluation and embedding expects a single .csv file.")
    parser.add_argument('-l','--label_txt', default='/Users/m144443/Documents/GitHub/mayo-ecapa-tdnn/src/labels.txt')
    parser.add_argument('--lib', default=False, type=bool, help="Specify whether to load using librosa as compared to torch audio")
    parser.add_argument("-c", "--checkpoint", default='/Users/m144443/Documents/GitHub/mayo-ecapa-tdnn/experiments/train/amr_subject_dedup_594_train_100_test_binarized_v20220620_5_adam_epoch1_ecapa_tdnn_mdl.pt', help="specify path to pre-trained model weight checkpoint")
    #GCS
    parser.add_argument('-b','--bucket_name', default='ml-e107-phi-shared-aif-us-p', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='ml-mps-aif-afdgpet01-p-6827', help='google cloud platform project name')
    parser.add_argument('--cloud', default=False, type=bool, help="Specify whether to save everything to cloud")
    #output
    parser.add_argument("--dataset", default=None,type=str, help="When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root")
    parser.add_argument("-o", "--exp_dir", default="./experiments/embedding", help='specify LOCAL output directory')
    parser.add_argument('--cloud_dir', default='m144443/temp_out/w2v2_ft_weighted', type=str, help="if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket")
    #Mode specific
    parser.add_argument("-m", "--mode", choices=['train','eval','extraction'], default='extraction')
    parser.add_argument('--embedding_type', type=str, default='pt', help='specify whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)', choices=['ft','pt'])
    #Audio transforms
    parser.add_argument("--resample_rate", default=16000,type=int, help='resample rate for audio files')
    parser.add_argument("--reduce", default=True, type=bool, help="Specify whether to reduce to monochannel")
    parser.add_argument("--clip_length", default=160000, type=int, help="If truncating audio, specify clip length in # of frames. 0 = no truncation")
    parser.add_argument("--trim", default=True, type=int, help="trim silence")
    #Model parameters
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="specify batch size")
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="specify number of parallel jobs to run for data loader")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="specify learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="specify number of training epochs")
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["adam"])
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["MSE", "BCE"])
    parser.add_argument("--scheduler", type=str, default=None, help="specify lr scheduler", choices=["onecycle", None])
    parser.add_argument("--max_lr", type=float, default=0.01, help="specify max lr for lr scheduler")
    #classification head parameters
    parser.add_argument("--activation", type=str, default='relu', help="specify activation function to use for classification head")
    parser.add_argument("--final_dropout", type=float, default=0.25, help="specify dropout probability for final dropout layer in classification head")
    parser.add_argument("--layernorm", type=bool, default=False, help="specify whether to include the LayerNorm in classification head")
    #ecapa-tdnn specific
    parser.add_argument("--n_mfcc", default=80, type=int)
    parser.add_argument("--n_fft", default=400, type=int)
    parser.add_argument("--n_mels", default=128, type=int)
    #OTHER
    parser.add_argument("--debug", default=True, type=bool)
    args = parser.parse_args()
    
    print('Torch version: ',torch.__version__)
    print('Cuda availability: ', torch.cuda.is_available())
    print('Cuda version: ', torch.version.cuda)
    
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
        if '.csv' in args.data_split_root:
            args.dataset = '{}_{}'.format(os.path.basename(os.path.dirname(args.data_split_root)), os.path.basename(args.data_split_root[:-4]))
        else:
            args.dataset = os.path.basename(args.data_split_root)
    
    # (4) get target labels
     #get list of target labels
    with open(args.label_txt) as f:
        target_labels = f.readlines()
    target_labels = [l.strip() for l in target_labels]
    args.target_labels = target_labels

    args.n_class = len(target_labels)

    # (5) check if output directory exists, SHOULD NOT BE A GS:// path
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # (6) check that clip length has been set
    if args.clip_length == 0:
        try: 
            assert args.batch_size == 1, 'Not currently compatible with different length wav files unless batch size has been set to 1'
        except:
            args.batch_size = 1

    # (7) check if checkpoint is stored in gcs bucket or confirm it exists on local machine
    if args.checkpoint is not None:
        if args.checkpoint[:5] =='gs://':
            checkpoint = args.checkpoint[5:].replace(args.bucket_name,'')[1:]
            checkpoint = download_model(checkpoint, bucket)
            args.checkpoint = checkpoint
        else:
            assert os.path.exists(args.checkpoint), 'Current checkpoint does not exist on local machine'

    # (8) dump arguments
    args_path = "%s/args.pkl" % args.exp_dir
    with open(args_path, "wb") as f:
        pickle.dump(args, f)
    #in case of error, everything is immediately uploaded to the bucket
    if args.cloud:
        upload(args.cloud_dir, args_path, bucket)

    #(9) add bucket to args
    args.bucket = bucket

    # (10) run model
    print(args.mode)
    if args.mode == "train":
        train_ecapa_tdnn(args)

    elif args.mode == 'eval':
        eval_only(args)
              
    elif args.mode == "extraction":
        df_embed = get_embeddings(args)
    
if __name__ == "__main__":
    main()