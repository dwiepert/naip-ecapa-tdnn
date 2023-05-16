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
import json
import os
import pickle

#third-party
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import pandas as pd
import pyarrow

from google.cloud import storage, bigquery
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import  DataLoader

#local
from utilities import *
from models import *

#GCS helper functions
def download_model(gcs_path,outpath, bucket):
    '''
    Download a model from google cloud storage and the args.pkl file located in the same folder(if it exists)

    Inputs:
    :param gcs_path: full file path in the bucket to a pytorch model(no gs://project-name in the path)
    :param outpath: string path to directory where you want the model to be stored
    :param bucket: initialized GCS bucket object
    Outputs:
    :return mdl_path: a string path to the local version of the finetuned model (args.pkl will be in the same folder as this model)
    '''
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    dir_path = os.path.dirname(gcs_path)
    bn = os.path.basename(gcs_path)
    blobs = bucket.list_blobs(prefix=dir_path)
    mdl_path = ''
    for blob in blobs:
        blob_bn = os.path.basename(blob.name)
        if blob_bn == bn:
            destination_uri = '{}/{}'.format(outpath, blob_bn) #download model 
            mdl_path = destination_uri
        elif blob_bn == 'args.pkl':
            destination_uri = '{}/model_args.pkl'.format(outpath) #download args.pkl as model_args.pkl
        else:
            continue #skip any other files
        if not os.path.exists(destination_uri):
            blob.download_to_filename(destination_uri)
   
    return mdl_path

def upload(gcs_prefix, path, bucket):
    '''
    Upload a file to a google cloud storage bucket
    Inputs:
    :param gcs_dir: directory path in the bucket to save file to (no gs://project-name in the path)
    :param path: local string path of the file to upload
    :param bucket: initialized GCS bucket object
    '''
    assert bucket is not None, 'no bucket given for uploading'
    if gcs_prefix is None:
        gcs_prefix = os.path.dirname(path)
    blob = bucket.blob(os.path.join(gcs_prefix, os.path.basename(path)))
    blob.upload_from_filename(path)

#Load functions
def load_args(args):
    '''
    Load in an .pkl file of args
    :param args: dict with all the argument values
    :return model_args: dict with all the argument values from the finetuned model
    '''
    # assumes that the model is saved in the same folder as an args.pkl file 
    folder = os.path.dirname(args.finetuned_mdl_path)

    if os.path.exists(os.path.join(folder, 'model_args.pkl')): #if downloaded from gcs into the exp dir, it should be saved under mdl_args.pkl to make sure it doesn't overwrite the args.pkl
        with open(os.path.join(folder, 'model_args.pkl'), 'rb') as f:
            model_args = pickle.load(f)
    elif os.path.exists(os.path.join(folder, 'args.pkl')): #if not downloaded and instead stored in a local place, it will be saved as args.pkl
        with open(os.path.join(folder, 'args.pkl'), 'rb') as f:
            model_args = pickle.load(f)
    else: #if there are no saved args
        print('No args.pkl or model_args.pkl stored with the finetuned model. Using the current args for initializing the finetuned model instead.')
        model_args = args
    
    return model_args

def setup_mdl_args(args):
    '''
    Get model args used during finetuning of the specified model
    :param args: dict with all the argument values
    :return model_args: dict with all the argument values from the finetuned model
    :return finetuned_mdl_path: updated finetuned_mdl_path (in case it needed to be downloaded from gcs)
    '''
    #if running a pretrained model only, use the args from this run
    if args.finetuned_mdl_path is None:
        model_args = args
    else:
    #if running a finetuned model
        #(1): check if saved on cloud and load the model and args.pkl
        if args.finetuned_mdl_path[:5] =='gs://':
                mdl_path = args.finetuned_mdl_path[5:].replace(args.bucket_name,'')[1:]
                args.finetuned_mdl_path = download_model(mdl_path, args.exp_dir, args.bucket)
        
        #(2): load the args used for finetuning
        model_args = load_args(args)

        #(3): check if the checkpoint for the finetuned model is downloaded
        if model_args.pretrained_mdl_path[:5] =='gs://': #if checkpoint on cloud
            checkpoint = model_args.pretrained_mdl_path[5:].replace(model_args.bucket_name,'')[1:]
            if model_args.bucket_name != args.bucket_name: #if the bucket is not the same as the current bucket, initialize the bucket for downloading
                if args.bucket_name is not None:
                    storage_client = storage.Client(project=model_args.project_name)
                    model_args.bucket = storage_client.bucket(model_args.bucket_name)
                else:
                    model_args.bucket = None

                checkpoint = download_model(checkpoint, model_args.bucket) #download with the new bucket
            else:
                checkpoint = download_model(checkpoint, args.bucket) #download with the current bucket
            model_args.pretrained_mdl_path = checkpoint #reset the checkpoint path
        else: #load in from local machine, just need to check that the path exists
            assert os.path.exists(model_args.pretrained_mdl_path), f'Current pretrain checkpoint does not exist on local machine: {model_args.pretrained_mdl_path}'

    return model_args, args.finetuned_mdl_path

def load_data(data_split_root, exp_dir, cloud, cloud_dir, bucket):
    """
    Load the train and test data from a directory. Assumes the train and test data will exist in this directory under train.csv and test.csv
    :param data_split_root: specify str path where datasplit csvs are located
    :param exp_dir: specify LOCAL output directory as str
    :param cloud: boolean to specify whether to save everything to google cloud storage
    :param cloud_dir: if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket
    :param bucket: google cloud storage bucket object
    :return train_df, val_df, test_df: loaded dataframes with annotations
    """
    train_path = f'{data_split_root}/train.csv'
    test_path = f'{data_split_root}/test.csv'
    #get data
    train_df = pd.read_csv(train_path, index_col = 'uid')
    test_df = pd.read_csv(test_path, index_col = 'uid')

    #randomly sample to get validation set 
    val_df = train_df.sample(50)
    train_df = train_df.drop(val_df.index)

    #save validation set
    val_path = os.path.join(exp_dir, 'validation.csv')
    val_df.to_csv(val_path, index=True)

    if cloud:
        upload(cloud_dir, val_path, bucket)

    #alter data columns
    train_df["distortions"]=((train_df["distorted Cs"]+train_df["distorted V"])>0).astype(int)
    val_df["distortions"]=((val_df["distorted Cs"]+val_df["distorted V"])>0).astype(int)
    test_df["distortions"]=((test_df["distorted Cs"]+test_df["distorted V"])>0).astype(int)

    return train_df, val_df, test_df

#data transformations
def get_transform(args):
    """
    Set up pre-processing transform for raw samples 
    Loads data, reduces to 1 channel, downsamples, trims silence, truncate(?) and run feature extraction
    :param args: dict with all the argument values
    return transform: transforms object 
    """
    waveform_loader = UidToWaveform(prefix = args.prefix, bucket=args.bucket, lib=args.lib)
    transform_list = [waveform_loader]
    if args.reduce:
        channel_sum = lambda w: torch.sum(w, axis = 0).unsqueeze(0)
        mono_tfm = ToMonophonic(reduce_fn = channel_sum)
        transform_list.append(mono_tfm)
    if args.resample_rate != 0: #16000
        downsample_tfm = Resample(args.resample_rate)
        transform_list.append(downsample_tfm)
    if args.trim:
        trim_tfm = TrimSilence()
        transform_list.append(trim_tfm)
    if args.clip_length != 0: #160000
        truncate_tfm = Truncate(length = args.clip_length)
        transform_list.append(truncate_tfm)

    tensor_tfm = ToTensor()
    transform_list.append(tensor_tfm)
    mfcc_tfm = MFCC(n_mfcc= args.n_mfcc, n_fft=args.n_fft, n_mels=args.n_mels)
    transform_list.append(mfcc_tfm)
    #mel_transforms = MelSpectrogram(n_fft=400, n_mels=128)
    #transform_list.append(mel_transforms)
    transform = torchvision.transforms.Compose(transform_list)
    return transform

#training loops
def train_loop(args, model, dataloader_train, dataloader_val = None):
    """
    Training loop 
    :param args: dict with all the argument values
    :param model: ECAPA-TDNN model
    :param dataloader_train: dataloader object with training data
    :param dataloader_val: dataloader object with validation data
    :return model: trained ECAPA-TDNN model
    """
    print('Training start')
    #send to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #loss
    if args.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError('MSE must be given for loss parameter')
    #optimizer
    if args.optim == 'adam':
        optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)
    elif args.optim == 'adamw':
         optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    else:
        raise ValueError('adam must be given for optimizer parameter')
    
    if args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.max_lr, steps_per_epoch=len(dataloader_train), epochs=args.epochs)
    else:
        scheduler = None
    
    #train
    for e in range(args.epochs):
        training_loss = list()
        #t0 = time.time()
        model.train()
        for batch in tqdm(dataloader_train):
            x = batch['mfcc']
            targets = batch['targets']
            x, targets = x.to(device), targets.to(device)
            optim.zero_grad()
            o = model(x)
            loss = criterion(o, targets)
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()
            loss_item = loss.item()
            training_loss.append(loss_item)

        if e % 10 == 0:
            #SET UP LOGS
            if scheduler is not None:
                lr = scheduler.get_last_lr()
            else:
                lr = args.learning_rate
            logs = {'epoch': e, 'optim':args.optim, 'loss_fn': args.loss, 'lr': lr}
    
            logs['training_loss_list'] = training_loss
            training_loss = np.array(training_loss)
            logs['running_loss'] = np.sum(training_loss)
            logs['training_loss'] = np.mean(training_loss)

            print('RUNNING LOSS', e, np.sum(training_loss) )
            print(f'Training loss: {np.mean(training_loss)}')

            if dataloader_val is not None:
                print("Validation start")
                validation_loss = val_loop(model, criterion, dataloader_val)

                logs['val_loss_list'] = validation_loss
                validation_loss = np.array(validation_loss)
                logs['val_running_loss'] = np.sum(validation_loss)
                logs['val_loss'] = np.mean(validation_loss)
                
                print('RUNNING VALIDATION LOSS',e, np.sum(validation_loss) )
                print(f'Validation loss: {np.mean(validation_loss)}')
            
            #SAVE LOGS
            json_string = json.dumps(logs)
            logs_path = os.path.join(args.exp_dir, 'logs_epoch{}.json'.format(e))
            with open(logs_path, 'w') as outfile:
                json.dump(json_string, outfile)
            
            #SAVE CURRENT MODEL
            print(f'Saving epoch {e}')
            mdl_path = os.path.join(args.exp_dir, 'ecapa_tdnn_mdl_epoch{}.pt'.format(e))
            torch.save(model.state_dict(), mdl_path)

            optim_path = os.path.join(args.exp_dir, 'ecapa_tdnn_optim_epoch{}.pt'.format(e))
            torch.save(optim.state_dict(), optim_path)

            if args.cloud:
                upload(args.cloud_dir, logs_path, args.bucket)
                #upload_from_memory(model.state_dict(), args.cloud_dir, mdl_path, args.bucket)
                upload(args.cloud_dir, mdl_path, args.bucket)
                upload(args.cloud_dir, optim_path, args.bucket)

    print('Saving final model')
    mdl_path = os.path.join(args.exp_dir, '{}_{}_{}_{}_epoch{}_ecapa_tdnn_mdl.pt'.format(args.dataset,args.model_size, args.n_class, args.optim, args.epochs))
    torch.save(model.state_dict(), mdl_path)

    optim_path = os.path.join(args.exp_dir, '{}_{}_{}_{}_epoch{}_ecapa_tdnn_optim.pt'.format(args.dataset,args.model_size, args.n_class, args.optim, args.epochs))
    torch.save(optim.state_dict(), optim_path)

    if args.cloud:
        upload(args.cloud_dir, mdl_path, args.bucket)
        upload(args.cloud_dir, optim_path, args.bucket)

    print('Training finished')
    return model

def val_loop(model, criterion, dataloader_val):
    '''
    Validation loop
    :param model: ECAPA-TDNN model
    :param criterion: loss function
    :param dataloader_val: dataloader object with validation data
    :return validation_loss: list with validation loss for each batch
    '''
    validation_loss = list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader_val):
            x = batch['mfcc']
            targets = batch['targets']
            x, targets = x.to(device), targets.to(device)
            o = model(x)
            val_loss = criterion(o, targets)
            validation_loss.append(val_loss.item())

    return validation_loss

def eval_loop(model, dataloader_eval, exp_dir, cloud=False, cloud_dir=None, bucket=None):
    """
    Start model evaluation
    :param model: ECAPA-TDNN model
    :param dataloader_eval: dataloader object with evaluation data
    :param exp_dir: specify LOCAL output directory as str
    :param cloud: boolean to specify whether to save everything to google cloud storage
    :param cloud_dir: if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket
    :param bucket: google cloud storage bucket object
    :return preds: model predictions
    :return targets: model targets (actual values)
    """
    print('Evaluation start')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = []
    t = []
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader_eval):
            x = batch['mfcc']
            x = x.to(device)
            targets = batch['targets']
            targets = targets.to(device)
            o = model(x)
            outputs.append(o)
            t.append(targets)

    outputs = torch.cat(outputs).cpu().detach()
    t = torch.cat(t).cpu().detach()
    # SAVE PREDICTIONS AND TARGETS 
    pred_path = os.path.join(exp_dir, 'ecapa_tdnn_eval_predictions.pt')
    target_path = os.path.join(exp_dir, 'ecapa_tdnn_eval_targets.pt')
    torch.save(outputs, pred_path)
    torch.save(t, target_path)

    if cloud:
        upload(cloud_dir, pred_path, bucket)
        upload(cloud_dir, target_path, bucket)

    print('Evaluation finished')
    return outputs, t

def embedding_loop(model, dataloader,embedding_type='ft'):
    """
    Run a specific subtype of evaluation for getting embeddings.
    :param model: W2V2 model
    :param dataloader_eval: dataloader object with data to get embeddings for
    :param embedding_type: string specifying whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)
    :return embeddings: an np array containing the embeddings
    """
    print('Getting embeddings')
    embeddings = np.array([])

    # send to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader):
            x = batch['mfcc']
            x = x.to(device)
            e = model.extract_embedding(x, embedding_type)
            e = e.cpu().numpy()
            if embeddings.size == 0:
                embeddings = e
            else:
                embeddings = np.append(embeddings, e, axis=0)
        
    return embeddings

#main running functions
def train(args):
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

    # (2) get data transforms    
    transform = get_transform(args)

    # (3) set up datasets and dataloaders
    dataset_train = WaveformDataset(train_df, target_labels = args.target_labels, transform = transform)
    dataset_val = WaveformDataset(val_df, target_labels = args.target_labels, transform = transform)
    dataset_test = WaveformDataset(test_df, target_labels = args.target_labels, transform = transform)

    dataloader_train = DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    dataloader_val= DataLoader(dataset_val, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    dataloader_test= DataLoader(dataset_test, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    #dataloader_test = DataLoader(dataset_test, batch_size = len(diag_test), shuffle = False, num_workers = args.num_workers)

    # (4) initialize model
    model = ECAPA_TDNNForSpeechClassification(args.n_mfcc, args.n_labels, args.activation, args.final_dropout, args.layernorm) #should look like the finetuned model (so using model_args). If pretrained model, will resort to current args
    
    # (5) start fine-tuning classification
    model = train_loop(args, model, dataloader_train, dataloader_val)

    # (6) start evaluating
    preds, targets = eval_loop(model, dataloader_test, args.exp_dir, args.cloud, args.cloud_dir, args.bucket, args.layer, args.weighted_layers)

    print('Finetuning finished')


def eval_only(args):
    """
    Run only evaluation of a pre-existing model
    :param args: dict with all the argument values
    """
    assert args.checkpoint is not None, 'must give a model checkpoint to load for embedding extraction.'
    # get original model args (or if no finetuned model, uses your original args)
    model_args, args.finetuned_mdl_path = setup_mdl_args(args)
    
   # (1) load data
    if '.csv' in args.data_split_root: 
        eval_df = pd.read_csv(args.data_split_root, index_col = 'uid')
    else:
        train_df, val_df, eval_df = load_data(args.data_split_root, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)
    
    if args.debug:
        eval_df = eval_df.iloc[0:8,:]

    # (2) get data transforms    
    transform = get_transform(args)

    # (3) set up datasets and dataloaders
    dataset_eval = WaveformDataset(eval_df, target_labels = model_args.target_labels, transform = transform)  #the dataset should be selecting targets based on the FINETUNED model, so if there is a mismatch, it defaults to the arguments used for finetuning
    dataloader_eval= DataLoader(dataset_eval, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    #dataloader_test = DataLoader(dataset_test, batch_size = len(diag_test), shuffle = False, num_workers = args.num_workers)

    # (4) initialize model
    model = ECAPA_TDNNForSpeechClassification(model_args.n_mfcc, model_args.n_labels, model_args.activation, model_args.final_dropout, model_args.layernorm) #should look like the finetuned model (so using model_args). If pretrained model, will resort to current args
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(sd, strict=False)

    # (6) start evaluating
    preds, targets = eval_loop(model, dataloader_eval, args.exp_dir, args.cloud, args.cloud_dir, args.bucket, args.layer, args.weighted_layers)

    print('Evaluation finished')

def get_embeddings(args):
    """
    Run embedding extraction from start to finish
    :param args: dict with all the argument values
    """
    print('Running Embedding Extraction: ')
    assert args.checkpoint is not None, 'must give a model checkpoint to load for embedding extraction. '
    # Get original 
    model_args, args.finetuned_mdl_path = setup_mdl_args(args)

    # (1) load data to get embeddings for
    assert '.csv' in args.data_split_root, f'A csv file is necessary for embedding extraction. Please make sure this is a full file path: {args.data_split_root}'
    annotations_df = pd.read_csv(args.data_split_root, index_col = 'uid') #data_split_root should use the CURRENT arguments regardless of the finetuned model

    if args.debug:
        annotations_df = annotations_df.iloc[0:8,:]

    # (2) get transforms
    transform = get_transform(args)
    
    # (3) set up dataloaders
    waveform_dataset = WaveformDataset(annotations_df = annotations_df, target_labels = model_args.target_labels, transform = transform) #not super important for embeddings, but the dataset should be selecting targets based on the FINETUNED model
    dataloader = DataLoader(waveform_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # (4) set up embedding model
    model = ECAPA_TDNNForSpeechClassification(model_args.n_mfcc, model_args.n_labels, model_args.activation, model_args.final_dropout, model_args.layernorm) #should look like the finetuned model (so using model_args). If pretrained model, will resort to current args
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(sd, strict=False)
    
    # (5) get embeddings
    embeddings = embedding_loop(model, dataloader, args.embedding_type)
        
    df_embed = pd.DataFrame([[r] for r in embeddings], columns = ['embedding'], index=annotations_df.index)

    try:
        pqt_path = '{}/{}_{}_ecapa_tdnn_embeddings.pqt'.format(args.exp_dir, args.dataset, args.embedding_type)
        
        # if args.finetuned_mdl_path is not None:
        #     args.finetuned_mdl_path = args.finetuned_mdl_path.replace(os.path.commonprefix([args.dataset, os.path.basename(args.finetuned_mdl_path)]), '')
        #     pqt_path = '{}/{}_{}_{}_embeddings.pqt'.format(args.exp_dir, args.dataset, os.path.basename(args.finetuned_mdl_path)[:-3], args.embedding_type) #TODO: can mess with naming conventions later
        # else:
        #     pqt_path = '{}/{}_{}_{}_embeddings.pqt'.format(args.exp_dir, args.dataset, os.path.basename(args.checkpoint), args.embedding_type) #TODO: can mess with naming conventions later
        df_embed.to_parquet(path=pqt_path, index=True, engine='pyarrow') #TODO: fix

        if args.cloud:
            upload(args.cloud_dir, pqt_path, args.bucket)
    except:
        print('Unable to save as pqt, saving instead as csv')
        csv_path = '{}/{}_{_ecapa_tdnn_embeddings.csv'.format(args.exp_dir, args.dataset, args.embedding_type)
        # if args.finetuned_mdl_path is not None:
        #     args.finetuned_mdl_path = args.finetuned_mdl_path.replace(os.path.commonprefix([args.dataset, os.path.basename(args.finetuned_mdl_path)]), '')
        #     csv_path = '{}/{}_{}_{}_embeddings.csv'.format(args.exp_dir, args.dataset, os.path.basename(args.finetuned_mdl_path)[:-3], args.embedding_type)
        # else:
        #     csv_path = '{}/{}_{}_{}_embeddings.csv'.format(args.exp_dir, args.dataset, os.path.basename(args.checkpoint), args.embedding_type)
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
    parser.add_argument('-l','--label_txt', default='./labels.txt')
    parser.add_argument('--lib', default=False, type=bool, help="Specify whether to load using librosa as compared to torch audio")
    parser.add_argument("-c", "--checkpoint", default="gs://ml-e107-phi-shared-aif-us-p/m144443/checkpoints/wav2vec2-base-960h", help="specify path to pre-trained model weight checkpoint")
    #GCS
    parser.add_argument('-b','--bucket_name', default='ml-e107-phi-shared-aif-us-p', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='ml-mps-aif-afdgpet01-p-6827', help='google cloud platform project name')
    parser.add_argument('--cloud', default=True, type=bool, help="Specify whether to save everything to cloud")
    #output
    parser.add_argument("--dataset", default=None,type=str, help="When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root")
    parser.add_argument("-o", "--exp_dir", default="./experiments_weighted/eval", help='specify LOCAL output directory')
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
    parser.add_argument("-pm", "--pooling_mode", default="mean", help="specify method of pooling last hidden layer", choices=['mean','sum','max'])
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
        train(args)

    elif args.mode == 'eval':
        eval_only(args)
              
    elif args.mode == "extraction":
        df_embed = get_embeddings(args)
    
if __name__ == "__main__":
    main()