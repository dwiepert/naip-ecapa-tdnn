"""
Model loops for training and extracting embeddings from ECAPA-TDNN model

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: loops.py
"""
#IMPORTS
#built-in
import json
import os

#third party
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

#local
from utilities import *

def train(model, dataloader_train, dataloader_val = None, 
          optim='adamw', learning_rate=0.001, loss_fn='BCE',
          sched='onecycle', max_lr=0.01,
          epochs=10, exp_dir='', cloud=False, cloud_dir='', bucket=None):
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
    if loss_fn == 'MSE':
        criterion = torch.nn.MSELoss()
    elif loss_fn == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError('MSE must be given for loss parameter')
    #optimizer
    if optim == 'adam':
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
    elif optim == 'adamw':
         optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    else:
        raise ValueError('adam must be given for optimizer parameter')
    
    if sched == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=max_lr, steps_per_epoch=len(dataloader_train), epochs=epochs)
    else:
        scheduler = None
    
    #train
    for e in range(epochs):
        training_loss = list()
        #t0 = time.time()
        model.train()
        for batch in tqdm(dataloader_train):
            x = batch['mfcc']
            targets = batch['targets']
            x, targets = x.to(device), targets.to(device)
            optimizer.zero_grad()
            o = model(x)
            loss = criterion(o, targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_item = loss.item()
            training_loss.append(loss_item)

        if e % 10 == 0:
            #SET UP LOGS
            if scheduler is not None:
                lr = scheduler.get_last_lr()
            else:
                lr = learning_rate
            logs = {'epoch': e, 'optim':optim, 'loss_fn': loss_fn, 'lr': lr}
    
            logs['training_loss_list'] = training_loss
            training_loss = np.array(training_loss)
            logs['running_loss'] = np.sum(training_loss)
            logs['training_loss'] = np.mean(training_loss)

            print('RUNNING LOSS', e, np.sum(training_loss) )
            print(f'Training loss: {np.mean(training_loss)}')

            if dataloader_val is not None:
                print("Validation start")
                validation_loss = validation(model, criterion, dataloader_val)

                logs['val_loss_list'] = validation_loss
                validation_loss = np.array(validation_loss)
                logs['val_running_loss'] = np.sum(validation_loss)
                logs['val_loss'] = np.mean(validation_loss)
                
                print('RUNNING VALIDATION LOSS',e, np.sum(validation_loss) )
                print(f'Validation loss: {np.mean(validation_loss)}')
            
            #SAVE LOGS
            json_string = json.dumps(logs)
            logs_path = os.path.join(exp_dir, 'logs_epoch{}.json'.format(e))
            with open(logs_path, 'w') as outfile:
                json.dump(json_string, outfile)
            
    return model


def validation(model, criterion, dataloader_val):
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

def evaluation(model, dataloader_eval, exp_dir, cloud=False, cloud_dir=None, bucket=None):
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

def embedding_extraction(model, dataloader, embedding_type='ft'):
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