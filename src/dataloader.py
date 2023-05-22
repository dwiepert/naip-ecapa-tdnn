# -*- coding: utf-8 -*-
'''
Dataset functions for ECAPA-TDNN

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: dataloader.py
'''

#IMPORTS
#third party
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional
from torch.utils.data import Dataset

#local
from utilities import *
from models import *

class ECAPA_TDNNDataset(Dataset):
    def __init__(self, annotations_df, target_labels, 
                 audio_conf, prefix, bucket=None, librosa=False):
        '''
        Dataset that manages audio recordings. 

        :param annotations_df: dataframe containing uid of audio file along with labels
        :type annotations_df: pd.DataFrame
        :param target_labels: list containing the specific target labels to select
        :type target_labels: List[Str]
        :param audio_conf: dictionary containing all information for transforms (audio configuration dict)
        :type audio_conf: dict
        :param prefix: location of files to download - can be either prefix in gcs bucket or input directory
        :type prefix: Str
        :param bucket: Google cloud storage bucket, default=None
        :type bucket: GCS bucket
        :param librosa: boolean indicating loading from librosa as compared to torchaudio
        :type librosa: boolean
        '''

        #set instance variables
        self.annotations_df = annotations_df
        self.target_labels = target_labels
        if self.target_labels is None:
            self.target_labels = np.array([])
        self.prefix = prefix
        self.bucket = bucket
        self.lib = librosa #set up using librosa vs. torchaudio for loading

        #AUDIO CONFIGURATION PARAMETERS
        self.audio_conf = audio_conf
        self.trained_mdl_path= self.audio_conf.get('trained_mdl_path')
        ### AUDIO TRANSFORMATIONS
        self.resample_rate = self.audio_conf.get('resample_rate') #resample if resample rate != 0 and if resample rate != sample rate
        self.reduce = self.audio_conf.get('reduce') #reduce to monochannel if True
        self.trim = self.audio_conf.get('trim') #trim silence if True
        self.clip_length = self.audio_conf.get('clip_length') #truncate clip to specified length if != 0
        ### SPECTROGRAM TRANSFORMATIONS
        self.n_mfcc = self.audio_conf.get('n_mfcc')
        self.n_fft = self.audio_conf.get('n_fft')
        self.n_mels = self.audio_conf.get('n_mels')
        self.fbank = self.audio_conf.get('fbank')
        if self.fbank:
            self.freqm = self.audio_conf.get('freqm') #frequency masking if freqm != 0
            self.timem = self.audio_conf.get('timem') #time masking if timem != 0
            self.norm_mean = self.audio_conf.get('mean')
            self.norm_std = self.audio_conf.get('std')
            ## skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
            # set it as True ONLY when you are getting the normalization stats.
            self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
            if self.skip_norm:
                print('now skip normalization (use it ONLY when you are computing the normalization stats).')
            else:
                print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
            ## if add noise for data augmentation
            self.noise = self.audio_conf.get('noise')
            if self.noise == True:
                print('now use noise augmentation')


        self.label_dim = len(self.target_labels)
        print('number of classes is {:d}'.format(self.label_dim))

        self.audio_transform = self._getaudiotransform() #get audio transforms
        self.spec_transform = self._getspectransform()

    def _getaudiotransform(self):
        '''
        Use audio configuration parameters to initialize classes for audio transformation. 
        Outputs two tranform variables, one for regular audio transformation and one for 
        augmentations using albumentations

        These transformations will always load the audio. 
        :outparam audio_transform: standard transforms
        '''
        waveform_loader = UidToWaveform(prefix = self.prefix, bucket=self.bucket, lib=self.lib)
        transform_list = [waveform_loader]
        if self.reduce:
            channel_sum = lambda w: torch.sum(w, axis = 0).unsqueeze(0)
            mono_tfm = ToMonophonic(reduce_fn = channel_sum)
            transform_list.append(mono_tfm)
        if self.resample_rate != 0: #16000
            downsample_tfm = Resample(self.resample_rate)
            transform_list.append(downsample_tfm)
        if self.trim:
            trim_tfm = TrimSilence()
            transform_list.append(trim_tfm)
        if self.clip_length != 0: #160000
            truncate_tfm = Truncate(length = self.clip_length)
            transform_list.append(truncate_tfm)

        tensor_tfm = ToTensor()
        transform_list.append(tensor_tfm)
        transform = torchvision.transforms.Compose(transform_list)
        return transform
    
    def _getspectransform(self):
        '''
        Use audio configuration parameters to initialize classes for spectrogram transformation. 
        Outputs one tranform variable. Will always generate the spectrogram, and has options 
        for frequency/time masking, normalization, and adding noise

        :outparam transform: spectrogram transforms
        '''
        if self.fbank:
            wav2bank = Wav2Fbank(self.n_fft, self.n_mels, None, None, override_wave=False) #override waveform so final sample does not contain the waveform - doing so because the waveforms are not the same shape
            transform_list = [wav2bank]
            if self.freqm != 0:
                freqm = FreqMask(self.freqm)
                transform_list.append(freqm)
            if self.timem != 0: 
                timem = TimeMask(self.timem)
                transform_list.append(timem)
            if not self.skip_norm:
                norm = Normalize(self.norm_mean, self.norm_std)
                transform_list.append(norm)
            if self.noise:
                #TODO:
                noise = Noise()
                transform_list.append(noise)
            #add freq mask and stuff
        
        else:
            mfcc_tfm = MFCC(n_mfcc= self.n_mfcc, n_fft=self.n_fft, n_mels=self.n_mels)
            transform_list = [mfcc_tfm]
        
        transform = torchvision.transforms.Compose(transform_list)
        return transform
    
    def __getitem__(self, idx):
        '''
        Given an index, load and run transformations then return the sample dictionary

        Will run transformations in this order:
        Standard audio transformations (load audio -> reduce channels -> resample -> clip -> subtract mean) - also convert labels to tensor
        Albumentation transformations (Time shift -> speed tune -> add gauss noise -> pitch shift -> alter gain -> stretch audio)
        Spectrogram transformations (convert to spectrogram -> frequency mask -> time mask -> normalize -> add noise)

        The resulting sample dictionary contains the following info
        'uid': audio identifier
        'waveform': audio (n_channels, n_frames)
        'fbank': spectrogram (target_length, frequency_bins)
        'sample_rate': current sample rate
        'targets': labels for current file as tensor

        '''
    
        #If not doing mix-up
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        uid = self.annotations_df.index[idx] #get uid to load
        targets = self.annotations_df[self.target_labels].iloc[idx].values #get target labels for given uid
        
        sample = {
            'uid' : uid,
            'targets' : targets
        }
        
        sample = self.audio_transform(sample) #load and perform standard transformation
        sample = self.spec_transform(sample) #spectrogram transforms

        if 'fbank' in sample:
            x = sample['fbank']
            x = x.unsqueeze(0)
            x = x.transpose(1, 2)
            sample['spec'] = x
            del sample['fbank']
        
        if 'mfcc' in sample:
            sample['spec'] = sample['mfcc']
            del sample['mfcc']
        return sample
    

    def __len__(self):
        return len(self.annotations_df)