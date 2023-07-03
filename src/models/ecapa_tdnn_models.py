'''
ECAPA-TDNN model classes

Base ECAPA-TDNN model from speech brain

Last modified: 07/2023
Author: Daniela Wiepert, Sampath Gogineni
Email: wiepert.daniela@mayo.edu
File: ecapa_tdnn_models.py
'''

#third-party
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

#local
from utilities import *

class ECAPA_TDNNForSpeechClassification(nn.Module):
    """
    ECAPA-TDNN class for speech feature classification. Wraps a speechbrain ECAPA_TDNN instance.
    Sets up a Classification Head.

    :param n_size: input size to base model
    :param label_dim: specify number of categories to classify - expects either a single number or a list of numbers
    :param lin_neurons: number of neurons in lienar layers, output of model will be (batch size, 1, lin_neurons). Functions as embedding dime
    :param shared_dense: specify whether to add a shared dense layer before classification head
    :param sd_bottleneck: size to reduce to in shared dense layer
    :param clf_bottleneck: size to reduce to in intial classifier dense layer
    :param activation: activation function for classification head
    :param final_dropout: amount of dropout to use in classification head
    :param layernorm: include layer normalization in classification head
    """
    def __init__(self, n_size=80, label_dim=6, lin_neurons=192, shared_dense=False, sd_bottleneck=768,
                 clf_bottleneck=150, activation='relu', final_dropout=0.3, layernorm=False):
        super().__init__()
        self.n_size=n_size
        self.label_dim = label_dim
        self.lin_neurons = lin_neurons
        self.sd_bottleneck=sd_bottleneck
        self.clf_bottleneck=clf_bottleneck

        self.model = ECAPA_TDNN(self.n_size, lin_neurons=self.lin_neurons, device='cuda')

         #adding a shared dense layer
        self.shared_dense = shared_dense
        if self.shared_dense:
            self.dense = nn.Linear(self.lin_neurons, self.sd_bottleneck)
            self.clf_input = self.sd_bottleneck
        else:
            self.clf_input = self.lin_neurons

        #check if a list or a single number
        self.classifiers = []
        if isinstance(self.label_dim, list):
            for dim in self.label_dim:
                self.classifiers.append(ClassificationHead(input_size=self.clf_input, bottleneck=self.clf_bottleneck, output_size=dim,
                                             activation=activation, final_dropout=final_dropout,layernorm=layernorm))
        else:
            self.classifiers.append(ClassificationHead(input_size=self.clf_input, bottleneck=self.clf_bottleneck, output_size=self.label_dim,
                                             activation=activation, final_dropout=final_dropout,layernorm=layernorm))
            
        self.classifiers = nn.ModuleList(self.classifiers)

    def extract_embedding(self, x, embedding_type = 'ft', pooling_mode='mean'):
        """
        Extract an embedding from various parts of the model
        :param x: waveform input (batch size, input size)
        :param embedding_type: 'ft', 'pt' to indicate whether to extract from classification head (ft), base model (pt), or shared dense layer (st)
        :param pooling_mode: method of pooling embeddings if required ("mean" or "sum")
        :return e: embeddings for a batch (batch_size, embedding dim)
        """
        ## EMBEDDING 'ft': extract from finetuned classification head
        if embedding_type == 'ft':
            assert pooling_mode == 'mean' or pooling_mode == 'sum', f"Incompatible pooling given: {pooling_mode}. Please give mean or sum"

            activation = {}
            def _get_activation(name):
                def _hook(model, input, output):
                    activation[name] = output.detach()
                return _hook
            
            x = x.transpose(1, 2)
            x = self.model(x)
            x = torch.squeeze(x, dim=1)

            if self.shared_dense:
                x = self.dense(x)
            
            embeddings = []
            for clf in self.classifiers:
                clf.head.dense.register_forward_hook(_get_activation('embeddings'))
                logits = clf(x)
                embeddings.append(activation['embeddings'])

            embeddings = torch.stack(embeddings, dim=1)
            if pooling_mode == "mean":
                e = torch.mean(embeddings, dim=1)
            else:
                e = torch.sum(embeddings, dim=1)

        elif embedding_type in ['pt', 'st']:
            x = x.transpose(1, 2)
            e = self.model(x)
            e = torch.squeeze(e, dim=1)

            if embedding_type == 'st':
                assert self.shared_dense == True, 'The model must be trained with a shared dense layer'
                e = self.dense(e)

        else:
            raise ValueError('Embedding type must be finetune (ft) or pretrain (pt) or shared dense (st)')

        return e
    
    def forward(self, x):
        """
        Run model
        :param x: input values to the model (batch_size, input_size)
        :return squeezed classifier output (batch_size, num_labels)
        """
        x = x.transpose(1, 2)
        x = self.model(x)
        x = torch.squeeze(x, dim=1)

        if self.shared_dense:
            x = self.dense(x)

        preds = []
        for clf in self.classifiers:
            pred = clf(x)
            preds.append(pred)

        logits = torch.column_stack(preds)
        #torch.squeeze(x, dim=1)
        return logits
    