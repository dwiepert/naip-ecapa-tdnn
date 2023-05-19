'''
ECAPA-TDNN model classes

Base ECAPA-TDNN model from speech brain

Last modified: 05/2023
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
    :param label_dim: specify number of categories to classify
    :param lin_neurons: number of neurons in lienar layers, output of model will be (batch size, 1, lin_neurons). Functions as embedding dime
    :param activation: activation function for classification head
    :param final_dropout: amount of dropout to use in classification head
    :param layernorm: include layer normalization in classification head
    """
    def __init__(self, n_size=80, label_dim=6, lin_neurons=192,
                  activation='relu', final_dropout=0.23, layernorm=False):
        super().__init__()
        self.n_size=n_size
        self.label_dim = label_dim
        self.lin_neurons = lin_neurons

        self.model = ECAPA_TDNN(self.n_size, lin_neurons=self.lin_neurons, device='cuda')

        self.classifier = ClassificationHead(self.lin_neurons, self.label_dim, 
                                             activation, final_dropout, layernorm)

    def extract_embedding(self, x, embedding_type = 'ft'):
        """
        Extract an embedding from various parts of the model
        :param x: waveform input (batch size, input size)
        :param embedding_type: 'ft', 'pt' to indicate whether to extract from classification head (ft), base model (pt)
        :return e: embeddings for a batch (batch_size, embedding dim)
        """
        if embedding_type == 'ft':
            activation = {}
            def _get_activation(name):
                def _hook(model, input, output):
                    activation[name] = output.detach()
                return _hook
            
            self.classifier.head.dense.register_forward_hook(_get_activation('embeddings'))
            
            logits = self.forward(x)
            e = activation['embeddings']

        elif embedding_type == 'pt':
            x = torch.squeeze(x, dim=1)
            x = x.transpose(1, 2)
            e = self.model(x)
            e = torch.squeeze(e, dim=1)

        else:
            raise ValueError('Embedding type must be finetune (ft) or pretrain (pt)')

        return e
    
    def forward(self, x):
        """
        Run model
        :param x: input values to the model (batch_size, input_size)
        :return squeezed classifier output (batch_size, num_labels)
        """
        x = torch.squeeze(x, dim=1)
        x = x.transpose(1, 2)
        x = self.model(x)
        x = torch.squeeze(x, dim=1)
        x = self.classifier(x)
        return torch.squeeze(x, dim=1)
    