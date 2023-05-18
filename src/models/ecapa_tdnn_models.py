'''
ECAPA-TDNN model classes

Base ECAPA-TDNN model from speech brain

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: ecapa_tdnn_models.py
'''

#third-party
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

#local
from utilities import *

class ECAPA_TDNNForSpeechClassification(nn.Module):
    """
    """
    def __init__(self, n_size=80, label_dim=6, lin_neurons=192,
                  activation='relu', final_dropout=0.23, layernorm=False):
        super().__init__()
        self.n_size=n_size
        self.label_dim = label_dim
        self.lin_neurons = lin_neurons

        self.model = ECAPA_TDNN(self.n_size, lin_neurons=self.lin_neurons, device='cuda')

        self.classifier = ClassificationHead(self.n_size, self.label_dim, 
                                             activation, final_dropout, layernorm)

    def extract_embedding(self, x, embedding_type = 'ft'):
        """
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
            x = x.squeeze().transpose(1, 2)
            e = self.model(x)

        else:
            raise ValueError('Embedding type must be finetune (ft) or pretrain (pt)')

        return e
    
    def forward(self, x):
        """
        """
        x = x.squeeze().transpose(1, 2)
        x = self.model(x)
        x = self.classifier(x)
        return torch.squeeze(x, dim=1)
    