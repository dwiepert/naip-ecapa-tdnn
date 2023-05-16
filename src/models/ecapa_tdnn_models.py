from speechbrain.lobes.models.ECAPA_TDNN import TDNNBlock, ECAPA_TDNN, Classifier
from utilities import *

class ECAPA_TDNNForSpeechClassification(nn.Module):
    """
    """
    def __init__(self, n_size=80, n_output=6, activation='relu', final_dropout=0.23, layernorm=False):
        super().__init__()
        self.model = ECAPA_TDNN(n_size, lin_neurons=192, device='cuda')

        self.classifier = ClassificationHead(n_size, n_output, activation, final_dropout, layernorm)

    def forward(self, x):
        x = x.squeeze().transpose(1, 2)
        x = self.model(x)
        x = self.classifier(x)
        return torch.squeeze(x, dim=1)
    
    def extract_embedding(self,
                          x,
                          embedding_type="ft"):
        """
        """
        activation = {}
        def _get_activation(name):
            def _hook(model, input, output):
                activation[name] = output.detach()
            return _hook
        
        if embedding_type == 'ft':
            self.classifier.head.dense.register_forward_hook(_get_activation('embeddings'))
            
            logits = self.forward(x)
            e = activation['embeddings']

        elif embedding_type == 'pt':
            e = self.model(x)

        else:
            raise ValueError('Embedding type must be finetune (ft) or pretrain (pt)')
        
        return e