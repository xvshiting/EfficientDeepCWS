from transformers import  BertModel
from torch import nn
from transformers import PretrainedConfig
import torch

class CWSRoberta(nn.Module):
    def __init__(self,
                 config
                 ):
        super( CWSRoberta,self).__init__()
        self.config = config
        self.encoder = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # self.activation = nn.ReLU()
        self.classifier = nn.Linear(self.config.cls_hidden_dim, self.config.label_num)
        
    def forward(self, input_ids, attention_mask,token_type_ids,ret_h=False, **kwargs):
        encoder_output = self.encoder(input_ids = input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids)
        h = encoder_output.last_hidden_state
        cls_input = self.dropout(h)
        # encoder_output = self.activation(encoder_output)
        logits = self.classifier(cls_input)
        if ret_h:
            return h, logits
        return logits
    
class CWSCNNModelWithEEConfig(PretrainedConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dim = kwargs['embedding_dim'] if 'embedding_dim' in kwargs else 300
        self.hidden_size = kwargs['hidden_size'] if 'hidden_size' in kwargs else 300
        # self.conv1d_kernel = kwargs['conv1d_kernel'] if 'conv1d_kernel' in kwargs else [3]*6 
        self.label_num = kwargs['label_num'] if 'label_num' in kwargs else 5
        self.proj_hidden_dim = kwargs['proj_hidden_dim'] if 'proj_hidden_dim' in kwargs else 768
        self.dropout_ratio = kwargs['dropout_ratio'] if 'dropout_ratio' in kwargs else 0.1
        self.activation = kwargs['activation'] if 'activation' in kwargs else 'relu'
        self.ee = kwargs['ee'] if 'ee' in kwargs else True
        self.conv1d_kernel = kwargs['conv1d_kernel'] if 'conv1d_kernel' in kwargs else 3
        self.conv1d_cls_layer_num = kwargs['conv1d_cls_layer_num'] if 'conv1d_cls_layer_num' in kwargs else 6

class ConvClsLayer(nn.Module):
    def __init__(self, config,  **kwargs):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=config.embedding_dim,
                                out_channels=config.hidden_size,
                                kernel_size=config.conv1d_kernel,
                                padding="same")
        self.classifier = nn.Linear(config.hidden_size, config.label_num)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv1d(x)
        hidden_x = x.permute(0,2,1)
        hidden_x = self.activation(hidden_x)
        logits = self.classifier(self.dropout(hidden_x))
        return hidden_x, logits

class ProjectClsLayer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(self.config.proj_hidden_dim, self.config.label_num)
        self.dropout = nn.Dropout(self.config.dropout_ratio) 
        self.projection_layer = nn.Linear(self.config.hidden_size, self.config.proj_hidden_dim)
    
    def forward(self, x):
        hidden_x = self.projection_layer(x)
        logits = self.classifier(self.dropout(hidden_x))
        return hidden_x, logits

class CWSCNNModelWithEE(nn.Module): 
    def __init__(self, config:CWSCNNModelWithEEConfig, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.config = config
        self.conv1d_cls_layers = nn.ModuleList()
        self.classifier_list = nn.ModuleList()
        self.embedding = nn.Embedding(21128, self.config.embedding_dim)
        self.proj_cls = ProjectClsLayer(self.config)
        self.dropout = nn.Dropout(self.config.dropout_ratio)
        self.activation = nn.ReLU()
        for i in range(self.config.conv1d_cls_layer_num): 
            self.conv1d_cls_layers.append(ConvClsLayer(config))
        
    def forward( self, input_ids:torch.Tensor,**kwargs) -> torch.Tensor:
        input_x = self.embedding(input_ids)
        logits_list = []
        for ind, layer in enumerate(self.conv1d_cls_layers):
            input_x, logits = layer(input_x)
            input_x = self.dropout(input_x)
            logits_list.append(logits)
        hidden_x, logits = self.proj_cls(input_x)
        logits_list.append(logits)
        return {"hidden_x":hidden_x, 
                "logits_list":logits_list}
    

class CWSCNNModel(nn.Module): 
    def __init__(self, config:CWSCNNModelWithEEConfig, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.config = config
        self.Cov1d_list = nn.ModuleList()
        self.embedding = nn.Embedding(21128, self.config.embedding_dim)
        
        self.dropout = nn.Dropout(self.config.dropout_ratio)
        self.activation = nn.ReLU()
        for i in range(self.config.conv1d_cls_layer_num): 
            # we need padding them according kernel 3, each side padding 1
            self.Cov1d_list.append(nn.Conv1d(in_channels=self.config.embedding_dim,
                                             out_channels = self.config.hidden_size,
                                             kernel_size = self.config.conv1d_kernel,
                                             padding="same"
                                             ))
        self.proj_cls = ProjectClsLayer(config)
        
    def forward( self, input_ids:torch.Tensor,**kwargs) -> torch.Tensor:
        input_x = self.embedding(input_ids) #B,L,C
        #transfer to B, C ,L
        input_x = input_x.permute(0,2,1)
        for ind, layer in enumerate(self.Cov1d_list):
            input_x = layer(input_x)
            input_x = self.activation(input_x)
            input_x = self.dropout(input_x)
        input_x = input_x.permute(0,2,1)
        hidden_x, logits = self.proj_cls(input_x)
        return logits 

model_cls_dict = {"CWSCNNModel":CWSCNNModel,
                  "CWSCNNModelWithEE":CWSCNNModelWithEE,
                  "CWSRoberta":CWSRoberta}







        





        




    


    
