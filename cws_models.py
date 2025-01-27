from transformers import  BertModel
from torch import nn
from transformers import PretrainedConfig
import torch
import os 
import onnxruntime as ort
import numpy as np 
import re

def get_entropy_np(x):
    # x: np.ndarray, logits BEFORE softmax
    exp_x = np.exp(x)
    A = np.sum(exp_x, axis=2)    # sum of exp(x_i)
    B = np.sum(x*exp_x, axis=2)  # sum of x_i * exp(x_i)
    return np.log(A) - B/A

def get_uncertainty_np(x):
    x = x[:,:,1:-1]
    # x: np.ndarray, logits BEFORE softmax
    num_tags = x.shape[-1]
    entropy = get_entropy_np(x)
    return entropy / np.log(num_tags)

def get_entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=2)    # sum of exp(x_i)
    B = torch.sum(x*exp_x, dim=2)  # sum of x_i * exp(x_i)
    return torch.log(A) - B/A

import math
def get_uncertainty(x):
    x = x[:,:,1:-1]
    # x: torch.Tensor, logits BEFORE softmax
    num_tags = x.size(-1)
    entropy_x = get_entropy(x)
    return entropy_x/math.log(num_tags)

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
        self.max_position_embeddings = kwargs['max_position_embeddings'] if 'max_position_embeddings' in kwargs else 512
        self.hidden_size = kwargs['hidden_size'] if 'hidden_size' in kwargs else 300
        # self.conv1d_kernel = kwargs['conv1d_kernel'] if 'conv1d_kernel' in kwargs else [3]*6 
        self.label_num = kwargs['label_num'] if 'label_num' in kwargs else 5
        self.proj_hidden_dim = kwargs['proj_hidden_dim'] if 'proj_hidden_dim' in kwargs else 768
        self.dropout_ratio = kwargs['dropout_ratio'] if 'dropout_ratio' in kwargs else 0.1
        self.activation = kwargs['activation'] if 'activation' in kwargs else 'relu'
        self.ee = kwargs['ee'] if 'ee' in kwargs else True
        self.conv1d_kernel = kwargs['conv1d_kernel'] if 'conv1d_kernel' in kwargs else 3
        self.conv1d_cls_layer_num = kwargs['conv1d_cls_layer_num'] if 'conv1d_cls_layer_num' in kwargs else 6
        self.conv1d_cls_in_channel_size_list = [300]*self.conv1d_cls_layer_num # for prune
        self.conv1d_cls_out_channel_size_list = [300]*self.conv1d_cls_layer_num # for prune

class ConvClsLayer(nn.Module):
    def __init__(self, config, in_channel_size, out_channel_size,  **kwargs):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channel_size,
                                out_channels=out_channel_size,
                                kernel_size=config.conv1d_kernel,
                                padding="same")
        self.classifier = nn.Linear(out_channel_size, config.label_num)
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
    def __init__(self, config, input_hidden_size, **kwargs):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(self.config.proj_hidden_dim, self.config.label_num)
        self.dropout = nn.Dropout(self.config.dropout_ratio) 
        self.projection_layer = nn.Linear(input_hidden_size, self.config.proj_hidden_dim)
    
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
        # self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.embedding_dim)
        self.proj_cls = ProjectClsLayer(self.config,input_hidden_size=config.conv1d_cls_out_channel_size_list[-1])
        self.dropout = nn.Dropout(self.config.dropout_ratio)
        self.activation = nn.ReLU()
        # self.register_buffer(
            # "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        # )
        for i in range(self.config.conv1d_cls_layer_num): 
            self.conv1d_cls_layers.append(ConvClsLayer(config,
                                                       in_channel_size=config.conv1d_cls_in_channel_size_list[i],
                                                       out_channel_size=config.conv1d_cls_out_channel_size_list[i],
                                                       ))
        
    def forward( self, input_ids:torch.Tensor,**kwargs) -> torch.Tensor:
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        # position_ids = self.position_ids[:, : seq_length ]
        # position_embeddings = self.position_embeddings(position_ids)
        input_x = self.embedding(input_ids)
        # +position_embeddings
        logits_list = []
        for ind, layer in enumerate(self.conv1d_cls_layers):
            input_x, logits = layer(input_x)
            input_x = self.dropout(input_x)
            logits_list.append(logits)
        hidden_x, logits = self.proj_cls(input_x)
        logits_list.append(logits)
        return {"hidden_x":hidden_x, 
                "logits_list":logits_list}
    
    def predict(self, input_ids:torch.Tensor,
                thres=0.5,
                force_layer = None,
                **kwargs):
        """
        force_layer:0,1,2,3,4,5,6
        """
        logits = None 
        uncertain_score_list = []
        logits_list = []
        flops = 0
        is_force = False
        if force_layer is not None:
            force_layer = max(0,min(force_layer,6))
         #early exit with uncertainty
        input_x = self.embedding(input_ids) 
        batch_size, seq_length, embedding_dim = input_x.size()
        flops += batch_size * seq_length * embedding_dim  # FLOPs for embedding lookup
        for ind, layer in enumerate(self.conv1d_cls_layers):
            if force_layer is not None and force_layer<ind: # 0<1 exit, 0=0 continue, 2>1 continue. 
                is_force=True
                break
            input_x, logits = layer(input_x)
            logits_list.append(logits)
            # p = torch.softmax(logits[:,0:-1,1:],  dim=-1) #1*len *C
            # U_s = -p*torch.log(p)/torch.log(torch.LongTensor(4))
            # U_s.sum(dim=-1)
            # print(logits.shape)
            # U_s.max(dim=1)
            # FLOPs for ConvClsLayer
            in_channels = layer.conv1d.in_channels
            out_channels = layer.conv1d.out_channels
            kernel_size = layer.conv1d.kernel_size[0]
            output_length = input_x.size(1)
            flops += in_channels * out_channels * kernel_size * output_length * batch_size
            flops += out_channels * 5 *  batch_size
            flops += logits.numel() * 5 
            U_s = get_uncertainty(logits)
            # print(U_s)
            # # print(U_s.shape)
            U_s = torch.max(U_s[:,1:-1]) #omit cls and sep
            uncertain_score_list.append(U_s)
            if force_layer is None and U_s<thres:
                break
        else:
            hidden_x, logits = self.proj_cls(input_x)
            logits_list.append(logits)
            ind = 6
            # FLOPs for final ProjectClsLayer
            input_features = hidden_x.size(-1)
            output_features = logits.size(-1)
            flops += input_features * output_features * batch_size
            flops += output_features * 5 *batch_size
        return {"logits":logits,
                "exit_layer":ind, 
                "uncertainty_score":uncertain_score_list,
                "is_force":is_force,
                "logits_list":logits_list,
                "flops":flops
        }            
    

class CWSCNNModel(nn.Module): 
    def __init__(self, config:CWSCNNModelWithEEConfig, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.config = config
        self.Cov1d_list = nn.ModuleList()
        self.embedding = nn.Embedding(21128, self.config.embedding_dim)
        # self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.embedding_dim)
        self.dropout = nn.Dropout(self.config.dropout_ratio)
        self.activation = nn.ReLU()
        # self.register_buffer(
        #     "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        # )
        for i in range(self.config.conv1d_cls_layer_num): 
            # we need padding them according kernel 3, each side padding 1
            self.Cov1d_list.append(nn.Conv1d(in_channels=self.config.embedding_dim,
                                             out_channels = self.config.hidden_size,
                                             kernel_size = self.config.conv1d_kernel,
                                             padding="same"
                                             ))
        self.proj_cls = ProjectClsLayer(config,input_hidden_size=self.config.hidden_size)
        
    def forward( self, input_ids:torch.Tensor,**kwargs) -> torch.Tensor:
        input_shape = input_ids.size()
        # print(input_ids.shape)
        seq_length = input_shape[1]
        # position_ids = self.position_ids[:, : seq_length ]
        # print(position_ids.shape)
        # position_embeddings = self.position_embeddings(position_ids)
        input_x = self.embedding(input_ids)
        # +position_embeddings
        #transfer to B, C ,L
        input_x = input_x.permute(0,2,1)
        for ind, layer in enumerate(self.Cov1d_list):
            input_x = layer(input_x)
            input_x = self.activation(input_x)
            input_x = self.dropout(input_x)
        input_x = input_x.permute(0,2,1)
        hidden_x, logits = self.proj_cls(input_x)
        return logits


class CWSCNNModelWithEE_onnx():
    def __init__(self, model_path, 
                 device="cpu",
                  max_length=500, 
                  ):
        self.model_path = model_path
        self.load_onnx_model(model_path)

    
    def load_onnx_model(self, model_dir):
        onnx_embedding_model_path = os.path.join(model_dir, "model_embedding.onnx") 
        self.embedding_ort_session = ort.InferenceSession(onnx_embedding_model_path)
        self.conv_ort_session_list = []
        file_name_list = os.listdir(model_dir)
        conv_file_name = [file_name  for file_name in file_name_list if file_name.startswith("conv") and file_name.endswith(".onnx")]
        #sort conv_file_name conv_1.onnx conv_2.onnx 
        conv_file_name.sort(key=lambda x: int(re.findall(r"\d+",x)[0]))
        for conv_file in conv_file_name:
            conv_ort_session = ort.InferenceSession(os.path.join(model_dir, conv_file))
            self.conv_ort_session_list.append(conv_ort_session)
        self.proj_cls_ort_session = ort.InferenceSession(os.path.join(model_dir, "proj_cls.onnx"))

    def predict(self, input_ids, thres=0.55, force_layer=None, **kwargs):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        input_ids = np.array(input_ids, dtype=np.int64)
        input_dict = {"input":input_ids}
        embedding_output = self.embedding_ort_session.run(None, input_dict)[0]
        conv_output = embedding_output
        logits  = None 
        uncertain_score_list = []
        logits_list = []
        if force_layer:
            is_force = True
        else:
            is_force = False
        for ind, conv_ort_session in enumerate(self.conv_ort_session_list):
            if force_layer is not None and ind == force_layer:
                break
            conv_output = conv_ort_session.run(None, {"input":conv_output})
            conv_output, conv_logits  = conv_output 
            logits = conv_logits
            logits_list.append(logits)
            #calculate uncertainty
            u_s = get_uncertainty_np(conv_logits)
            uncertain_score_list.append(u_s)
            u_s = np.max(u_s[:,1:-1])
            if force_layer is None and u_s < thres:
                break
            
        else:
            proj_cls_output = self.proj_cls_ort_session.run(None, {"input":conv_output})
            hidden_x, logits = proj_cls_output
            logits_list.append(logits)
        return {
            "logits":torch.tensor(logits),
            "exit_layer":ind, 
            "uncertainty_score":uncertain_score_list,
            "is_force":is_force,
            "logits_list":logits_list
            }

       


model_cls_dict = {"CWSCNNModel":CWSCNNModel,
                  "CWSCNNModelWithEE":CWSCNNModelWithEE,
                  "CWSRoberta":CWSRoberta,
                  "CWSCNNModelWithEE_onnx":CWSCNNModelWithEE_onnx}



    





        





        




    


    
