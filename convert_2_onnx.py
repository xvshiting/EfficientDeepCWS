import torch
import torch.nn as nn
import torch.onnx
from segmentor import Segmentor
from cws_models import CWSCNNModelWithEE
import os 
from file_utils import init_dir

def CNNEEconver2onnx(model:CWSCNNModelWithEE, output_onnx_dir:str):
    # 示例输入（FloatTensor）
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # 转换为 LongTensor
    input_tensor = input_tensor.long()
    #convert embedding layer
    torch.onnx.export(
    model.embedding,
    input_tensor,
    os.path.join(output_onnx_dir,"model_embedding.onnx"),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_length"},  # 允许输入序列长度 (1) 和批量大小 (0) 动态
        "output": {0: "batch_size",1: "seq_length"}                  # 输出的批量大小动态
    },
    opset_version=11  # 使用 ONNX 的 opset 版本
)   
    input_tensor = model.embedding(input_tensor)
    for ind,layer in enumerate(model.conv1d_cls_layers):
        torch.onnx.export(
    layer,
    input_tensor,
    os.path.join(output_onnx_dir,"conv_{}.onnx".format(ind+1)),
    input_names=["input"],
    output_names=["output","logits"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_length"},  # 允许输入序列长度 (1) 和批量大小 (0) 动态
        "output": {0: "batch_size",1: "seq_length"},                  # 输出的批量大小动态
        "logits": {0: "batch_size",1: "seq_length"} 
    },
    opset_version=11  # 使用 ONNX 的 opset 版本
)   
        input_tensor, logits = layer(input_tensor)
    torch.onnx.export(
    model.proj_cls,
    input_tensor,
    os.path.join(output_onnx_dir,"proj_cls.onnx".format(ind+1)),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_length"},  # 允许输入序列长度 (1) 和批量大小 (0) 动态
        "output": {0: "batch_size",1: "seq_length"}                  # 输出的批量大小动态
    },
    opset_version=11  # 使用 ONNX 的 opset 版本
)
    model.config.model_name = model.config.model_name+"_onnx"
    model.config.save_pretrained(output_onnx_dir)
    

    

def main(args):
    seg = Segmentor(args.model_dir, args.model_name)
    init_dir(args.output_onnx_dir)
    CNNEEconver2onnx(seg.model,args.output_onnx_dir)
    seg.tokenizer.save_pretrained(args.output_onnx_dir)

    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./output/pku_pruned_CWSCNNModelWithEE_Phase_2_lr_0.0001_epoch_100_Sun-Jan-19-16:34:06-2025")
    parser.add_argument("--model_name",type=str, default="pytorch_model.bin")
    parser.add_argument("--output_onnx_dir", type=str, default="./output/onnx/pku_pruned")
    
    args = parser.parse_args()
    main(args)
