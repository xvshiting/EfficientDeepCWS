import torch
from segmentor import Segmentor
def calculate_conv1d_kernel_weight_l1(filters:torch.Tensor):
    """ 
       filters: torch.Tensor, shape [output_channels, input_channels, kernel_size]
       return: torch.Tensor, shape [output_channels]
       return the L1 norm of the kernel
    """
    # calculate the weight of conv1d kernel, return L1 norm of the kernel
    assert filters.ndim == 3, "权重张量应为三维 [output_channels, input_channels, kernel_size]"
    l1_scores = torch.sum(torch.abs(filters), dim=-1).mean(dim=1) # mean equal sum here
    return l1_scores

def prune_conv_outchanel_weights(filters, bias, prune_ratio):
    """
    对卷积层权重按 L1 范数进行剪枝，并调整形状。
    
    Args:
        weights (torch.Tensor): 卷积层权重张量，形状为 [output_channels, input_channels, kernel_size].
        prune_ratio (float): 剪枝比例，取值范围为 [0, 1]。

    Returns:
        pruned_weights (torch.Tensor): 剪枝后的权重张量，形状发生改变。
    """
    # 确保输入合法
    assert 0 <= prune_ratio <= 1, "剪枝比例应在 [0, 1] 范围内"
    
    # 计算每个卷积核的 L1 范数，形状为 [output_channels, input_channels]
    l1_scores = calculate_conv1d_kernel_weight_l1(filters)  # 平均输入通道的 L1 分数 
    
    # 根据 L1 范数排序
    sorted_scores, sorted_indices = torch.sort(l1_scores)  # 按 L1 范数从小到大排序
    
    # 确定需要保留的输出通道数
    num_prune = int(prune_ratio * filters.shape[0])  # 要剪掉的输出通道数量
    # print(num_prune)
    keep_indices = sorted_indices[num_prune:]  # 保留的输出通道索引
    
    # 根据保留索引重构权重张量
    pruned_filters = filters[keep_indices, :, :]  # 选择保留的输出通道
    pruned_bias = bias[keep_indices]
    return {"weight":pruned_filters, 
            "bias":pruned_bias, 
            "keep_indices":keep_indices}

def prune_conv_inputchannel_weights(filters, bias, keep_indices):
    """
    剪枝卷积层的输入通道
    filters: 卷积核权重张量, 形状为 (输出通道数, 输入通道数, 卷积核大小)
    keep_indices: 要保留的输入通道索引
    返回: 剪枝后的权重张量和偏置
    """
    pruned_weight = filters[:,keep_indices,:]
    bias = bias
    return {"weight":pruned_weight, 
            "bias":bias,
           }

def prune_cls_layer_weights(weight,bias, keep_indices):
    """
    剪枝全连接层的输入通道
    weight: 全连接层权重张量, 形状为 (输出特征数, 输入特征数)
    keep_indices: 要保留的输入特征索引
    返回: 剪枝后的权重张量和偏置
    """
    pruned_weight = weight[:,keep_indices]
    bias = bias
    return {"weight":pruned_weight, 
            "bias":bias,
           }

def prune_cnnee_model(cnn_model, prune_ratio_list = [0,0,0,0,0,0.5]):
    """
    剪枝CNN模型
    cnn_model: 要剪枝的CNN模型
    prune_ratio_list: 每层卷积层的剪枝比例列表
    返回: 剪枝后的CNN模型
    """
    total_conv_layer_num = len(cnn_model.config.conv1d_cls_out_channel_size_list)
    for ind, prune_ratio in enumerate(prune_ratio_list):
        if ind>= total_conv_layer_num:
            print("prune ratio number {} exceed conv layer num {}".format(len(prune_ratio_list), total_conv_layer_num))
            break
        pruned_ret = prune_conv_outchanel_weights(cnn_model.conv1d_cls_layers[ind].conv1d.weight,
                                     cnn_model.conv1d_cls_layers[ind].conv1d.bias,
                                     prune_ratio
                                    )
        #prune current layer conv 
        cnn_model.conv1d_cls_layers[ind].conv1d.weight = torch.nn.Parameter(pruned_ret["weight"])
        cnn_model.conv1d_cls_layers[ind].conv1d.bias = torch.nn.Parameter(pruned_ret["bias"])
        cnn_model.conv1d_cls_layers[ind].conv1d.in_channels = pruned_ret["weight"].shape[1]
        cnn_model.conv1d_cls_layers[ind].conv1d.out_channels = pruned_ret["weight"].shape[0]
        cnn_model.config.conv1d_cls_out_channel_size_list[ind] = pruned_ret["weight"].shape[0]
        # prune cls
        cls_layer_pruned_ret = prune_cls_layer_weights(cnn_model.conv1d_cls_layers[ind].classifier.weight,
                                cnn_model.conv1d_cls_layers[ind].classifier.bias,
                                pruned_ret["keep_indices"])
        cnn_model.conv1d_cls_layers[ind].classifier.weight = torch.nn.Parameter(cls_layer_pruned_ret["weight"])
        cnn_model.conv1d_cls_layers[ind].classifier.bias = torch.nn.Parameter(cls_layer_pruned_ret["bias"])
        if ind < total_conv_layer_num-1: # not last layer, prune next conv layer input_channel
            pruned_in_ret = prune_conv_inputchannel_weights(cnn_model.conv1d_cls_layers[ind+1].conv1d.weight,
                                                           cnn_model.conv1d_cls_layers[ind+1].conv1d.bias,
                                                           pruned_ret["keep_indices"])
            cnn_model.conv1d_cls_layers[ind+1].conv1d.weight = torch.nn.Parameter(pruned_in_ret["weight"])
            cnn_model.conv1d_cls_layers[ind+1].conv1d.bias = torch.nn.Parameter(pruned_in_ret["bias"])
            cnn_model.conv1d_cls_layers[ind+1].conv1d.in_channels = pruned_in_ret["weight"].shape[1]
            cnn_model.config.conv1d_cls_in_channel_size_list[ind+1] = pruned_in_ret["weight"].shape[1]
        elif ind==total_conv_layer_num-1: # need prune projection layer
            pruned_proj_ret = prune_cls_layer_weights(cnn_model.proj_cls.projection_layer.weight,
                                    cnn_model.proj_cls.projection_layer.bias,
                                    pruned_ret["keep_indices"])
            cnn_model.proj_cls.projection_layer.weight = torch.nn.Parameter(pruned_proj_ret["weight"])
            cnn_model.proj_cls.projection_layer.bias = torch.nn.Parameter(pruned_proj_ret["bias"])
            
    return cnn_model

def main(args):
    seg = Segmentor(args.model_dir)
    cnn_model = seg.model
    cnn_model = prune_cnnee_model(cnn_model, args.prune_ratio_list)
    output_dir = args.output_model_dir
    cnn_model.config.save_pretrained(output_dir)
    seg.tokenizer.save_pretrained(output_dir)
    torch.save({"model_state_dict":cnn_model.state_dict()}, output_dir+"/pytorch_model.bin")
    print("pruned model saved to {}".format(output_dir))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./output/pku_CWSCNNModelWithEE_Phase_2_lr_0.0001_epoch_100_Sun-Jan-19-16:34:06-2025/")
    parser.add_argument("--prune_ratio_list", nargs="+", type=float, default=[0.15,0.40,0.55,0.75,0.70,0.70])
    parser.add_argument("--output_model_dir", type=str, default="./output/pku_pruned_CWSCNNModelWithEE_Phase_2_lr_0.0001_epoch_100_Sun-Jan-19-16:34:06-2025/")
    args = parser.parse_args()
    main(args)


