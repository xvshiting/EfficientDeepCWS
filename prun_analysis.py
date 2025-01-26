import os
import matplotlib.pyplot as plt
from dataset_utils import get_normal_train_dataloader, label_dict
from segmentor import Segmentor
from prune_cnn_model import prune_cnnee_model



def main(args):
    cnn_model_dir = args.cnn_model_dir 
    model_name = args.model_name
    dataset_name = args.dataset_name
    model_path = os.path.join(cnn_model_dir, model_name)

    seg = Segmentor(cnn_model_dir, device="cuda")
    dataiter = get_normal_train_dataloader(seg.tokenizer, label_dict, batch_size=1,dataset_name=dataset_name)
    orig_model_f1 = seg.test(dataiter, thres=0)["f1"]
    analysis_prune_ratio_step =  [r/100 for r in list(range(10,100,10))]
    conv_layer_num = len(seg.config.conv1d_cls_in_channel_size_list)
    prune_f1_list_list = []

    for ind in range(conv_layer_num):
        print("prune layer {}".format(ind+1))
        prune_ratio_list = [0]*conv_layer_num
        prune_f1_list = [orig_model_f1]
        
        for ratio in analysis_prune_ratio_step:
            seg = Segmentor(cnn_model_dir, device="cuda")
            print("prune ratio {}".format(ratio))
            prune_ratio_list[ind] = ratio
            prune_cnnee_model(seg.model, prune_ratio_list)
            ret = seg.test(dataiter, thres=0, verbose=True)
            prune_f1_list.append(ret["f1"])
        prune_f1_list_list.append(prune_f1_list)
    #plot prune_f1_list_list with analysis_prune_ratio_step
    analysis_prune_ratio_step = [0] + analysis_prune_ratio_step
    #multiply 100 
    analysis_prune_ratio_step = [x*100 for x in analysis_prune_ratio_step]
    # plot with dot line
    # improve pixels
    plt.figure(figsize=(10, 10))

    for ind in range(conv_layer_num):
        line_style = "--" if ind%2==0 else "-"
        plt.plot(analysis_prune_ratio_step, prune_f1_list_list[ind], label="conv{}".format(ind+1), linestyle=line_style,  marker="o")
    # y-label f1-score
    plt.ylabel("f1-score")
    # x-label prune ratio
    plt.xlabel("prune ratio(%)")
    
    plt.legend()
    plt.show()
    #save as pdf
    plt.savefig(args.plot_save_path)
    return prune_f1_list_list


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn_model_dir", type=str, default="./output/pku_CWSCNNModelWithEE_Phase_2_lr_0.0001_epoch_100_Sun-Jan-19-16:34:06-2025/" )#cnn_model_dir
    parser.add_argument("--model_name", type=str, default="checkpoint_best.pt")
    # parser.add_argument("--analysis_prune_ratio_step", type=list, default=[r/100 for r in list(range(10,100,5))])
    parser.add_argument("--dataset_name", type=str, default="pku")
    #plot save path
    parser.add_argument("--plot_save_path", type=str, default="./output/pku_CWSCNNModelWithEE_Phase_2_lr_0.0001_epoch_100_Sun-Jan-19-16:34:06-2025/prune_analysis.pdf")
    args = parser.parse_args()
    main(args)