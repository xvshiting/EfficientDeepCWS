"""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1
"""
from segmentor import Segmentor
from dataset_utils import get_normal_train_dataloader, label_dict
import numpy as np
import time
from tqdm import tqdm

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

def main(args):
    seg = Segmentor(args.model_dir)
    dataiter = get_normal_train_dataloader(seg.tokenizer, label_dict, batch_size=1,dataset_name=args.dataname)
    time_cost_list = []
    char_num_list = []
    for d in tqdm(dataiter["test"]):
        char_num_list.append(d["input_ids"].shape[1])
        if seg.config.model_name.startswith("CWSCNNModelWithEE"):
            start = time.time()
            ret = seg.model.predict(d["input_ids"], thres=args.thres)
            end = time.time()
            time_cost_list.append(end-start)
        else:
            start = time.time()
            ret = seg.model(**d)
            end = time.time()
            time_cost_list.append(end-start)

    print(np.mean(time_cost_list))
    print(np.mean(char_num_list))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--dataname", type=str, default="pku")
    parser.add_argument("--thres",type=float, default=0)
    args = parser.parse_args()
    main(args)