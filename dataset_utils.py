import os
import sys 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import random
from sklearn.model_selection import train_test_split

WORD_SEPPER="  "
DATASET_PATH={"pku":{"train":"/data/dataset/cws/icwb2-data/training/pku_training.utf8",
                    "test":"/data/dataset/cws/icwb2-data/gold/pku_test_gold.utf8"},
              "law":{"train":"/data/dataset/cws/law-20w/train.utf",
                      "test":"/data/dataset/cws/law-20w/gold.utf"}
             }

def init_dir(dir_path):
    if os.path.exist(dir_path):
        print("{} exist!".format(dir_path))
    else:
        os.makedirs(dir_path)
        print("Build {} success!".format(dir_path))

def load_txt(txt_path):
    content = []
    with open(txt_path,"r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                content.append(line.strip())
    return content 

def words_2_bmes(splited_sentence_str):
    word_list = splited_sentence_str.split(WORD_SEPPER)
    labels = []
    valid_word_list = []
    for word in word_list:
        # word = word.strip()
        if len(word)<=0:
            continue 
        else:
            if len(word)==1:
                labels.append("S")
            else:
                labels.append("B"+"M"*(len(word)-2)+"E")
    sentence = "".join(word_list)
    label = "".join(labels)
    assert len(sentence)==len(label)
    return {"label":label, "sentence":sentence}

def bmes_2_words(sentence, label):
    words_list = []
    assert len(sentence)==len(label)
    cur_word = ""
    for c,l in zip(sentence, label):
        if l=="E":
            cur_word += c
            words_list.append(cur_word)
            cur_word = ""
        elif l=="S":
            if cur_word:
                words_list.append(cur_word)
                cur_word = ""
            words_list.append(c)
        elif l=="B":
            if cur_word:
                words_list.append(cur_word)
                cur_word = ""
            cur_word = c 
        elif l=="M":
            cur_word += c 
        else:
            raise "label {} not valid".format(l)
    else:
        if cur_word:
            words_list.append(cur_word)
    return words_list 

class LabelDict:
    def __init__(self, label_str_list=["B","M","E","S"]):
        self.labels = ["PAD"]+label_str_list
        self.label_num = len(self.labels)
        self.label_dict = {item:ind for ind,item in enumerate(self.labels)}
    def id2label(self,_id):
        if _id<0 or _id >= self.label:
            raise "id {} not valid !".format(_id)
        return self.labels[_id]
    def label2id(self,label):
        return self.label_dict.get(label,0)
    def convert_labels2ids(self, labels):
        if type(labels)==str:
            labels = list(labels)
        ids = []
        for label in labels:
            ids.append(self.label2id(label))
        return ids 
    def convert_ids2labels(self, ids):
        labels = []
        for _id in ids:
            labels.append(self.labels[_id])
        return labels

class CWSDataset(Dataset):
    def __init__(self, datalist, tokenizer, label_dict, max_length=500, labeled=True):
        self.datalist = datalist
        self.sample_size = len(datalist)
        self.tokenizer = tokenizer
        self.label_dict = label_dict 
        self.max_length = max_length
        self.labeled=labeled
    def __len__(self):
        return self.sample_size

    def __getitem__(self,ind):
        res = words_2_bmes(self.datalist[ind]) # if it not labeled ,the res label will be "BMMMMM...E"
        input_dict = self.tokenizer.encode_plus(" ".join(list(res["sentence"])), truncation=True, max_length=self.max_length)
        #alingment
        if self.labeled:
            label = label_dict.convert_labels2ids(res["label"])
            label = label[:len(input_dict["input_ids"])-2]
            label = [0]+label+[0]
            assert len(label)==len(input_dict["input_ids"])
            input_dict["label"] = label
        return input_dict

def cws_Collate_fn(inputs):
    max_length = max([ len(item["label"]) for item in inputs])
    sample_num = len(inputs)
    ret_inputs = dict()
    for sample in inputs:
        for k,v in sample.items():
            cur_container = ret_inputs.get(k,[])
            padded_value = v+[0]*(max_length-len(v))
            cur_container.append(padded_value)
            ret_inputs[k] = cur_container
    for k,v in ret_inputs.items():
        ret_inputs[k] = torch.LongTensor(v)
    return ret_inputs

def get_normal_train_dataloader(tokenizer,
                                label_dict,
                                shuffle=True,
                                batch_size = 16,
                                max_length = 500,
                                dataset_name="pku",
                                random_seed=443,
                                used_collate_fn=cws_Collate_fn,
                                train_size = 0.9
                                ):
    train = load_txt(DATASET_PATH[dataset_name]["train"])
    test = load_txt(DATASET_PATH[dataset_name]["test"])
    train, valid = train_test_split(train,random_state=random_seed, train_size=train_size)
    train_dataset = CWSDataset(train,tokenizer, label_dict, max_length)
    valid_dataset = CWSDataset(valid,tokenizer, label_dict, max_length)
    test_dataset = CWSDataset(test,tokenizer, label_dict, max_length)
    ret = {"train":DataLoader(train_dataset,
                                         collate_fn=used_collate_fn,
                                         shuffle=shuffle,
                                         batch_size=batch_size,
                                        ),
            "valid":DataLoader(valid_dataset,
                                         collate_fn=used_collate_fn,
                                         shuffle=False,
                                         batch_size=batch_size,
                                        ),
            "test":DataLoader(test_dataset,
                                         collate_fn=used_collate_fn,
                                         shuffle=False,
                                         batch_size=batch_size,
                                        )
          }
    return ret


label_dict = LabelDict()
