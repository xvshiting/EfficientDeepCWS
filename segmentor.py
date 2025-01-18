from dataset_utils import get_normal_train_dataloader, label_dict
import json
import os
from transformers import PretrainedConfig
import torch
from dataset_utils import get_normal_train_dataloader, label_dict, bmes_2_words
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel
from cws_models import  model_cls_dict
import numpy as np
import torch
from eval_utils import cws_evaluate_word_PRF
import re 

class Segmentor:
    def __init__(self, model_dir, device="cpu", max_length=500):
        self.device = device
        self.load_cws_model(model_dir)
        self.max_length = max_length
        
    def load_cws_model(self, model_dir, checkpoint_name="checkpoint_best.pt"):
        self.config = PretrainedConfig.from_json_file(os.path.join(model_dir,"config.json"))
        self.checkpoint_path = os.path.join(model_dir, checkpoint_name)
        self.model_cls = model_cls_dict[self.config.model_name]
        self.model = self.model_cls(self.config)
        self.model.load_state_dict(torch.load(self.checkpoint_path)["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.label_dict = label_dict
    
    def predict_logits(self, sentence):
        sentence = sentence.strip().replace("  "," ")
        # print(sentence)
        sentence = sentence[:self.max_length]
        input_ids = self.tokenizer.convert_tokens_to_ids(list(sentence))
        input_ids = [101]+input_ids+[102]
        inputs = {"input_ids":input_ids,
                 "attention_mask":[1]*len(input_ids),
                 "token_type_ids":[0]*len(input_ids)}
        for k,v in inputs.items():
            inputs[k] = torch.LongTensor([v])
            inputs[k] = inputs[k].to(self.device)
        logits = self.model(**inputs)
        return sentence, logits

    def __call__(self, sentence):
        sentence, logits = self.predict_logits(sentence)
        preds  = torch.argmax(logits,dim=-1)
        # print(preds)
        preds_list = preds.tolist()[0][1:-1]
        # print(preds_list)
        labels = label_dict.convert_ids2labels(preds_list)
        # print(len(sentence),len(labels))
        words_list = bmes_2_words(sentence, labels)
        return self.post_process_text(words_list)
        
    def post_process_text(self, segmented_words):
        """
        修复分词结果，将拆分的英文单词（字母）合并为完整单词。
        
        Args:
            segmented_words (list): 分词结果，包含中文词和可能拆分的英文字母。
        
        Returns:
            list: 修复后的词序列。
        """
        processed_words = []
        temp_word = ""
    
        for word in segmented_words:
            # 如果单词是字母，合并到临时词中
            if re.match(r'[a-zA-Z]', word):
                temp_word += word
            else:
                # 如果遇到非字母词，将临时词添加到结果中，并重置临时词
                if temp_word:
                    processed_words.append(temp_word)
                    temp_word = ""
                processed_words.append(word)
        
        # 如果最后一个词是字母，添加到结果中
        if temp_word:
            processed_words.append(temp_word)
        
        return processed_words
        

    def test(self, dataiter):
        label_list = []
        preds_list  = []
        for _data in dataiter["valid"]:
            for k,v in _data.items():
                _data[k] = v.to(self.device)
            with torch.no_grad():
                logits = self.model(**_data)
                preds  = torch.argmax(logits,dim=-1)
                label_list.extend(_data["label"].tolist())
                preds_list.extend(preds.tolist())
        correct_num = 0 
        new_label_list = []
        new_preds_list = []
        for ind in range(len(label_list)):
            index = 0
            assert len(label_list[ind])==len(preds_list[ind])
            for l,p in zip(label_list[ind],preds_list[ind]) :
                if l!=0:
                    new_label_list.append(l)
                    new_preds_list.append(p)
                else:
                    continue
        new_label_list_str = label_dict.convert_ids2labels(new_label_list)
        new_preds_list_str = label_dict.convert_ids2labels(new_preds_list)
        cws_evaluate_word_PRF(new_preds_list_str, new_label_list_str)
        