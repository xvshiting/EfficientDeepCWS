from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel
from dataset_utils import get_normal_train_dataloader, label_dict
from torch import nn
from transformers import get_scheduler
from torch.optim import AdamW
import torch
import numpy as np
from file_utils import init_dir, delete_file
import time 
from cws_models import CWSCNNModel,CWSCNNModelWithEEConfig, model_cls_dict, CWSCNNModelWithEE, CWSRoberta
import os 
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.nn.utils import clip_grad_norm_
from train_utils import save_checkpoint_util



from train_utils import set_all_random_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="CWSCNNModelWithEE")
parser.add_argument("--random_seed", type=int, default=443)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--epoch_num", type=int, default=50)
parser.add_argument("--valid_step_num", type=int, default=1000)
parser.add_argument("--save_step_num", type=int, default=1000)
parser.add_argument("--save_best_checkpoint", type=bool, default=True)
parser.add_argument("--keep_last_checkpoint_num", type=int, default=5)
parser.add_argument("--print_step_num", type=int, default=300)
parser.add_argument("--pretrianed_model_path", type=str, default="/data/model_hub/chinese-roberta-wwm-ext")
#warm up ratio
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_len", type=int, default=500)
parser.add_argument("--dataset_name", type=str, default="pku")
# early_stop = True
parser.add_argument("--early_stop", action="store_true", default=False)
# early_stop_num = 10
parser.add_argument("--early_stop_num", type=int, default=10)
args = parser.parse_args()

#print all args
print(args)

# Rest of the code in train_teacher.py
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


loss_fn = nn.CrossEntropyLoss(ignore_index=0)


# Replace hardcoded values with args.xxx
random_seed = args.random_seed
lr = args.lr
max_grad_norm = args.max_grad_norm
epoch_num = args.epoch_num
valid_step_num = args.valid_step_num
save_step_num = args.save_step_num
save_best_checkpoint = args.save_best_checkpoint
keep_last_checkpoint_num = args.keep_last_checkpoint_num
print_step_num = args.print_step_num
model_name = args.model_name
pretrained_model_path=args.pretrianed_model_path

cur_step_num = 0
cur_time = "-".join(time.asctime().split())
dataset_name = args.dataset_name
work_dir = "output/{dataset_name}_{model_name}_lr_{lr}_epoch_{epoch_num}_{time}".format(dataset_name = dataset_name,model_name=model_name, lr=lr,epoch_num=epoch_num, time=cur_time)
init_dir(work_dir)
early_stop = args.early_stop
early_stop_num = args.early_stop_num

best_valid_loss = float("inf")
best_checkpoint_info=dict()
best_checkpoint_temp = "checkpoint_best.pt"
last_checkpoint_temp = "checkpoint_last.pt"
normal_checkpoint_temp = "checkpoint_{epoch}_{step}.pt"
warmup_ratio = args.warmup_ratio

# 初始化 TensorBoard
log_dir = os.path.join(work_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir,)

# Set random seed for reproducibility
set_all_random_seeds(random_seed)

class GlobalHolder:
    pass

def train_initialize():
    GlobalHolder.best_valid_loss = float("inf")
    GlobalHolder.cur_step_num = 0 
    GlobalHolder.patient_time = 0
    GlobalHolder.best_valid_loss_for_early_stop = float("inf")
    GlobalHolder.need_early_stop = False
last_checkpoint_list = []

my_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
dataset_loader = get_normal_train_dataloader(dataset_name=dataset_name,
                                             tokenizer=my_tokenizer,
                                             label_dict=label_dict)
num_training_steps = int(np.ceil(len(dataset_loader["train"].dataset)*epoch_num / dataset_loader["train"].batch_size))
num_warmup_steps = int(np.ceil(warmup_ratio * num_training_steps))

"""Revised: Specific for CNN model"""
config = CWSCNNModelWithEEConfig()
model = CWSCNNModel(config)

model.to(device)
optimizer = AdamW(model.parameters(),lr=lr)
scheduler = get_scheduler("linear",
                          optimizer=optimizer,
                          num_warmup_steps=num_warmup_steps,
                          num_training_steps=num_training_steps)
#save config and tokenizer
config.model_name = model_name
model.config.save_pretrained(work_dir)
my_tokenizer.save_pretrained(work_dir)

def compute_loss(logits, labels):
    batch, max_seq_len, _ = logits.shape
    flat_logits = logits.view(batch*max_seq_len, -1)
    flat_labels = labels.view(batch*max_seq_len)
    
    loss = loss_fn(flat_logits, flat_labels)
    return loss

def save_checkpoint( loss,  epoch, step, is_best = False):
    
    # 保存路径
    last_save_path = os.path.join(work_dir, last_checkpoint_temp)
    keep_save_path = os.path.join(work_dir, normal_checkpoint_temp.format(epoch = epoch, step=step))
    best_save_path = os.path.join(work_dir, best_checkpoint_temp)
    other_save_info = {'loss': loss,
        'epoch': epoch,
        "step":step}
    save_checkpoint_util(model,
                         last_save_path,
                         keep_save_path,
                         best_save_path,
                         last_checkpoint_list,
                         keep_last_checkpoint_num,
                         other_save_info,
                         is_best)


def run_valid(epoch,step_num, save=False):
    model.eval()
    epoch_loss = []
    total_labels = []
    total_preds = []
    for _data in dataset_loader["valid"]:
        for k,v in _data.items():
            _data[k] = v.to(device)
        with torch.no_grad():
            logits = model(**_data)
            loss = compute_loss(logits, _data["label"])
        epoch_loss.append(loss.tolist())
    loss = np.mean(epoch_loss)
    print("[Valid] epoch_{}-step_{}, avg  loss: {:.3f}".format(epoch, step_num, loss))
    # 记录到 TensorBoard
    writer.add_scalar("Loss/Valid", loss, step_num)
    model.train()
    if save:
        is_best = False
        if loss<GlobalHolder.best_valid_loss:
            is_best = True
            GlobalHolder.best_valid_loss = loss
            print("[Best checkpoint]")
        save_checkpoint(loss, epoch,step_num, is_best)
    if early_stop:
        if loss<GlobalHolder.best_valid_loss_for_early_stop:
               GlobalHolder.best_valid_loss_for_early_stop = loss
               GlobalHolder.patient_time = 0
        else:
               GlobalHolder.patient_time += 1

def run_train(epoch, cur_step_num):
    epoch_loss = []
    model.train()
    for _data in dataset_loader["train"]:
        cur_step_num += 1
        optimizer.zero_grad()
        for k,v in _data.items():
            _data[k] = v.to(device)
        logits = model(**_data)
        loss = compute_loss(logits, _data["label"])
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        # 更新学习率
        scheduler.step()
        epoch_loss.append(loss.tolist())
        if cur_step_num%print_step_num==0:
            print("[train] epoch_{}-step_{}, cur step loss: {:3f}, avg epoch loss: {:.3f}".format(epoch, cur_step_num, epoch_loss[-1],
                                                                                              np.mean(epoch_loss)))
                # 记录到 TensorBoard
            writer.add_scalar("Loss/Train", loss.item(), cur_step_num)
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], cur_step_num)
        if cur_step_num % valid_step_num==0:
            is_save = False
            if cur_step_num % save_step_num==0:
                is_save = True
            run_valid(epoch, cur_step_num, is_save)    
            if GlobalHolder.patient_time>=early_stop_num:
                break   
    return dict(cur_step_num=cur_step_num, epoch_loss=epoch_loss)

def train():
    train_initialize()
    for epoch in range(1, epoch_num+1):
        print("Epoch {}".format(epoch))
        train_info = run_train(epoch, GlobalHolder.cur_step_num)
        GlobalHolder.cur_step_num  = train_info["cur_step_num"]
        if GlobalHolder.patient_time>=early_stop_num:
            print("Early stop at epoch {} step {}!".format( epoch, GlobalHolder.cur_step_num))
            break
        avg_epoch_loss = np.mean(train_info["epoch_loss"])
        print("[train] epoch_{}-step_{}, avg epoch loss: {:.3f}".format(epoch, 
                                                                        GlobalHolder.cur_step_num,
                                                                        avg_epoch_loss))
        # 记录 epoch 的平均训练损失
        writer.add_scalar("Loss/Train_Epoch", avg_epoch_loss, epoch)
        run_valid(epoch, GlobalHolder.cur_step_num, True) 
        if GlobalHolder.patient_time>=early_stop_num:
            print("Early stop at epoch {} step {}!".format( epoch, GlobalHolder.cur_step_num))
            break
    # 关闭 TensorBoard
    writer.close()

train() 