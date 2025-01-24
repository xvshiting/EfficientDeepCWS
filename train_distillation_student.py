from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel,PretrainedConfig
from dataset_utils import get_normal_train_dataloader, label_dict
from torch import nn
from transformers import get_scheduler
from torch.optim import AdamW
import torch
import numpy as np
from file_utils import init_dir, delete_file
import time 
from cws_models import CWSCNNModel,CWSCNNModelWithEEConfig, model_cls_dict,CWSCNNModelWithEE,CWSRoberta
import os 
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.nn.utils import clip_grad_norm_
from train_utils import save_checkpoint_util
from torch.nn import functional as F



from train_utils import set_all_random_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--phase_num", type=int, default=1, help="1 or 2")
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
parser.add_argument("--teacher_model_dir", type=str, default="./output/pku_CWSRoberta_lr_5e-05_epoch_50_Fri-Jan-17-22:35:28-2025")
parser.add_argument("--teacher_model_name", type=str, default="checkpoint_best.pt")
parser.add_argument("--phase_1_student_model_dir", type=str, default="./output/pku_CWSCNNModelWithEE_Phase_1_lr_0.0001_epoch_50_Sat-Jan-18-23:23:35-2025")
parser.add_argument("--phase_1_student_model_name",type=str,default="checkpoint_best.pt" )
parser.add_argument("--gradual_unfrozen", action="store_true", default=False)
parser.add_argument("--gradual_unfrozen_patience_step", type=int, default=5)
#warm up ratio
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_len", type=int, default=500)
parser.add_argument("--dataset_name", type=str, default="pku")
parser.add_argument("--unlabeled_dataset_name",type=str, default="law")
# early_stop = True
parser.add_argument("--early_stop", action="store_true", default=False)
# early_stop_num = 10
parser.add_argument("--early_stop_num", type=int, default=10)
parser.add_argument("--refine", action="store_true", default=False)
parser.add_argument("--refine_student_model_dir", type=str, default="./output/pku_CWSCNNModelWithEE_Phase_1_lr_0.0001_epoch_50_Sat-Jan-18-23:23:35-2025")
parser.add_argument("--refine_student_model_name",type=str,default="checkpoint_best.pt" )
args = parser.parse_args()

#print all args
print(args)

# Rest of the code in train_teacher.py
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


loss_fn = nn.CrossEntropyLoss(ignore_index=0)

# refine after prune
refine_student_model_name = args.refine_student_model_name
refine_student_model_dir = args.refine_student_model_dir
refine = args.refine

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
unlabeled_dataset_name = args.unlabeled_dataset_name
phase_num = args.phase_num
work_dir = "output/{dataset_name}_{model_name}_Phase_{pahse}_lr_{lr}_epoch_{epoch_num}_{time}".format(dataset_name = dataset_name,
                                                                                                      pahse=phase_num,
                                                                                                      model_name=model_name,
                                                                                                      lr=lr,epoch_num=epoch_num,
                                                                                                      time=cur_time)
init_dir(work_dir)
early_stop = True
early_stop_num = 10 

best_valid_loss = float("inf")
best_checkpoint_info=dict()
best_checkpoint_temp = "checkpoint_best.pt"
last_checkpoint_temp = "checkpoint_last.pt"
normal_checkpoint_temp = "checkpoint_{epoch}_{step}.pt"
warmup_ratio = args.warmup_ratio
teacher_model_dir = args.teacher_model_dir
teacher_model_name = args.teacher_model_name
phase_1_student_model_dir = args.phase_1_student_model_dir
phase_1_student_model_name = args.phase_1_student_model_name 
gradual_unfrozen_patience_step = args.gradual_unfrozen_patience_step
gradual_unfrozen = args.gradual_unfrozen

# 初始化 TensorBoard
log_dir = os.path.join(work_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir,)

# Set random seed for reproducibility
set_all_random_seeds(random_seed)

class GlobalHolder:
    best_valid_loss = float("inf")
    cur_step_num = 0 
    patient_time = 0
    best_valid_loss_for_early_stop = float("inf")
    need_early_stop = None
    unfrozen_layer_level = None
    unfrozen_patience_step = None
    unfrozen_cur_patience_step = 0
    best_valid_loss_for_unfrozen = float("inf")
    optimizer = None
    scheduler = None

def train_initialize_phase_1():
    GlobalHolder.best_valid_loss = float("inf")
    GlobalHolder.cur_step_num = 0 
    GlobalHolder.patient_time = 0
    GlobalHolder.best_valid_loss_for_early_stop = float("inf")
    GlobalHolder.need_early_stop = early_stop
    GlobalHolder.unfrozen_layer_level = config.conv1d_cls_layer_num
    GlobalHolder.unfrozen_patience_step = gradual_unfrozen_patience_step
    GlobalHolder.unfrozen_cur_patience_step = 0
    GlobalHolder.best_valid_loss_for_unfrozen = float("inf")
    GlobalHolder.optimizer = AdamW(student_model.parameters(),lr=lr)
    GlobalHolder.scheduler = get_scheduler("linear",
                          optimizer=GlobalHolder.optimizer,
                          num_warmup_steps=num_warmup_steps,
                          num_training_steps=num_training_steps)
    if phase_num == 2 and gradual_unfrozen:
        frozen_model_initialize()

def train_initialize_phase_2():
    GlobalHolder.best_valid_loss = float("inf")
    GlobalHolder.cur_step_num = 0 
    GlobalHolder.patient_time = 0
    GlobalHolder.best_valid_loss_for_early_stop = float("inf")
    GlobalHolder.need_early_stop = early_stop
    GlobalHolder.unfrozen_layer_level = config.conv1d_cls_layer_num
    GlobalHolder.unfrozen_patience_step = gradual_unfrozen_patience_step
    GlobalHolder.unfrozen_cur_patience_step = 0
    GlobalHolder.best_valid_loss_for_unfrozen = float("inf")
    GlobalHolder.optimizer = AdamW(student_model.parameters(),lr=lr)
    GlobalHolder.scheduler = get_scheduler("linear",
                          optimizer=GlobalHolder.optimizer,
                          num_warmup_steps=num_warmup_steps,
                          num_training_steps=num_training_steps)
    if phase_num == 2 and gradual_unfrozen:
        frozen_model_initialize()

def train_initialize_phase_2_subtask():
    GlobalHolder.best_valid_loss = float("inf")
    GlobalHolder.cur_step_num = 0 
    GlobalHolder.patient_time = 0
    GlobalHolder.best_valid_loss_for_early_stop = float("inf")
    GlobalHolder.need_early_stop = early_stop
    # GlobalHolder.unfrozen_layer_level = config.conv1d_cls_layer_num
    GlobalHolder.unfrozen_patience_step = gradual_unfrozen_patience_step
    GlobalHolder.unfrozen_cur_patience_step = 0
    GlobalHolder.best_valid_loss_for_unfrozen = float("inf")
    GlobalHolder.optimizer = AdamW(student_model.parameters(),lr=lr)
    GlobalHolder.scheduler = get_scheduler("linear",
                          optimizer=GlobalHolder.optimizer,
                          num_warmup_steps=num_warmup_steps,
                          num_training_steps=num_training_steps)
last_checkpoint_list = []

# 加载数据集
my_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
labeled_dataset_loader = get_normal_train_dataloader(dataset_name=dataset_name,
                                             tokenizer=my_tokenizer,
                                             label_dict=label_dict)
unlabeled_dataset_loader = get_normal_train_dataloader(dataset_name=unlabeled_dataset_name,
                                             tokenizer=my_tokenizer,
                                             label_dict=label_dict,
                                             train_size=0.99)

if phase_num == 2:
    num_training_steps = int(np.ceil(len(labeled_dataset_loader["train"].dataset)*epoch_num / labeled_dataset_loader["train"].batch_size))
elif phase_num == 1:
    num_training_steps = int(np.ceil(len(unlabeled_dataset_loader["train"].dataset)*epoch_num / unlabeled_dataset_loader["train"].batch_size))

num_warmup_steps = int(np.ceil(warmup_ratio * num_training_steps))

"""Revised Load student and teacher model"""
## define student model
config = CWSCNNModelWithEEConfig()
student_model = CWSCNNModelWithEE(config) #defualt we train a new model
if phase_num==2: #need load phase 1 model
    if refine:#load pruned model
        config = PretrainedConfig.from_json_file(os.path.join(refine_student_model_dir,"config.json")) 
        student_model = CWSCNNModelWithEE(config)
        if os.path.exists(refine_student_model_dir):
           student_model.load_state_dict(torch.load(os.path.join(refine_student_model_dir, refine_student_model_dir))["model_state_dict"])
        else:
            print("Pruned model dir {} not valid! training from new!".format(refine_student_model_dir))
           
    else:#load phase 1 model
        if os.path.exists(phase_1_student_model_dir): # phase_1_student_model_dir not none then load ,else train from brand new model.
            student_model.load_state_dict(torch.load(os.path.join(phase_1_student_model_dir, phase_1_student_model_name))["model_state_dict"])
        else:
            print("Phase  I model dir {} not valid! training from new!".format(phase_1_student_model_dir))
            
student_model.to(device)
print(student_model)
#load teacher model 
teacher_config = PretrainedConfig.from_json_file(os.path.join(teacher_model_dir,"config.json"))
teacher_checkpoint_path = os.path.join(teacher_model_dir, teacher_model_name)
teacher_model = CWSRoberta(teacher_config)
teacher_model.load_state_dict(torch.load(teacher_checkpoint_path)["model_state_dict"])
teacher_model.eval()
teacher_model.to(device)





# optimizer = AdamW(student_model.parameters(),lr=lr)
# scheduler = get_scheduler("linear",
#                           optimizer=optimizer,
#                           num_warmup_steps=num_warmup_steps,
#                           num_training_steps=num_training_steps)
#save config and tokenizer
config.model_name = model_name
student_model.config.save_pretrained(work_dir)
my_tokenizer.save_pretrained(work_dir)

def frozen_model_initialize():
    """frozen all layers except top layer"""
    for param in student_model.embedding.parameters():
            param.requires_grad = False 
    for layer in student_model.conv1d_cls_layers:
        for param in layer.parameters():
            param.requires_grad = False
    print("[Frozen] all layer except projection classification layer!")
        
def unfrozen_layer():
    "unfrozen one layer according unfrozen layer level"
    GlobalHolder.patient_time = 0
    GlobalHolder.unfrozen_cur_patience_step = 0
    GlobalHolder.best_valid_loss_for_unfrozen = float("inf")
    GlobalHolder.best_valid_loss_for_early_stop = float("inf")
    if GlobalHolder.unfrozen_layer_level==0:
        for param in  student_model.conv1d_cls_layers[0].parameters() :
            param.requires_grad = True
        for  param in student_model.embedding.parameters():
            param.requires_grad = True
        print("[Unfrozen] layer {} and embedding, all layers have been unfrozen!".format(GlobalHolder.unfrozen_layer_level+1))
    elif GlobalHolder.unfrozen_layer_level<=5 and GlobalHolder.unfrozen_layer_level>=1:
        for param in  student_model.conv1d_cls_layers[GlobalHolder.unfrozen_layer_level].parameters() :
            param.requires_grad = True
        print("[Unfrozen] layer {}".format(GlobalHolder.unfrozen_layer_level+1))

def compute_ce_loss(logits, labels):
    batch, max_seq_len, _ = logits.shape
    flat_logits = logits.view(batch*max_seq_len, -1)
    flat_labels = labels.view(batch*max_seq_len)
    loss = loss_fn(flat_logits, flat_labels)
    return loss

def compute_mse_loss_with_mask(rep_teacher,
                         rep_student,
                         attention_mask):
    """ omitting the padding tokens , mse of student and teancher logits, according attention_mask=0"""
    batch, max_seq_len, _ = rep_teacher.shape
    flat_logits_teacher = rep_teacher.view(batch*max_seq_len, -1)
    flat_logits_student = rep_student.view(batch*max_seq_len, -1)
    flat_attention_mask = attention_mask.view(batch*max_seq_len)
    flat_logits_teacher = flat_logits_teacher[flat_attention_mask == 1]
    flat_logits_student = flat_logits_student[flat_attention_mask == 1]
    loss = F.mse_loss(flat_logits_student, flat_logits_teacher)
    return loss

def compute_phase_1_loss(rep_teacher,
                         rep_student,
                         attention_mask):
    return compute_mse_loss_with_mask(rep_teacher, rep_student, attention_mask)

def compute_phase_2_loss(logits_student_list,
                         logits_teacher=None,
                         attention_mask=None,
                         label = None):
    """ label = None compute for unlabed data and mse loss will be used, otherwise compute loss for labeled data and crossentropy loss will be used!"""
    # 获取起始层
    start_layer = getattr(GlobalHolder, "unfrozen_layer_level", 0)
    if start_layer < 0 or start_layer >= len(logits_student_list):
        raise ValueError(f"`unfrozen_layer_level` ({start_layer}) is out of range for logits_student_list of length {len(logits_student_list)}.")
    
    total_loss = 0.0
    for ind in range(start_layer, len(logits_student_list)):
        if label is not None:  # 有标签数据
            total_loss += (ind+1)*compute_ce_loss(logits_student_list[ind], label)
        else:  # 无标签数据
            total_loss += (ind+1)*compute_mse_loss_with_mask(logits_teacher,logits_student_list[ind], attention_mask)
    
    # 计算平均损失
    # mean_loss = total_loss / (len(logits_student_list) - start_layer)
    mean_loss = total_loss / sum(list(range(start_layer+1, len(logits_student_list)+1)))
    return mean_loss


def save_checkpoint( loss,  epoch, step, is_best = False):
    
    # 保存路径
    last_save_path = os.path.join(work_dir, last_checkpoint_temp)
    keep_save_path = os.path.join(work_dir, normal_checkpoint_temp.format(epoch = epoch, step=step))
    best_save_path = os.path.join(work_dir, best_checkpoint_temp)
    other_save_info = {'loss': loss,
        'epoch': epoch,
        "step":step}
    save_checkpoint_util(student_model,
                         last_save_path,
                         keep_save_path,
                         best_save_path,
                         last_checkpoint_list,
                         keep_last_checkpoint_num,
                         other_save_info,
                         is_best)

def run_valid_phase_1(epoch,step_num, save=False):
    student_model.eval()
    epoch_loss = []
    total_labels = []
    total_preds = []
    for _data in unlabeled_dataset_loader["valid"]:
        for k,v in _data.items():
            _data[k] = v.to(device)
        with torch.no_grad():
            student_ret = student_model(**_data)
            student_rep = student_ret["hidden_x"]
            # student_logits_list = student_ret["logits_list"]
            teacher_rep, teacher_logits = teacher_model(**_data, ret_h=True)
            loss = compute_phase_1_loss(student_rep, teacher_rep,
                                        _data["attention_mask"],
                                        )
        epoch_loss.append(loss.tolist())
    student_model.train()
    loss = np.mean(epoch_loss)
    print("[Valid] epoch_{}-step_{}, avg  loss: {:.3f}".format(epoch, step_num, loss))
    # 记录到 TensorBoard
    writer.add_scalar("Loss/Valid", loss, step_num)
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

def run_valid_phase_2(epoch,step_num, save=False):
    student_model.eval()
    epoch_loss = []
    total_labels = []
    total_preds = []
    for _data in labeled_dataset_loader["valid"]:
        for k,v in _data.items():
            _data[k] = v.to(device)
        with torch.no_grad():
            student_logits = student_model(**_data)["logits_list"]
            loss = compute_phase_2_loss(logits_student_list = student_logits,
                                        label=_data["label"])
        epoch_loss.append(loss.tolist())
    student_model.train()
    loss = np.mean(epoch_loss)
    print("[Frozen-{}] [Valid] epoch_{}-step_{}, avg  loss: {:.3f}".format(GlobalHolder.unfrozen_layer_level, epoch, step_num, loss))
    # 记录到 TensorBoard
    writer.add_scalar("Loss/Valid/fronzen-{}".format(GlobalHolder.unfrozen_layer_level), loss, step_num)
    
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
    # if gradual_unfrozen and loss<GlobalHolder.best_valid_loss_for_unfrozen:
    #     GlobalHolder.best_valid_loss_for_unfrozen = loss 
    #     GlobalHolder.unfrozen_cur_patience_step = 0
    # else:
    #     GlobalHolder.unfrozen_cur_patience_step += 1

def run_train_phase_1(epoch, cur_step_num):
    epoch_loss = []
    student_model.train()
    # unlabed_dataiter = iter(unlabeled_dataset_loader)
    for _data in unlabeled_dataset_loader["train"]:
        cur_step_num += 1
        GlobalHolder.optimizer.zero_grad()
        for k,v in _data.items():
            _data[k] = v.to(device)
        
        # labeled data
        student_ret = student_model(**_data)
        student_rep = student_ret["hidden_x"]
        # student_logits_list = student_ret["logits_list"]
        teacher_rep, teacher_logits = teacher_model(**_data, ret_h=True)
        loss = compute_phase_1_loss(student_rep, teacher_rep,
                                    _data["attention_mask"],
                                    )
        loss.backward()
        clip_grad_norm_(student_model.parameters(), max_grad_norm)
        GlobalHolder.optimizer.step()
        # 更新学习率
        GlobalHolder.scheduler.step()
        epoch_loss.append(loss.tolist())
        if cur_step_num%print_step_num==0:
            print("[train] epoch_{}-step_{}, cur step loss: {:3f}, avg epoch loss: {:.3f}".format(epoch, cur_step_num, epoch_loss[-1],
                                                                                              np.mean(epoch_loss)))
                # 记录到 TensorBoard
            writer.add_scalar("Loss/Train", loss.item(), cur_step_num)
            writer.add_scalar("Learning Rate", GlobalHolder.scheduler.get_last_lr()[0], cur_step_num)
        if cur_step_num % valid_step_num==0:
            is_save = False
            if cur_step_num % save_step_num==0:
                is_save = True
            run_valid_phase_1(epoch, cur_step_num, is_save)    
            if early_stop and GlobalHolder.patient_time>=early_stop_num:
                break  
    return dict(cur_step_num=cur_step_num, epoch_loss=epoch_loss)

def run_train_phase_2(epoch, cur_step_num):
    epoch_loss = []
    student_model.train()
    unlabeled_dataiter = iter(unlabeled_dataset_loader["train"])
    for _data in labeled_dataset_loader["train"]:
        cur_step_num += 1
        GlobalHolder.optimizer.zero_grad()
        for k,v in _data.items():
            _data[k] = v.to(device)
        
        # labeled data
        student_ret = student_model(**_data)
        student_logits_list = student_ret["logits_list"]
        loss_labeled = compute_phase_2_loss(logits_student_list= student_logits_list,
                                    label = _data["label"])
        
        try:
            unlabeled_data = next(unlabeled_dataiter)
        except StopIteration:
            unlabed_dataiter = iter(unlabeled_dataset_loader["train"])  # 重新初始化
            unlabeled_data = next(unlabeled_dataiter)
        for k,v in unlabeled_data.items():
            unlabeled_data[k] = v.to(device)
        unlabeled_student_ret = student_model(**unlabeled_data)
        unlabeled_student_logits_list = unlabeled_student_ret["logits_list"]
        _, teacher_logits = teacher_model(**unlabeled_data, ret_h=True)
        loss_unlabeled = compute_phase_2_loss(logits_student_list = unlabeled_student_logits_list,
                                    logits_teacher = teacher_logits,
                                    attention_mask = unlabeled_data["attention_mask"],
                                    )
        loss = loss_labeled*0.5 + loss_unlabeled*0.5
        loss.backward()
        clip_grad_norm_(student_model.parameters(), max_grad_norm)
        GlobalHolder.optimizer.step()
        # 更新学习率
        GlobalHolder.scheduler.step()
        epoch_loss.append(loss.tolist())
        if cur_step_num%print_step_num==0:
            print("[Frozen-{}] [train] epoch_{}-step_{}, cur step loss: {:3f}, avg epoch loss: {:.3f}".format(GlobalHolder.unfrozen_layer_level, epoch, cur_step_num, epoch_loss[-1],
                                                                                              np.mean(epoch_loss)))
                # 记录到 TensorBoard
            writer.add_scalar("Loss/Train/frozen-{}".format(GlobalHolder.unfrozen_layer_level), loss.item(), cur_step_num)
            writer.add_scalar("Learning Rate/fronzen-{}".format(GlobalHolder.unfrozen_layer_level), GlobalHolder.scheduler.get_last_lr()[0], cur_step_num)
        if cur_step_num % valid_step_num==0:
            is_save = False
            if cur_step_num % save_step_num==0:
                is_save = True
            run_valid_phase_2(epoch, cur_step_num, is_save)    
            if early_stop and GlobalHolder.patient_time>=early_stop_num:
                break  
            # if gradual_unfrozen and GlobalHolder.unfrozen_cur_patience_step>GlobalHolder.unfrozen_patience_step:
            #     GlobalHolder.unfrozen_layer_level -= 1
            #     if GlobalHolder.unfrozen_layer_level>=0:
            #         unfrozen_layer() 
    return dict(cur_step_num=cur_step_num, epoch_loss=epoch_loss)

def train_phase_2_ungradual_unfrozen():
    train_initialize_phase_2()
    for unfrozen_layer_level in range(6,-1,-1):
        GlobalHolder.unfrozen_layer_level = unfrozen_layer_level
        unfrozen_layer()
        train_initialize_phase_2_subtask()
        for epoch in range(1, epoch_num+1):
            print("[Frozen-{}] Epoch {}".format(GlobalHolder.unfrozen_layer_level, epoch))
            train_info = run_train_phase_2(epoch, GlobalHolder.cur_step_num)
            GlobalHolder.cur_step_num  = train_info["cur_step_num"]
            if GlobalHolder.patient_time>=early_stop_num:
                print("[Frozen-{}] Early stop at epoch {} step {}!".format(GlobalHolder.unfrozen_layer_level,  epoch, GlobalHolder.cur_step_num))
                break
            avg_epoch_loss = np.mean(train_info["epoch_loss"])
            print("[Frozen-{}][train] epoch_{}-step_{}, avg epoch loss: {:.3f}".format(GlobalHolder.unfrozen_layer_level, epoch, 
                                                                            GlobalHolder.cur_step_num,
                                                                            avg_epoch_loss))
            # 记录 epoch 的平均训练损失
            writer.add_scalar("Loss/Train_Epoch/fronzen-{}".format(GlobalHolder.unfrozen_layer_level), avg_epoch_loss, epoch)
            run_valid_phase_2(epoch, GlobalHolder.cur_step_num, True) 
    # 关闭 TensorBoard
    writer.close()


                                
def train_phase_2():
    train_initialize_phase_2()
    for epoch in range(1, epoch_num+1):
        
        print("Epoch {}".format(epoch))
        train_info = run_train_phase_2(epoch, GlobalHolder.cur_step_num)
        GlobalHolder.cur_step_num  = train_info["cur_step_num"]
        if GlobalHolder.patient_time>=early_stop_num:
            print("Early stop at epoch {} step {}!".format( epoch, GlobalHolder.cur_step_num))
            break
        avg_epoch_loss = np.mean(train_info["epoch_loss"])
        print("[train] epoch_{}-step_{}, avg epoch loss: {:.3f}".format(epoch, 
                                                                        GlobalHolder.cur_step_num,
                                                                        avg_epoch_loss))
        # 记录 epoch 的平均训练损失
        writer.add_scalar("Loss/Train_Epoch/fronzen-{}".format(GlobalHolder.unfrozen_layer_level), avg_epoch_loss, epoch)
        run_valid_phase_2(epoch, GlobalHolder.cur_step_num, True) 
    # 关闭 TensorBoard
    writer.close()

def train_phase_1():
    train_initialize_phase_1()
    for epoch in range(1, epoch_num+1):
        print("Epoch {}".format(epoch))
        train_info = run_train_phase_1(epoch, GlobalHolder.cur_step_num)
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
        run_valid_phase_1(epoch, GlobalHolder.cur_step_num, True) 
    # 关闭 TensorBoard
    writer.close()

if phase_num ==2:
    if gradual_unfrozen:
        train_phase_2_ungradual_unfrozen()
    else:
        train_phase_2() 
elif phase_num == 1:
    train_phase_1()