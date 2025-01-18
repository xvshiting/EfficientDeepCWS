import torch
import random
import numpy as np
from torch import nn 
import os 
from file_utils import delete_file

def set_all_random_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

# Add save checkpoint function
def save_checkpoint(model, optimizer, epoch, device):
    model.to(device=device)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, f'checkpoint_{epoch}.pt')

def save_checkpoint_util( model,
                     last_save_path,
                     keep_save_path,
                     best_save_path,
                     last_checkpoint_list,
                     keep_last_checkpoint_num,
                     other_save_info_dict,
                     is_best = False):
    
    # 保存路径
    # last_save_path = os.path.join(work_dir, last_checkpoint_temp)
    # keep_save_path = os.path.join(work_dir, normal_checkpoint_temp.format(epoch = epoch, step=step))
    # best_save_path = os.path.join(work_dir, best_checkpoint_temp)
    # 保存检查点数据
    checkpoint = {
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        # 'loss': loss,
        # 'epoch': epoch,
        # "step":step
    }
    for k,v in other_save_info_dict.items():
        checkpoint[k] = v
    if len(last_checkpoint_list)>=keep_last_checkpoint_num:
        earlist_checkpoint_path = last_checkpoint_list.pop(0)
        delete_file(earlist_checkpoint_path)
    last_checkpoint_list.append(keep_save_path)
    delete_file(last_save_path)
    torch.save(checkpoint, keep_save_path)
    torch.save(checkpoint, last_save_path)
    print(f"Checkpoint saved at {last_save_path}")
    print(f"Checkpoint saved at {keep_save_path}")
    if is_best:
        delete_file(best_save_path)
        torch.save(checkpoint, best_save_path)
        print(f"Checkpoint saved at {best_save_path}")


    


