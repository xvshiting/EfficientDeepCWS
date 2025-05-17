
# 🔧 Hyperparameter Guide for DEEP-CWS

This document provides a detailed overview of the hyperparameters used in the training, distillation, pruning, and deployment phases of DEEP-CWS across different datasets.

---

## 1. 🧠 Teacher Model Training (`train_teacher.py`)

| Parameter              | Value                          |
|------------------------|--------------------------------|
| Model                  | `hfl/chinese-roberta-wwm-ext`  |
| Optimizer              | AdamW                          |
| Learning Rate          | 5e-5                           |
| Batch Size             | 16                             |
| Epochs                | 50 (early stop enabled)        |
| Warmup Ratio           | 0.1                            |
| Max Gradient Norm      | 1.0                            |
| Max Sequence Length    | 500                            |
| Early Stopping         | Enabled                        |
| Early Stop Patience    | 10                             |
| Datasets Supported     | PKU / MSR / AS / CITYU         |

Example:
```bash
nohup python train_teacher.py --early_stop --dataset_name pku > train_teacher_log.txt 2>&1 &
```

---

## 2. 🏗️ CNN Student (Without Distillation) (`train_cnn_wo_distillation.py`)

| Parameter              | Value         |
|------------------------|---------------|
| Optimizer              | AdamW         |
| Learning Rate          | 5e-5          |
| Batch Size             | 16            |
| Epochs                 | 50            |
| Early Stopping         | Enabled       |
| Max Sequence Length    | 500           |

Example:
```bash
nohup python train_cnn_wo_distillation.py --early_stop --dataset_name msr > train_log.txt 2>&1 &
```

---

## 3. 🔁 Distillation Phase I & II (`train_distillation_student.py`)

### Phase I:
| Parameter              | Value             |
|------------------------|------------------|
| Learning Rate          | 5e-5              |
| Epochs                 | 50                |
| Distillation Phase     | 1 (default)       |
| Early Stopping         | Enabled           |

```bash
python train_distillation_student.py --early_stop
```

### Phase II:
| Parameter              | Value                     |
|------------------------|----------------------------|
| Learning Rate          | 1e-4                       |
| Epochs                 | 100                        |
| Gradual Unfreezing     | Enabled (`--gradual_unfrozen`) |
| Phase Num              | 2                          |
| Early Stop Patience    | 20                         |

```bash
nohup python train_distillation_student.py --early_stop --epoch_num 100 --phase_num 2 --early_stop_num 20 --gradual_unfrozen --lr 1e-4 > log.txt 2>&1 &
```

---

## 4. ✂️ Model Pruning (`prune_cnn_model.py`)

| Parameter              | Value                         |
|------------------------|-------------------------------|
| Prune Ratios           | 0.15, 0.40, 0.55, 0.75, 0.70, 0.70 |
| Base Model Dir         | `output/pku_CWSCNNModelWithEE_Phase_2` |
| Output Dir             | `output/pku_pruned_CWSCNNModelWithEE` |

```bash
python prune_cnn_model.py --model_dir ./output/... --prune_ratio_list 0.15 0.40 0.55 0.75 0.70 0.70 --output_model_dir ./output/...
```

---

## 5. 🔁 Pruned Model Refinement

```bash
nohup python train_distillation_student.py --early_stop --epoch_num 100 --phase_num 2 --early_stop_num 20 --gradual_unfrozen --lr 1e-4 --refine > refine_log.txt 2>&1 &
```

---

## 🔗 Supported Datasets
- `--dataset_name pku`
- `--dataset_name msr`
- `--dataset_name as`
- `--dataset_name cityu`

Ensure dataset files are correctly placed and tokenized before training.

---

## 📌 Notes
- All training commands support `--early_stop`, `--dataset_name`, and `--lr` override.
- Checkpoints and logs are saved in the `output/` and `training_logs/` folders respectively.
