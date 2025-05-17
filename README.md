# Efficient Deep CWS 

This is the code for the paper "DEEP-CWS: Distilling Efficient Pre-trained models with Early Exit and Pruning for Scalable Chinese Word Segmentation".

Chinese Word Segmentation is a fundamental task in Chinese NLP. This project aims to accelerate the inference speed of Chinese Word Segmentation models while maintaining high accuracy. We achieve this by combining knowledge distillation, pruning, and early exit techniques. The final model is deployed using ONNX for further optimization.

We public all training scripts， 📘 For detailed training settings, see [hyperparameters.md](./hyperparameters.md)

## Requirements

- Python 3.9+
- torch 2.1.0
- transformers 4.38.1

## 📂 Directory Structure

```
.
├── run.sh                      # All training and evaluation commands
├── train_teacher.py           # RoBERTa teacher model training
├── train_cnn_wo_distillation.py  # CNN model baseline
├── train_distillation_student.py # Phase I / II distillation & refine
├── prune_cnn_model.py         # Pruning interface
├── convert_2_onnx.py          # ONNX export
├── hyperparameters.md         # All configurable parameters
├── paper.pdf                  # Main paper
├── supplement.pdf             # Extended results and discussion
└── README.md
```

---

## Usage

### ⚙️ Pipeline Overview

DEEP-CWS consists of the following modular stages:

```
Pretrained RoBERTa (Teacher)
        │
        ▼
Distillation Phase I ──► CNN Backbone
        │
        ▼
Distillation Phase II (Gradual Unfreezing)
        │
        ├──► Pruning (L1-based)
        ▼
    Refined CNN
        │
        ▼
Export to ONNX → Efficient Inference
```

### Train



Each stage is optional and configurable. Users can stop after Phase I, or skip pruning if full accuracy is needed.

Training scripts are in the `scripts` folder. You can run them directly.

- train_teacher.py: Train the teacher model.
- train_student.py: Train the student model with knowledge distillation.
- train_cnn_wo_distillation.py: Train the student model without knowledge distillation.
- train_distillation_student.py: Train the student model with knowledge distillation.

### Prune
- prunne_cnn_model.py: Prune the student model.
- prune_analysis.py: Analyze the pruning results, try different pruning rates.

### ONNX
- convert_2_onnx.py: Convert the model to ONNX format.

## 📊 Results

The DEEP-CWS framework achieves high segmentation accuracy while significantly improving inference efficiency. Key performance highlights:
The following table reports the performance of the final optimized DEEP-CWS model (after distillation, pruning, and ONNX acceleration) across four widely-used Chinese Word Segmentation datasets.

| Dataset | F1 Score (%) | Inference Time (ms/sentence) | Model Size |
|---------|--------------|-------------------------------|------------|
| PKU     | 97.81        | 1.32                          | ~1.1M      |
| MSR     | 98.72        | 1.44                          | ~1.1M      |
| AS      | 97.39        | 1.58                          | ~1.1M      |
| CITYU   | 98.21        | 1.40                          | ~1.1M      |
---


## 📘 Citation

If you use this project in your work, please cite:

```
@article{deepcws2025,
  title={DEEP-CWS: Distilling Efficient Pre-trained models with Early Exit and Pruning for Scalable Chinese Word Segmentation},
  author={Xu, Shiting},
  journal={TBD},
  year={2025}
}
```
