# Efficient Deep CWS 

This is the code for the paper "Over 100x Faster Word Segmentation: A Distillation, Pruning, and
Early Exit Approach with ONNX Deployment".

Chinese Word Segmentation is a fundamental task in Chinese NLP. This project aims to accelerate the inference speed of Chinese Word Segmentation models while maintaining high accuracy. We achieve this by combining knowledge distillation, pruning, and early exit techniques. The final model is deployed using ONNX for further optimization.

We public all training scripts and models.

## Requirements

- Python 3.9+
- torch 2.1.0
- transformers 4.38.1

## Usage

### Train
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

# Result On BenchMark

## PKU (fake data now)

| Model | P | R | F1 | Speed |
| --- | --- | --- | --- | --- |
| Teacher | 97.91 | 97.94 | 97.92 | 0.12s |
| CNNModel|94.48 | 94.56 | 94.52 | 0.12s |
| Student | 96.38 | 96.38 | 96.38 | 0.12s |
| Student (Pruned) | 96.38 | 96.38 | 96.38 | 0.12s |
| Student (ONNX) | 96.38 | 96.38 | 96.38 | 0.12s |
