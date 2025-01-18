# pku
nohup python train_teacher.py > training_logs/train_teacher_log.txt 2>&1 &

nohup python train_cnn_wo_distillation.py > training_logs/train_cnn_wo_distillation_log.txt 2>&1 &

