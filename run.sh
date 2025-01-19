# pku
nohup python train_teacher.py --early_stop > training_logs/train_teacher_log.txt 2>&1 &

nohup python train_cnn_wo_distillation.py --early_stop > training_logs/train_cnn_wo_distillation_log.txt 2>&1 &

## distillation phase 1
python train_distillation_student.py --early_stop > training_logs/train_cnn_distillation_phase_1_log.txt 2>&1 &

nohup python train_distillation_student.py --early_stop --phase_num 2 --gradual_unfrozen > training_logs/train_cnn_distillation_phase2_log.txt 2>&1 &