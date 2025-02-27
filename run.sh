# pku
nohup python train_teacher.py --early_stop > training_logs/train_teacher_log.txt 2>&1 &

nohup python train_cnn_wo_distillation.py --early_stop > training_logs/train_cnn_wo_distillation_log.txt 2>&1 &

## distillation phase 1
python train_distillation_student.py --early_stop > training_logs/train_cnn_distillation_phase_1_log.txt 2>&1 &

## distillation phase II
nohup python train_distillation_student.py --early_stop --epoch_num 100  --phase_num 2 --early_stop_num 20 --gradual_unfrozen --lr 1e-4 > training_logs/pku_train_cnn_distillation_phase2_log.txt 2>&1 &

## prune 
python prune_cnn_model.py --model_dir ./output/pku_CWSCNNModelWithEE_Phase_2 \
              --prune_ratio_list 0.15 0.40 0.55 0.75 0.70 0.70 \
              --output_model_dir ./output/pku_pruned_CWSCNNModelWithEE

## refine 
nohup python train_distillation_student.py --early_stop --epoch_num 100  --phase_num 2 --early_stop_num 20 --gradual_unfrozen --lr 1e-4 --refine > training_logs/pku_train_cnn_distillation_phase2_refine_pruned_log.txt 2>&1 &



#MSR

nohup python train_teacher.py --early_stop --dataset_name msr > training_logs/msr_train_teacher_log.txt 2>&1 &
nohup python train_cnn_wo_distillation.py --early_stop --dataset_name msr > training_logs/msr_train_cnn_wo_distillation_log.txt 2>&1 &


#AS
nohup python train_teacher.py --early_stop --dataset_name as > training_logs/as_train_teacher_log.txt 2>&1 &
nohup python train_cnn_wo_distillation.py --early_stop --dataset_name as > training_logs/as_train_cnn_wo_distillation_log.txt 2>&1 &




#cityu
nohup python train_teacher.py --early_stop --dataset_name cityu > training_logs/cityu_train_teacher_log.txt 2>&1 &
nohup python train_cnn_wo_distillation.py --early_stop --dataset_name cityu > training_logs/cityu_train_cnn_wo_distillation_log.txt 2>&1 &
