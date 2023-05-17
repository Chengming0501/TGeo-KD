#!/bin/bash

#python main.py --mode student --model_arch shufflenetv1 \
# --teacher_path output/teacher_0_resnet50_CELoss_05_12_08_37_50/models/best_checkpoint.pth \
# --adapt 0 --lr 1e-2 --epoch 400 --step_size 80
#
#python main.py --mode student --model_arch resnet18 \
# --teacher_path output/teacher_0_resnet50_CELoss_05_12_08_37_50/models/best_checkpoint.pth \
# --adapt 1 --lr 1e-2 --epoch 400 --step_size 80 --seed 1

python main.py --mode student --model_arch shufflenetv2 \
 --teacher_path output/teacher_0_resnet152_CELoss_05_16_09_32_11/models/best_checkpoint.pth \
 --adapt 1 --lr 1e-2 --epoch 400 --step_size 80 --seed 5