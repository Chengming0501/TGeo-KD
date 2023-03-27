#!/bin/bash
# Baseline
#python main.py --mode student --model_arch resnet18 --loss CELoss --seed 1 --device cuda:1

python main.py --mode student --model_arch resnet18 \
 --teacher_path output/teacher_resnet18_CELoss_1_03_16_23_22_54/models/best_checkpoint.pth \
 --loss KDLoss --device cuda:1 \
 --lr 1e-3 --epoch 400 --step_size 80