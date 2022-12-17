#!/bin/bash
# Baseline
#python main.py --mode student --model_arch resnet18 --loss CELoss --seed 1

python main.py --mode student --model_arch resnet18 \
 --loss KDLoss --device cuda:1 --lr 0.1 --epoch 600 --step_size 150 \
 --teacher_path output/teacher_resnet50_CELoss_012_13_11_13_06/models/best_checkpoint.pth