#!/bin/bash

python main.py --mode student --model_arch shufflenetv1 \
 --teacher_path output/teacher_0_resnet50_CELoss_05_12_08_37_50/models/best_checkpoint.pth \
 --adapt 2 --lr 1e-2 --epoch 200 --step_size 80

python main.py --mode student --model_arch shufflenetv1 \
 --teacher_path output/teacher_0_resnet50_CELoss_05_12_08_37_50/models/best_checkpoint.pth \
 --adapt 2 --lr 1e-2 --epoch 200 --step_size 80

 python main.py --mode student --model_arch shufflenetv1 \
 --teacher_path output/teacher_0_resnet101_CELoss_05_14_11_48_37/models/best_checkpoint.pth \
 --adapt 2 --lr 1e-2 --epoch 200 --step_size 80