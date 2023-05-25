#!/bin/bash

python main.py --mode student --model_arch resnet18 \
 --teacher_path output/teacher_0_resnet101_CELoss_05_14_11_48_37/models/best_checkpoint.pth \
 --adapt 1 --lr 1e-2 --epoch 400 --step_size 80 --seed 1