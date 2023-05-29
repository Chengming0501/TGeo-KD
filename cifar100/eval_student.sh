#!/bin/bash

python teacher_eval.py --model_arch resnet18 \
  --write_name student \
  --ckpt_path /home/../Knowledge_agreement/output/student_resnet18_KDLoss_012_14_22_16_28/models/best_checkpoint.pth