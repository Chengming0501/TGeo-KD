#!/bin/bash

python teacher_eval.py --model_arch resnet18 \
  --write_name regular \
  --ckpt_path /home/../Knowledge_agreement/output/student_resnet18_CELoss_112_14_21_12_29/models/best_checkpoint.pth