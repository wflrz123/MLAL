#!/bin/bash
#
# For 5-shot, 5-class training
# Hyper-parameters follow https://github.com/twitter/meta-learning-lstm
#                --resume logs-252/ckpts/meta-learner-50000.pth.tar \
python main.py --mode train \
               --n-shot 1 \
               --n-eval 15 \
               --n-class 5 \
               --input-size 4 \
               --hidden-size 4 \
               --lr 1e-3 \
               --episode 50000 \
               --episode-val 100 \
               --epoch 16 \
               --batch-size 25 \
               --image-size 84 \
               --grad-clip 0.25 \
               --bn-momentum 0.95 \
               --bn-eps 1e-3 \
               --data miniimagenet \
               --data-root data/imagenet/ \
               --pin-mem True \
               --log-freq 100 \
               --val-freq 1000
