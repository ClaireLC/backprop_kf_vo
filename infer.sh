#!/bin/bash

dataset="kitti"
#traj_nums=(0 1 2 3 4 5 6 7 8 9)
traj_nums=(0 1 2 3)

for i in "${traj_nums[@]}"; do
    python3 cnn_infer.py --checkpoint log/normal_cn_forward_300epoch_1.00e-05/checkpoints/2019-06-11_02_48_1.00e-05_bestloss_feed_forward.tar  --save cnn_results/normal_cn_forward_300epoch_1.00e-05  --traj_num "$i"
done
