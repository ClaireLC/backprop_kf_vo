#!/bin/bash

dataset="kitti"
traj_nums=(0 1 2 3 4 5 6 7 8 9)
model_path="/home/clairech/backprop_kf_vo/log/normal_cn_forward_300epoch_3.00e-04/checkpoints/2019-06-10_20_48_3.00e-04_bestloss_feed_forward.tar"
save_dir="./cnn_results/normal_cn_forward_300epoch_3.00e-04/"

for i in "${traj_nums[@]}"; do
  python3 cnn_infer.py --traj_num "$i" --checkpoint $model_path --save $save_dir
done
