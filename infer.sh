#!/bin/bash

dataset="kitti"
traj_nums=(0 1 2 3 4 5 6 7 8 9)

for i in "${traj_nums[@]}"; do
  python3 cnn_infer.py --traj_num "$i"
done
