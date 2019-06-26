# backprop_kf_vo
Implementation of backprop kf (Haarnoja, et al) using PyTorch for visual odometry. Trained with KITTI dataset.

## Descriptions of files and directories in this repository

### Data preprocessing scripts for KITTI and Ouija trajectories
The `preprocessing/` directory contains scripts for preprocessing data.

#### KITTI

* `preprocess_sequences.py` 
Resizes KITTI images and computes difference images.

#### Ouijabot

These scripts preprocess Ouija trajectories that are saved as rosbags for testing. Not needed for real-time inference.

* `ouija_images.py`

Resizes images from on-board camera to 150x50 and computes difference images. Saves current and difference images at each timestes. **Make sure to modify file paths to trajectory directory**

* `ouija_optitrack.py`

Class that parses optitrack data from data.txt file (generated from rosbag). Contains a function to calculate heading angles from quaternions and a function to calculate ground truth forward and angular velocity from robot locations. 

### PyTorch datasets (for dataloaders)

* `kitti_dataset.py`

Formats KITTI dataset samples, where each sample  as a dict containing "curr_img", "diff_img", "pose", "vel" and "curr_time".
Creates or loads .npy files for training and validation datasets (shuffled) and inference dataset (samples are in order).

* `kitti_dataset_seq.py`

Generates sequences of specified length out of KITTI trajectories.
Creates or loads .npy files for training and validation datasets (shuffled) and inference dataset (samples are in order).

### PyTorch models

The `models/` directory contains PyTorch models of the feed forward cnn and differentiable extended Kalman filter for the KITTI dynamics model.

* `feed_forward_cnn_model.py`
* `kalmanfilter_model.py`

### Piecewise KF training and inference scripts

### End to end training and inference scripts

### Plotting results

