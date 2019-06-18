"""
Run inferrence on a KITTI trajectory
"""
import numpy as np
import time
import random
import csv
import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import statistics
from synth_vis_state_est_data_generator import SynthVisStateEstDataGenerator
from matplotlib import pyplot as plt

from models.feed_forward_cnn_model import FeedForwardCNN
from models.kalmanfilter_model import KalmanFilter
from kitti_dataset import KittiDataset
from kitti_dataset_seq import KittiDatasetSeq

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', dest='checkpoint', default='', help='model checkpoint')
parser.add_argument('--save', dest='save', default='./cnn_results/', help='save location')
parser.add_argument("--traj_num", dest='traj_num', default='0', help="Trajectory number")

args = parser.parse_args()
os.makedirs(args.save, exist_ok=True)

# Device specification
#device = torch.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup
image_dims = np.array((50, 150))

# Dataset directories
SEQ_DIR = "/mnt/disks/dataset/dataset_post/sequences/"
POSES_DIR = "/mnt/disks/dataset/dataset_post/poses/"
OXTS_DIR = "/mnt/disks/dataset/dataset_post/oxts/"

def infer(model_path, sequence_num, camera_num):
  """
  Loads a model and infers velocities from once sequence of data
  model_path: path to model
  sequence_num: sequence ID whose velocities we want to infer
  camera_num: camera id
  """
  # Load model from path
  print('Loading model from: ',model_path)

  batch_size = 1

  CNNModel = FeedForwardCNN(image_channels=6, image_dims=np.array((50, 150)), z_dim=2, output_covariance=True, batch_size=batch_size).to(device)

  KFModel = KalmanFilter(device).to(device)

  checkpoint = torch.load(model_path, map_location=device)
  CNNModel.load_state_dict(checkpoint['cnn_model_state_dict'], strict=False)
  KFModel.load_state_dict(checkpoint['kf_model_state_dict'])

  # Set model to eval mode
  CNNModel.eval()
  KFModel.eval()

  # Construct loss function
  loss_function = torch.nn.MSELoss(reduction='sum')

  # Create dataset
  seq_length = 100
  seq_dataset = KittiDatasetSeq(SEQ_DIR, POSES_DIR, OXTS_DIR, seq_length, mode="infer")

  # Dataset sampler to get one sequence from specified camera
  #sampler = SequenceSampler(sequence_num, camera_num)

  # Dataloader for sequence
  seq_dataloader = DataLoader(
                              dataset = seq_dataset,
                              batch_size = 1,
                              shuffle = False,
                              )

  # Write csv header
  results_save_path = args.save + "/kitti_{}.txt".format(sequence_num)
  with open(results_save_path, mode="a+") as csv_id:
    writer = csv.writer(csv_id, delimiter=",")
    writer.writerow(["predicted x", "predicted y", "predicted theta"])

  # Run inference for each sample in sequence
  losses = []
  errors = []
  start_time = time.time()

  for i, sample in enumerate(tqdm(seq_dataloader)):
    # Format data
    # (T, N, 6, 50, 150)
    images = torch.stack([sample[ii][0] for ii in range(len(sample))]).float().to(device)
    # (T, N, 3)
    positions = torch.stack([torch.stack(sample[ii][1][0:3],1) for ii in range(len(sample))]).float().to(device) #[x, y, theta] stacked
    # (T, N,) sequence timestamps
    times = torch.stack([sample[ii][2] for ii in range(len(sample))]).float().to(device)

    # (N, 5)
    μ0s = torch.cat([torch.stack(sample[0][1][0:3],1), torch.stack(sample[0][1][3:5],1)], 1).float().to(device) 

    # Reshape images so everything can be processed in parallel by utilizing batch size 
    T, N, H, W =  images.shape[0], images.shape[1], images.shape[3], images.shape[4]
    no_seq_images = images.view(T * N, 6, H, W)

    # Forward pass
    # output (T * N, dim_output)
    vel_L_prediction = CNNModel(no_seq_images)
    
    #print("CNN predictions {}".format(vel_L_prediction[0]))
    #print("vels {}".format(vels[0][0]))

    # Decompress the results into original images format
    z_and_L_hat_list = vel_L_prediction.view(T, N, vel_L_prediction.shape[1])

    # Pass through KF
    pose_prediction = KFModel(z_and_L_hat_list, μ0s, times)

    pose_prediction_array = pose_prediction.data.cpu().numpy()[0]
    positions_array = positions.data.cpu().numpy()[0]

    loss = loss_function(pose_prediction, positions)

    # Record loss and error
    losses.append(loss.item())

    print(pose_prediction[0:10], positions[0:10])
    quit()
    # Compute and record error
    error = torch.norm(y_actual-y_prediction)
    errors.append(error.item())

    #print("Actual: {} Prediction {}".format(y_actual.data.cpu().numpy()[0], y_prediction_array))

    # Save results to file
    with open(results_save_path, mode="a+") as csv_id:
      writer = csv.writer(csv_id, delimiter=",")
      writer.writerow([y_prediction_array[0], y_prediction_array[1]])

  # Finish up
  print('Elapsed time: {}'.format(time.time() - start_time))
  print('Testing mean RMS error: {}'.format(np.mean(np.sqrt(losses))))
  print('Testing std  RMS error: {}'.format(np.std(np.sqrt(losses))))

def main():
  traj_num = args.traj_num
  print("Running inference on KITTI trajectory {}".format(traj_num))

  model_path = args.checkpoint
  camera_num = 2

  infer(model_path, int(traj_num), camera_num)

if __name__ == "__main__":
  main()
