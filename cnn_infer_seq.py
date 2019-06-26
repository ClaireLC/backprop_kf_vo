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
from kitti_dataset import KittiDataset, ToTensor, SequenceSampler
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
  #seq_dataset = KittiDatasetSeq(SEQ_DIR, POSES_DIR, OXTS_DIR, seq_length, mode="infer")

  # Individual frames dataset
  dataset = KittiDataset(SEQ_DIR, POSES_DIR, OXTS_DIR, transform=transforms.Compose([ToTensor()]), mode="infer")
  # Dataset sampler to get one sequence from specified camera
  sampler = SequenceSampler(sequence_num, camera_num)
  # Dataloader for sequence
  dataloader = DataLoader(
                              dataset = dataset,
                              batch_size = 1,
                              sampler = sampler,
                              shuffle = False,
                              )

  ## Dataloader for sequence
  #seq_dataloader = DataLoader(
  #                            dataset = seq_dataset,
  #                            batch_size = 1,
  #                            shuffle = False,
  #                            )

  # Write csv header
  results_save_path = args.save + "/kitti_{}.txt".format(sequence_num)
  with open(results_save_path, mode="a+") as csv_id:
    writer = csv.writer(csv_id, delimiter=",")
    writer.writerow(["predicted x", "predicted y", "predicted theta"])

  # Run inference for each sample in sequence
  losses = []
  errors = []
  #start_time = time.time()

  # Get initial state of first frame in sequence
  # (5)
  init_sample = next(iter(dataloader))
  mu0 = torch.cat([init_sample["pose"],init_sample["vel"]], 1).type('torch.FloatTensor').to(device) 
  #μ0s = torch.cat([torch.stack(init_sample["pose"],1), torch.stack(init_sample["vel"],1)], 1).float().to(device) 
  prev_time = init_sample["curr_time"].type('torch.FloatTensor').to(device)
  print("Init state: {}".format(mu0))
  print("Init time: {}".format(prev_time))
  
  mu = mu0
  sig = torch.eye(5).to(device)

  for i, sample in enumerate(tqdm(dataloader)):
    # Skip frame 1
    if i == 0:
      continue

    print("\nSample number {}".format(i))

    # Format data
    # (N, 6, 50, 150)
    image = torch.cat((sample["curr_im"], sample["diff_im"]), 1).type('torch.FloatTensor').to(device)
    # (N, 3)
    pose = sample["pose"].type('torch.FloatTensor').to(device)
    #pose = torch.unsqueeze(sample["pose"],0).type('torch.FloatTensor').to(device)
    # (T,)
    curr_time = sample["curr_time"].type('torch.FloatTensor').to(device)

    # Create list of prev_time and curr_time to input into KFModel
    times = torch.unsqueeze(torch.cat([prev_time, curr_time], 0),1).type('torch.FloatTensor').to(device)
    print("times {}".format(times))


    # Forward pass
    # output (N, dim_output)
    vel_L_prediction = CNNModel(image)
    
    #print("CNN predictions {}".format(vel_L_prediction))
    #print("vels {}".format(vels[0][0]))

    # Add dimension to vel_L_prediction
    vel_L_prediction = vel_L_prediction.view(1, 1, vel_L_prediction.shape[1])

    # Pass through KF
    mu, sig = KFModel(vel_L_prediction, mu, sig, prev_time, curr_time)
    mu = torch.squeeze(mu,0)
    sig = torch.squeeze(sig,0)

    # Update prev_time
    prev_time = curr_time

    print("mu {}".format(mu))
    pose_prediction = mu[:,0:3]
    pose_prediction_array = pose_prediction.data.cpu().numpy()[0]
    pose_array = pose.data.cpu().numpy()[0]

    print(pose_prediction.shape, pose.shape)
    loss = loss_function(pose_prediction, pose)

    # Record loss and error
    losses.append(loss.item())

    # Compute and record error
    error = torch.norm(pose - pose_prediction)
    errors.append(error.item())

    print("Actual: {} Prediction {}".format(pose_array, pose_prediction_array))

    #if i == 100:
    #  break

    # Save results to file
    #with open(results_save_path, mode="a+") as csv_id:
    #  writer = csv.writer(csv_id, delimiter=",")
    #  writer.writerow([y_prediction_array[0], y_prediction_array[1]])

  # Finish up
  #print('Elapsed time: {}'.format(time.time() - start_time))
  print('Testing mean RMS error: {}'.format(np.mean(np.sqrt(errors))))
  print('Testing std  RMS error: {}'.format(np.std(np.sqrt(errors))))

def main():
  traj_num = args.traj_num
  print("Running inference on KITTI trajectory {}".format(traj_num))

  model_path = args.checkpoint
  camera_num = 2

  infer(model_path, int(traj_num), camera_num)

if __name__ == "__main__":
  main()
