"""
Run inferrence on a KITTI trajectory
"""
import numpy as np
import time
import random
import csv
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import statistics
from synth_vis_state_est_data_generator import SynthVisStateEstDataGenerator
from matplotlib import pyplot as plt

from models.feed_forward_cnn_model import FeedForwardCNN
from kitti_dataset import KittiDataset, ToTensor, SequenceSampler

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', dest='checkpoint', default='', help='model checkpoint')
parser.add_argument('--save', dest='save', default='./cnn_results/', help='save location')
parser.add_argument("--traj_num", dest='traj_num', default='0', help="Trajectory number")
  
args = parser.parse_args()

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
  model = FeedForwardCNN(image_channels=6, image_dims=np.array((50, 150)), z_dim=2, output_covariance=False, batch_size=1)
  model = model.to(device=device)  # move model to speicified device
  checkpoint = torch.load(model_path, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  # Set model to eval mode
  model.eval()

  # Construct loss function
  loss_function = torch.nn.MSELoss(reduction='sum')
  
  # Create dataset
  dataset = KittiDataset(SEQ_DIR, POSES_DIR, OXTS_DIR, transform=transforms.Compose([ToTensor()]), mode="infer")

  # Dataset sampler to get one sequence from specified camera
  sampler = SequenceSampler(sequence_num, camera_num)

  # Dataloader for sequence
  seq_dataloader = DataLoader(
                              dataset = dataset,
                              batch_size = 1,
                              sampler = sampler,
                              shuffle = False,
                              )

  # Write csv header
  results_save_path = args.save + "/kitti_{}.txt".format(sequence_num)
  with open(results_save_path, mode="a+") as csv_id:
    writer = csv.writer(csv_id, delimiter=",")
    writer.writerow(["predicted forward vel", "predicted angular vel"])
  
  # Run inference for each sample in sequence
  losses = []
  errors = []
  start_time = time.time()
  
  for i, sample in enumerate(tqdm(seq_dataloader)):
      # Format data
      x = torch.cat((sample["curr_im"], sample["diff_im"]), 1).type('torch.FloatTensor').to(device)
      y_actual = sample["vel"].type('torch.FloatTensor').to(device)
  
      # Forward pass
      y_prediction = model(x)
      loss = loss_function(y_prediction, y_actual)
  
      # Extract values from tensors
      y_prediction_array = y_prediction.data.cpu().numpy()[0]

      # Record loss and error
      losses.append(loss.item())

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
