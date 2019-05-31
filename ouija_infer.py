import numpy as np
import time
import random
import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import statistics
from synth_vis_state_est_data_generator import SynthVisStateEstDataGenerator
from matplotlib import pyplot as plt
from skimage import io

from models.feed_forward_cnn_model import FeedForwardCNN
from kitti_dataset import KittiDataset, ToTensor, SequenceSampler

# Device specification
device = torch.device('cuda')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup
image_dims = np.array((50, 150))

def infer(model_path, sequence_num):
  """
  Loads a model and infers velocities from once sequence of data
  model_path: path to model
  sequence_num: sequence ID whose velocities we want to infer
  camera_num: camera id
  """
  # Trajectory data directory
  SEQ_DIR = "/home/clairech/test_traj_" + str(sequence_num) + "/frames_post/"

  # Load model from path
  print('Loading model from: ',model_path)
  model = FeedForwardCNN(image_channels=6, image_dims=np.array((50, 150)), z_dim=2, output_covariance=False, batch_size=1)
  model = model.to(device=device)  # move model to speicified device
  checkpoint = torch.load(model_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  # Set model to eval mode
  model.eval()

  # Construct loss function
  loss_function = torch.nn.MSELoss(reduction='sum')
  
  # Write csv header
  results_save_path = "./results.txt"
  with open(results_save_path, mode="a+") as csv_id:
    writer = csv.writer(csv_id, delimiter=",")
    writer.writerow(["predicted forward vel", "predicted angular vel"])
  
  # Run inference for each sample in sequence
  losses = []
  errors = []
  start_time = time.time()
  
  num_samples = len(os.listdir(SEQ_DIR + "current/"))
  print(num_samples)

  for i in range(1, num_samples+1):
      # Load images from directory
      curr_im_path = SEQ_DIR + "current/" + str(i).zfill(4) + ".jpg"
      diff_im_path = SEQ_DIR + "diff/" + str(i).zfill(4) + ".jpg"
  
      curr_im = torch.from_numpy(io.imread(curr_im_path).transpose((2,0,1)))
      diff_im = torch.from_numpy(io.imread(diff_im_path).transpose((2,0,1)))

      # Format data
      x = torch.cat((curr_im, diff_im), 0).type('torch.FloatTensor').to(device)[None, :, :, :]
  
      # Forward pass
      y_prediction = model(x)
  
      # Extract values from tensors
      y_prediction_array = y_prediction.data.cpu().numpy()[0]

      print(y_prediction_array)
      # Save results to file
      with open(results_save_path, mode="a+") as csv_id:
        writer = csv.writer(csv_id, delimiter=",")
        writer.writerow([y_prediction_array[0], y_prediction_array[1]])
  
  # Finish up
  print('Elapsed time: {}'.format(time.time() - start_time))
  print('Testing mean RMS error: {}'.format(np.mean(np.sqrt(losses))))
  print('Testing std  RMS error: {}'.format(np.std(np.sqrt(losses))))

def main():
  print("Running inference on trajectory")
  model_path = "./log/2019-05-24_11_24_1.00e-04_bestloss_feed_forward.tar"
  sequence_num = 7

  infer(model_path, sequence_num)

if __name__ == "__main__":
  main()
