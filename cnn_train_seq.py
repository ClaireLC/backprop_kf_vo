import numpy as np
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils
from kitti_dataset import KittiDataset, SubsetSampler
from kitti_dataset_seq import KittiDatasetSeq
from models.feed_forward_cnn_model import FeedForwardCNN
from models.kalmanfilter_model import KalmanFilter
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--load', dest='load', default=None, help='Directory of model checkpoint to load')

args = parser.parse_args()
checkpoint_dir = args.load

# Dataset specifications
SEQ_DIR = "/mnt/disks/dataset/dataset_post/sequences/"
POSES_DIR = "/mnt/disks/dataset/dataset_post/poses/"
OXTS_DIR = "/mnt/disks/dataset/dataset_post/oxts/"
SAVE_DIR = "/mnt/disks/dataset/"

# Pretrained weights path
CNN_DIR = "/home/clairech/backprop_kf_vo/log/cnnForwardPass/"
CNN_NAME = "2019-05-24_11_24_1.00e-04_bestloss_feed_forward.tar"

# LOG DIRECTORY
LOG_DIR = "./log/"

# Device specification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global parameters
batch_size = 20
epochs = 150

seq_length = 100

#learning_rates = [1e-3, 1e-4]
learning_rates = [1e-4]


def predict(CNNModel, KFModel, sample_batched, loss_function):
  """
  Takes in the models and the batched sample
  Returns the loss, and positions, and pose predictions
  """
  # Sample dimensions (T, 3, N, 6, 50, 150)
  # 3 because each sample is a tuple (image, state, time)

  # Format data
  # Images in sequence, not including image 0 (T-1, N, 6, 50, 150)
  images = torch.stack([sample_batched[ii][0] for ii in range(1, len(sample_batched))]).float().to(device)

  # make the column order (pose, velocities)
  # (N, 5)
  μ0s = torch.cat([torch.stack(sample_batched[0][1][0:3],1), torch.stack(sample_batched[0][1][3:5],1)], 1).float().to(device) 
  # initial times (N,)
  t0 = torch.stack([sample_batched[0][2]]).float().to(device)

  sig = torch.eye(5).to(device)

  # (T, N, 3)
  positions = torch.stack([torch.stack(sample_batched[ii][1][0:3],1) for ii in range(1, len(sample_batched))]).float().to(device) #[x, y, theta] stacked

  # (T, N, 2)
  vels = torch.stack([torch.stack(sample_batched[ii][1][3:5],1) for ii in range(1, len(sample_batched))]).float().to(device) #[x, y, theta] stacked

  # (T, N,) sequence timestamps
  times = torch.stack([sample_batched[ii][2] for ii in range(1, len(sample_batched))]).float().to(device)

  # Reshape images so everything can be processed in parallel by utilizing batch size 
  T, N, H, W =  images.shape[0], images.shape[1], images.shape[3], images.shape[4]
  no_seq_images = images.view(T * N, 6, H, W)

  # Forward pass
  # output (T * N, dim_output)
  vel_L_prediction = CNNModel(no_seq_images)

  # Decompress the results into original images format
  z_and_L_hat_list = vel_L_prediction.view(T, N, vel_L_prediction.shape[1])

  # Pass through KF
  state_prediction, sig = KFModel(z_and_L_hat_list, μ0s, sig, t0, times)

  # Extract pose prediction  from state_prediction (first 3 elements)
  pose_prediction = state_prediction[:,:,0:3]

  # Compute loss
  loss = loss_function(pose_prediction, positions)

  #print("init mu: {}".format(μ0s))
  #print("init time: {}".format(t0))
  #print("first positions: {}".format(positions[0:5,:,:]))
  #print("first times: {}".format(times[0:5,:]))
  #print("first vel predictions: {}".format(vel_L_prediction[0:5,:]))
  #print("first vel actual: {}".format(vels[0:5,:,:]))
  #print("z and L hat list: {}".format(vels[0:5,:,:]))
  #print("state pred: {}".format(state_prediction[0:5,:,:]))
  #print("loss: {}".format(loss))
  #
  #quit()
  
  return loss, positions, pose_prediction

def train_model(CNNModel, KFModel, optimizer, loss_function, lr=1e-4, starting_epoch=-1,
  train_dataloader=None, val_dataloader=None, checkpoint_dir=None):
  """
    starting_epoch: the epoch to start training. If -1, this means we
                    start training model from scratch.
    model_id: timestamp of model whose checkpoint we want to load
  """
  lr_str = "{0:.2e}".format(lr)
  print("Training feed forward CNN with lr =", lr_str)

  start_time = time.time()
  start_time_str = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d_%H_%M')

  # If a checkpoint path is specified, load model
  if checkpoint_dir:
    checkpoint_fname = checkpoint_dir + "best_loss.tar"
    print("Loading checkpoint {}".format(checkpoint_fname))
    checkpoint = torch.load(checkpoint_fname)
    CNNModel.load_state_dict(checkpoint['cnn_model_state_dict'])
    KFModel.load_state_dict(checkpoint['kf_model_state_dict'])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("Epoch {}: Loss {}".format(epoch, loss))
    starting_epoch = epoch
  else:
    # Set model name with unique start time
    model_name = "e2e_" + start_time_str + "_lr." + lr_str
    checkpoint_dir = LOG_DIR + model_name + "/"

    print("Training new model named: {}".format(model_name))

    starting_epoch = -1
    # Load pre-trained feed forward weights for just conv layers
    print("Loading pretrained CNN weights from {}".format(CNN_NAME))
    if starting_epoch < 0:
      cnn_weights_path = CNN_DIR + CNN_NAME
      checkpoint = torch.load(cnn_weights_path)

      # Change dimensions of last fully connected layer to be output dim 5
      # initialize weights from uniform random distribution
      old_fc_weight = checkpoint['model_state_dict']['model.18.weight']
      old_fc_bias = checkpoint['model_state_dict']['model.18.bias']
      #new_fc_weight = torch.nn.init.uniform_(torch.zeros([3,128]), -0.2, 0.2).to(device)
      #new_fc_bias = torch.nn.init.uniform_(torch.zeros([3]), -0.2, 0.2).to(device)
      new_fc_weight = torch.zeros([3,128]).to(device)
      new_fc_bias = torch.zeros([3]).to(device)
      checkpoint['model_state_dict']['model.18.weight'] = torch.cat([old_fc_weight, new_fc_weight], 0).to(device)
      checkpoint['model_state_dict']['model.18.bias'] = torch.cat([old_fc_bias, new_fc_bias], 0).to(device)
      
      CNNModel.load_state_dict(checkpoint['model_state_dict'])

  # Log files for losses
  loss_file         = checkpoint_dir+ "trn_loss.txt"
  val_loss_file     = checkpoint_dir+ "val_loss.txt"
  best_loss_file    = checkpoint_dir+ "best_loss.tar"
  end_loss_file     = checkpoint_dir+ "end_loss.tar"

  # Make model directory if one does not exist
  os.makedirs(checkpoint_dir, exist_ok=True)

  # Training
  losses = []
  errors = []
  with open(loss_file, "a+") as loss_save:
    loss_save.write('epoch, iteration, loss, error\n')
  with open(val_loss_file, "a+") as loss_save:
    loss_save.write('epoch, val_loss\n')

  lowest_loss = None

  for epoch in range(starting_epoch + 1, epochs):
    # Set model to training model
    CNNModel.train()
    KFModel.train()

    for i_batch, sample_batched in enumerate(train_dataloader):
        # Compute loss
        loss, positions,pose_prediction = predict(CNNModel, KFModel, sample_batched, loss_function)

        # Backward pass()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if i_batch % (len(train_dataloader) // 10) == 0:
            print('epoch {}/{}, iteration {}/{}, loss = {}'.format(epoch, (epochs-1), i_batch, len(train_dataloader) - 1, loss.item()))
            losses.append(loss.item())
            current_error = torch.norm(positions-pose_prediction)
            errors.append(current_error)

            # Log info in loss_file
            out_text = "{}, {}, {}, {}\n".format(epoch, i_batch, loss.item(), current_error)
            with open(loss_file, "a+") as loss_save:
              loss_save.write(out_text)

    # End of epoch
    val_loss = validation_loss(CNNModel, KFModel, val_dataloader, loss_function)
    print()
    print('epoch {}, validation loss = {}'.format(epoch, val_loss))
    print()
    with open(val_loss_file, "a+") as val_loss_save:
      val_loss_save.write("{}, {}\n".format(epoch, val_loss))

    # Save the best model after each epoch based on the lowest achieved validation loss
    if lowest_loss is None or lowest_loss > val_loss:
      lowest_loss = val_loss
      torch.save({
                  "epoch": epoch,
                  "cnn_model_state_dict": CNNModel.state_dict(),
                  "kf_model_state_dict": KFModel.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "loss": loss.item(),
                  "batch_size": batch_size,
                  }, best_loss_file)
      print("Saved model with best validation loss so far to: "+ best_loss_file)

  # Finish up. End of training
  print('elapsed time: {}'.format(time.time() - start_time))
  model_name = 'log/' + start_time_str + '_' + lr_str +  '_end_feed_forward.tar'
  torch.save({
              "epoch": epochs, # the end
              "cnn_model_state_dict": CNNModel.state_dict(),
              "kf_model_state_dict": KFModel.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "loss": loss.item(),
              "batch_size": batch_size,
              }, end_loss_file)
  print("Saved final epoch model to: "+ end_loss_file)


def validation_loss(CNNModel, KFModel, val_dataloader, loss_function):
  with torch.no_grad():
    CNNModel.eval()
    KFModel.eval()
    total_loss = 0.0
    num_batches = 0
    for i_batch, sample_batched in enumerate(val_dataloader):
        num_batches += 1
        loss, positions, pose_prediction = predict(CNNModel, KFModel, sample_batched, loss_function)
        total_loss += loss

  # Return the average validation loss across all batches
  return total_loss / num_batches

def create_dataloaders(dataset, batch_size, sampler=None):
  # Load dataset
  if sampler is None:
    dataloader = DataLoader(
                            dataset = dataset,
                            batch_size = batch_size,
                            )
  else:
    dataloader = DataLoader(
                            dataset = dataset,
                            batch_size = batch_size,
                            sampler = sampler,
                            shuffle = False,
                            )
  return dataloader


def main():
  print("Creating dataloaders...")
  # Create dataset
  train_dataset = KittiDatasetSeq(SEQ_DIR, POSES_DIR, OXTS_DIR, seq_length, mode="train")
  val_dataset = KittiDatasetSeq(SEQ_DIR, POSES_DIR, OXTS_DIR, seq_length, mode="val")
  #sampler = SubsetSampler(20)

  train_dataloader = create_dataloaders(train_dataset, batch_size)
  val_dataloader = create_dataloaders(val_dataset, batch_size)
  #dataloader_sampler = create_dataloaders(train_dataset, batch_size, sampler)
  print("Done creating dataloaders.")

  # Construct feed forward CNN model
  CNNModel = FeedForwardCNN(image_channels=6, image_dims=np.array((50, 150)), z_dim=2, output_covariance=True, batch_size=batch_size).to(device)

  KFModel = KalmanFilter(device).to(device)

  #print("CNN Model:")
  #print(CNNModel)

  # Construct loss function and optimizer
  loss_function = torch.nn.MSELoss(reduction='sum')
 
  for learning_rate in learning_rates:
    optimizer = torch.optim.Adam(list(CNNModel.parameters()) + list(KFModel.parameters()), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    train_model(CNNModel, KFModel, optimizer, loss_function, lr=learning_rate, starting_epoch=-1, train_dataloader=train_dataloader, val_dataloader=val_dataloader, checkpoint_dir=checkpoint_dir)

if __name__ == "__main__":
  main()
