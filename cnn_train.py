import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils
from kitti_dataset import KittiDataset, ToTensor, SubsetSampler
from models.feed_forward_cnn_model import FeedForwardCNN

# Dataset specifications
SEQ_DIR = "/mnt/disks/dataset/dataset_post/sequences/"
POSES_DIR = "/mnt/disks/dataset/dataset_post/poses/"
OXTS_DIR = "/mnt/disks/dataset/dataset_post/oxts/"

# Device specification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global parameters
batch_size = 4
epochs = 100

def train_model(model, optimizer, loss_function, lr=1e-4, starting_epoch=-1, model_id=None,
  train_dataloader=None, val_dataloader=None, dataloader_sampler=None):
  """
    starting_epoch: the epoch to start training. If -1, this means we
                    start training model from scratch.
    model_id: timestamp of model whose checkpoint we want to load
  """

  print("Training feed forward CNN with lr =", str(lr))

  # Create loss_file name using starting time to log training process
  if starting_epoch >= 0:
    start_time = model_id
  else:
    start_time = time.time()

  # Logs all files
  loss_file = 'log/' + str(int(start_time)) + '_lr_' + str(lr) + '_loss.txt'

  # If we are starting from a saved checkpoint epoch, load that checkpoint
  if starting_epoch >= 0:
    checkpoint_path = "log/" + str(int(start_time)) + "_" + str(starting_epoch) + "_feed_forward.tar"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(epoch, loss)

  # Set model to training model
  model.train()


  # Training
  losses = []
  errors = []
  with open(loss_file, "a+") as loss_save:
    loss_save.write('epoch, iteration, loss, error\n')

  lowest_loss = None

  for epoch in range(starting_epoch + 1, epochs):
      for i_batch, sample_batched in enumerate(train_dataloader):
          # Format data
          x = torch.cat((sample_batched["curr_im"], sample_batched["diff_im"]), 1).type('torch.FloatTensor').to(device)
          y_actual = sample_batched["vel"].type('torch.FloatTensor').to(device)

          # Forward pass
          y_prediction = model(x)

          # Compute loss
          loss = loss_function(y_prediction, y_actual)

          # Backward pass()
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # Print loss
          if i_batch % 100 == 0:
              print('epoch {}/{}, iteration {}/{}, loss = {}'.format(epoch, (epochs-1), i_batch, int(len(train_dataloader) / batch_size - 1), loss.item()))
              losses.append(loss.item())
              current_error = torch.norm(y_prediction-y_actual)
              errors.append(current_error)

              # Log info in loss_file
              out_text = "{}, {}, {}, {}\n".format(epoch, i_batch, loss.item(), current_error)
              with open(loss_file, "a+") as loss_save:
                loss_save.write(out_text)

      # Save the best model after each epoch based on the lowest achieved loss
      if lowest_loss is None or lowest_loss > loss:
        lowest_loss = loss
        model_name = 'log/' + str(int(start_time)) + '_bestloss_feed_forward.tar'
        torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "batch_size": batch_size,
                    }, model_name)

  # Finish up
  print('elapsed time: {}'.format(time.time() - start_time))
  model_name = 'log/' + str(int(start_time)) + '_end_feed_forward.tar'
  torch.save({
              "epoch": epochs, # the end
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "loss": loss.item(),
              "batch_size": batch_size,
              }, model_name)
  print('saved model: '+ model_name)

def create_dataloaders(dataset, batch_size, sampler=None):
  # Load dataset
  if sampler is not None:
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
  train_dataset = KittiDataset(SEQ_DIR, POSES_DIR, OXTS_DIR, transform=transforms.Compose([ToTensor()]), train=True, val_idx=9)
  val_dataset = KittiDataset(SEQ_DIR, POSES_DIR, OXTS_DIR, transform=transforms.Compose([ToTensor()]), train=False, val_idx=9)
  sampler = SubsetSampler(20)

  train_dataloader = create_dataloaders(train_dataset, batch_size)
  dataloader_sampler = create_dataloaders(train_dataset, batch_size, sampler)
  val_dataloader = create_dataloaders(val_dataset, batch_size)
  print("Done.")

  # Construct feed forward CNN model
  model = FeedForwardCNN(image_channels=6, image_dims=np.array((50, 150)), z_dim=2, output_covariance=False, batch_size=batch_size)
  model = model.to(device)  # move model to speicified device
  print(model)

  # Construct loss function and optimizer
  loss_function = torch.nn.MSELoss(reduction='sum')
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

  train_model(model, optimizer, loss_function, lr=1e-3, starting_epoch=-1, train_dataloader=dataloader_sampler, dataloader_sampler=dataloader_sampler)

if __name__ == "__main__":
  main()