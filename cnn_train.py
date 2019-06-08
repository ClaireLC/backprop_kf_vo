import numpy as np
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils
from kitti_dataset import KittiDataset, SubsetSampler, ToTensor
from models.feed_forward_cnn_model import FeedForwardCNN

# Dataset specifications
SEQ_DIR = "/mnt/disks/dataset/dataset_post/sequences/"
POSES_DIR = "/mnt/disks/dataset/dataset_post/poses/"
OXTS_DIR = "/mnt/disks/dataset/dataset_post/oxts/"

# Device specification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global parameters
batch_size = 50
epochs = 100

def train_model(model, optimizer, loss_function, lr=1e-4, starting_epoch=-1, model_id=None,
  train_dataloader=None, val_dataloader=None):
  """
    starting_epoch: the epoch to start training. If -1, this means we
                    start training model from scratch.
    model_id: timestamp of model whose checkpoint we want to load
  """
  lr_str = "{0:.2e}".format(lr)
  print("Training feed forward CNN with lr =", lr_str)

  # Create loss_file name using starting time to log training process
  if starting_epoch >= 0:
    start_time = model_id
  else:
    start_time = time.time()
    start_time_str = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d_%H_%M')

  # Logs all files
  loss_file = 'log/' + start_time_str + '_lr_' + lr_str + '_loss.txt'
  val_loss_file = 'log/' + start_time_str + '_lr_' + lr_str + '_val_loss.txt'

  # If we are starting from a saved checkpoint epoch, load that checkpoint
  if starting_epoch >= 0:
    checkpoint_path = "log/" + start_time_str + "_" + str(starting_epoch) + "_feed_forward.tar"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(epoch, loss)


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
    model.train()

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
        if i_batch % (len(train_dataloader) // 10) == 0:
            print('epoch {}/{}, iteration {}/{}, loss = {}'.format(epoch, (epochs-1), i_batch, len(train_dataloader) - 1, loss.item()))
            losses.append(loss.item())
            current_error = torch.norm(y_prediction-y_actual)
            errors.append(current_error)

            # Log info in loss_file
            out_text = "{}, {}, {}, {}\n".format(epoch, i_batch, loss.item(), current_error)
            with open(loss_file, "a+") as loss_save:
              loss_save.write(out_text)

    # End of epoch
    val_loss = validation_loss(model, val_dataloader, loss_function)
    print()
    print('epoch {}, validation loss = {}'.format(epoch, val_loss))
    print()
    with open(val_loss_file, "a+") as val_loss_save:
      val_loss_save.write("{}, {}\n".format(epoch, val_loss))

    # Save the best model after each epoch based on the lowest achieved validation loss
    if lowest_loss is None or lowest_loss > val_loss:
      lowest_loss = val_loss
      model_name = 'log/' + start_time_str + '_' + lr_str  + '_bestloss_feed_forward.tar'
      torch.save({
                  "epoch": epoch,
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "loss": loss.item(),
                  "batch_size": batch_size,
                  }, model_name)

  # Finish up. End of training
  print('elapsed time: {}'.format(time.time() - start_time))
  model_name = 'log/' + start_time_str + '_' + lr_str +  '_end_feed_forward.tar'
  torch.save({
              "epoch": epochs, # the end
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "loss": loss.item(),
              "batch_size": batch_size,
              }, model_name)
  print('saved model: '+ model_name)


def validation_loss(model, val_dataloader, loss_function):
  with torch.no_grad():
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for i_batch, sample_batched in enumerate(val_dataloader):
        num_batches += 1
        # Format data
        x = torch.cat((sample_batched["curr_im"], sample_batched["diff_im"]), 1).type('torch.FloatTensor').to(device)
        y_actual = sample_batched["vel"].type('torch.FloatTensor').to(device)

        # Forward pass
        y_prediction = model(x)
        # Compute loss
        loss = loss_function(y_prediction, y_actual).item()
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
  train_dataset = KittiDataset(SEQ_DIR, POSES_DIR, OXTS_DIR, transform=transforms.Compose([ToTensor()]), mode="train")
  val_dataset = KittiDataset(SEQ_DIR, POSES_DIR, OXTS_DIR, transform=transforms.Compose([ToTensor()]), mode="val")
  sampler = SubsetSampler(20)

  train_dataloader = create_dataloaders(train_dataset, batch_size)
  val_dataloader = create_dataloaders(val_dataset, batch_size)
  dataloader_sampler = create_dataloaders(train_dataset, batch_size, sampler)
  print("Done.")

  # Construct feed forward CNN model
  model = FeedForwardCNN(image_channels=6, image_dims=np.array((50, 150)), z_dim=2, output_covariance=False, batch_size=batch_size)
  model = model.to(device)  # move model to speicified device
  print(model)

  # Construct loss function and optimizer
  loss_function = torch.nn.MSELoss(reduction='sum')

  learning_rates = [1e-3, 1e-4]
  for learning_rate in learning_rates:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)
    train_model(model, optimizer, loss_function, lr=learning_rate, starting_epoch=-1, train_dataloader=train_dataloader, val_dataloader=val_dataloader)


if __name__ == "__main__":
  main()
