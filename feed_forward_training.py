import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from feed_forward_cnn import FeedForwardCNN
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from kitti_dataset import KittiDataset, ToTensor

# Device specification
device = torch.device('cpu')
#device = torch.device('cuda')

# Dataset specifications
seq_dir = "/mnt/disks/dataset/dataset_post/sequences/"
poses_dir = "/mnt/disks/dataset/dataset_post/poses/"
oxts_dir = "/mnt/disks/dataset/dataset_post/oxts/"
dataset = KittiDataset(seq_dir, poses_dir, oxts_dir, transform=transforms.Compose([ToTensor()]))

# Setup
# seed(0)
num_samples = 100
seq_length = 100
num_obstacles = 10
image_size = 128
image_dims = np.array((image_size, image_size))

# Load dataset
batch_size = 4
dataloader = DataLoader(
                        dataset = dataset,
                        batch_size = batch_size,
                       )
  
# Construct feed forward CNN model
image_channels = 6
z_dim = 2
output_covariance = False
model = FeedForwardCNN(image_channels, image_dims, z_dim, output_covariance, batch_size)
print(model)
model = model.to(device=device)  # move model to speicified device

# Construct loss function and optimizer
loss_function = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Training
epochs = 10
losses = []
errors = []
start_time = time.time()
loss_file = 'claire_models/'+str(int(start_time))+'_loss.txt'
with open(loss_file, "a+") as loss_save:
  loss_save.write('epoch, iteration, loss, error\n')
for e in range(epochs):
    for i_batch, sample_batched in enumerate(dataloader):
        # # Extract training pairs from sequence of sequences (from Haruki)
        # x = torch.stack([minibatch[ii][0] for ii in range(len(minibatch))]).float().permute(0,1,4,2,3).squeeze(1).to(device)
        # y_z = torch.stack([minibatch[ii][1][:, 0:2] for ii in range(len(minibatch))]).float().to(device)

        # Format data
        x = torch.cat((sample_batched["curr_im"], sample_batched["diff_im"]), 1).to(device).type('torch.FloatTensor')
        y_actual = sample_batched["vel"].to(device).type('torch.FloatTensor')
        #print(y_actual)

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
            print('epoch {}/{}, iteration {}/{}, loss = {}'.format(e,(epochs-1),i_batch,int(len(dataset)/batch_size-1),loss.item()))
            losses.append(loss.item())
            current_error = torch.norm(y_prediction-y_actual)
            errors.append(current_error)
            out_text = "{}, {}, {}, {}\n".format(e, i_batch, loss.item(), current_error)
            with open(loss_file, "a+") as loss_save:
              loss_save.write(out_text)
    # Save current model file after epoch
    model_name = 'claire_models/'+str(int(start_time))+'_'+str(e)+'_feed_forward'
    torch.save(model, model_name)

# Finish up
print('elapsed time: {}'.format(time.time() - start_time))
model_name = 'claire_models/'+str(int(start_time))+'_feed_forward'
torch.save(model, model_name)
print('saved model: '+model_name)
