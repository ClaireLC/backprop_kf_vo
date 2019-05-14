import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision import transforms, utils
from kitti_dataset import KittiDataset, ToTensor

class Flatten(nn.Module):

  def forward(self, x):
    N = x.shape[0]
    return x.view(N,-1)

class FeedForwardCNN(nn.Module):

  def __init__(self, image_channels=6, image_dims=np.array([150, 50]), z_dim=2, output_covariance=False, batch_size = 1):
    super(FeedForwardCNN, self).__init__()

    # Set output vector size based on dimension of z and if covariance is required
    output_dim = 0
    if output_covariance:
      L_dim = 0
      for l in range(z_dim):
        L_dim += (z_dim-l)
      output_dim = z_dim + L_dim
    else:
      output_dim = z_dim

    # Intermediate channel definitions
    # conv 1
    conv1_in = image_channels
    conv1_kernel_size = (7,7)
    conv1_stride = (1,1)
    conv1_pad = (3,3)
    conv1_out = 16

    # conv2
    conv2_in = conv1_out
    conv2_kernel_size = (5,5)
    conv2_stride = (2,1)
    conv2_pad = (3,2)
    conv2_out = 16

    # conv3
    conv3_in = conv2_out
    conv3_kernel_size = (5,5)
    conv3_stride = (2,1)
    conv3_pad = (6,2)
    conv3_out = 16

    # conv3
    conv4_in = conv3_out
    conv4_kernel_size = (5,5)
    conv4_stride = (2,2)
    conv4_pad = (3,3)
    conv4_out = 16

    H_1 = int(1+(image_dims[1]-conv1_kernel_size[1]+2*conv1_pad[1])/conv1_stride[1])
    W_1 = int(1+(image_dims[0]-conv1_kernel_size[0]+2*conv1_pad[0])/conv1_stride[0])
    print(W_1, H_1)

    H_2 = int(1+(H_1-conv2_kernel_size[1]+2*conv2_pad[1])/conv2_stride[1])
    W_2 = int(1+(W_1-conv2_kernel_size[0]+2*conv2_pad[0])/conv2_stride[0])
    print(W_2, H_2)

    H_3 = int(1+(H_2-conv3_kernel_size[1]+2*conv3_pad[1])/conv3_stride[1])
    W_3 = int(1+(W_2-conv3_kernel_size[0]+2*conv3_pad[0])/conv3_stride[0])
    print(W_3, H_3)

    H_4 = int(1+(H_3-conv4_kernel_size[1]+2*conv4_pad[1])/conv4_stride[1])
    W_4 = int(1+(W_3-conv4_kernel_size[0]+2*conv4_pad[0])/conv4_stride[0])
    print(W_4, H_4)

    # Define sequential 3D convolutional neural network model
    self.model = nn.Sequential(
        # conv1
        nn.Conv2d(in_channels=conv1_in,
                  out_channels=conv1_out,
                  kernel_size=conv1_kernel_size,
                  stride=conv1_stride,
                  padding=conv1_pad,
                  dilation=1,
                  groups=1,
                  bias=True),
        nn.ReLU(),
        nn.LayerNorm([batch_size, 16, 50, 150], eps=1e-05, elementwise_affine=True),
        #nn.LayerNorm(H_1*W_1, eps=1e-05, elementwise_affine=True),

        # conv2
        nn.Conv2d(in_channels=conv2_in,
                  out_channels=conv2_out,
                  kernel_size=conv2_kernel_size,
                  stride=conv2_stride,
                  padding=conv2_pad,
                  dilation=1,
                  groups=1,
                  bias=True),
        nn.ReLU(),
        nn.LayerNorm([batch_size, 16, 26, 150], eps=1e-05, elementwise_affine=True),
        #nn.LayerNorm(H_2*W_2, eps=1e-05, elementwise_affine=True),

        # conv3
        nn.Conv2d(in_channels=conv3_in,
                  out_channels=conv3_out,
                  kernel_size=conv3_kernel_size,
                  stride=conv3_stride,
                  padding=conv3_pad,
                  dilation=1,
                  groups=1,
                  bias=True),
        nn.ReLU(),
        #nn.LayerNorm(H_3*W_3, eps=1e-05, elementwise_affine=True),
        nn.LayerNorm([batch_size, 16, 17, 150], eps=1e-05, elementwise_affine=True),

        # conv4
        nn.Conv2d(in_channels=conv4_in,
                  out_channels=conv4_out,
                  kernel_size=conv4_kernel_size,
                  stride=conv4_stride,
                  padding=conv4_pad,
                  dilation=1,
                  groups=1,
                  bias=True),
        nn.ReLU(),
        nn.LayerNorm([batch_size, 16, 10, 76], eps=1e-05, elementwise_affine=True),
        #nn.LayerNorm(H_4*W_4, eps=1e-05, elementwise_affine=True),
        nn.Dropout(p=0.9),
        Flatten(),
        # Fully connected layers
        nn.Linear(12160, 128, bias=True),
        #nn.Linear(conv4_out*H_4*W_4, 128, bias=True),
        nn.Linear(128, 128, bias=True),
        nn.Linear(128, output_dim, bias=True)
      )

  def forward(self,x):
    y_pred = self.model(x)
    return y_pred
  
  def train_model(self, starting_epoch = -1, model_id = None):
    """
      starting_epoch: the epoch to start training. If -1, this means we
                      start training model from scratch.
      model_id: timestamp of model whose checkpoint we want to load
    """
  
    print("Training feed forward CNN")
    # Device specification
    device = torch.device('cpu')
    #device = torch.device('cuda')
    
    # Tensorboard writer
    writer = SummaryWriter()
    
    # Dataset specifications
    seq_dir = "/mnt/disks/dataset/dataset_post/sequences/"
    poses_dir = "/mnt/disks/dataset/dataset_post/poses/"
    oxts_dir = "/mnt/disks/dataset/dataset_post/oxts/"
    dataset = KittiDataset(seq_dir, poses_dir, oxts_dir, transform=transforms.Compose([ToTensor()]))
    
    # Load dataset
    batch_size = 10
    dataloader = DataLoader(
                            dataset = dataset,
                            batch_size = batch_size,
                           )
      
    # Construct feed forward CNN model
    image_dims = np.array((150, 50))
    image_channels = 6
    z_dim = 2
    output_covariance = False
    model = FeedForwardCNN(image_channels, image_dims, z_dim, output_covariance, batch_size)
    print(model)
    model = model.to(device=device)  # move model to speicified device

    # Construct loss function and optimizer
    loss_function = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
  
    # File saving info
    if starting_epoch >= 0:
      start_time = model_id
    else:
      start_time = time.time()

    loss_file = 'claire_models/'+str(int(start_time))+'_loss.txt'
    
    # If we are starting from a saved checkpoint epoch, load that checkpoint
    if starting_epoch >= 0:
      checkpoint_path = "claire_models/" + str(int(start_time)) + "_" + str(starting_epoch)+ "_feed_forward"
      checkpoint = torch.load(checkpoint_path)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      epoch = checkpoint["epoch"]
      loss = checkpoint["loss"]
      print(epoch, loss)
    model.train() # Set model to training model
    
    # Tensorboard model
    #dummy_input = torch.rand(4,6,50,150)
    #writer.add_graph(model,dummy_input)
    
    # Training
    epochs = 10
    losses = []
    errors = []
    with open(loss_file, "a+") as loss_save:
      loss_save.write('epoch, iteration, loss, error\n')
    for e in range(starting_epoch + 1, epochs):
        for i_batch, sample_batched in enumerate(dataloader):
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
        model_name = 'claire_models/'+str(int(start_time))+'_'+str(e)+'_feed_forward.tar'
        torch.save({
                    "epoch": e,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "batch_size": batch_size,
                    }, model_name)
    
    # Finish up
    print('elapsed time: {}'.format(time.time() - start_time))
    model_name = 'claire_models/'+str(int(start_time))+'_feed_forward.tar'
    torch.save(model, model_name)
    print('saved model: '+model_name)
  
def main():
  model = FeedForwardCNN()
  print(model)
  model.train_model(0, 1557788216)

if __name__ == "__main__":
  main()
