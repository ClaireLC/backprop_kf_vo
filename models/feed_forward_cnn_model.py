import numpy as np
import torch
import torch.nn as nn

class Flatten(nn.Module):

  def forward(self, x):
    return x.view(x.size(0), -1)

class FeedForwardCNN(nn.Module):

  def __init__(self, image_channels=6, image_dims=np.array([50, 150]), z_dim=2, output_covariance=False, batch_size = 1):
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
    #conv2_stride = (2,1)
    conv2_stride = (1,2)
    #conv2_pad = (2,3)
    conv2_pad = (3,2)
    conv2_out = 16

    # conv3
    conv3_in = conv2_out
    conv3_kernel_size = (5,5)
    conv3_stride = (1,2)
    conv3_pad = (2,6)
    #conv3_pad = (6,2)
    conv3_out = 16

    # conv3
    conv4_in = conv3_out
    conv4_kernel_size = (5,5)
    conv4_stride = (2,2)
    conv4_pad = (3,3)
    conv4_out = 16

    W_1 = int(1+(image_dims[1]-(conv1_kernel_size[1]-1)+2*conv1_pad[1]-1)/conv1_stride[1])
    H_1 = int(1+(image_dims[0]-(conv1_kernel_size[0]-1)+2*conv1_pad[0]-1)/conv1_stride[0])
    #print(H_1, W_1)

    W_2 = int(1+(W_1-(conv2_kernel_size[1]-1)+2*conv2_pad[1]-1)/conv2_stride[1])
    H_2 = int(1+(H_1-(conv2_kernel_size[0]-1)+2*conv2_pad[0]-1)/conv2_stride[0])
    #print(H_2, W_2)

    W_3 = int(1+(W_2-(conv3_kernel_size[1]-1)+2*conv3_pad[1]-1)/conv3_stride[1])
    H_3 = int(1+(H_2-(conv3_kernel_size[0]-1)+2*conv3_pad[0]-1)/conv3_stride[0])
    #print(H_3, W_3)

    W_4 = int(1+(W_3-(conv4_kernel_size[1]-1)+2*conv4_pad[1]-1)/conv4_stride[1])
    H_4 = int(1+(H_3-(conv4_kernel_size[0]-1)+2*conv4_pad[0]-1)/conv4_stride[0])
    #print(H_4, W_4)

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
        nn.BatchNorm2d(16, eps=1e-05, affine=True),
        #nn.LayerNorm([batch_size, 16, H_1, W_1], eps=1e-05, elementwise_affine=True),

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
        nn.BatchNorm2d(16, eps=1e-05, affine=True),
        #nn.LayerNorm([batch_size, 16, H_2, W_2], eps=1e-05, elementwise_affine=True),

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
        nn.BatchNorm2d(16, eps=1e-05, affine=True),
        #nn.LayerNorm([batch_size, 16, H_3, W_3], eps=1e-05, elementwise_affine=True),

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
        nn.BatchNorm2d(16, eps=1e-05, affine=True),
        #nn.LayerNorm([batch_size, 16, H_4, W_4], eps=1e-05, elementwise_affine=True),
        nn.Dropout(p=0.9),
        Flatten(),
        # Fully connected layers
        nn.Linear(H_4 * W_4 * 16, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, output_dim, bias=True)
      )

  def forward(self,x):
    # x: (N, 3, self.image_size, self.image_size) tensor
    y_pred = self.model(x)
    return y_pred