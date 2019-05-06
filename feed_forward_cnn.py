import numpy as np
import torch.nn as nn

class Flatten(nn.Module):

  def forward(self, x):
    N = x.shape[0]
    return x.view(N,-1)

class FeedForwardCNN(nn.Module):

  def __init__(self, image_channels=3, image_dims=np.array((128, 128)), z_dim=2, output_covariance=False):
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
    conv_kernel_size = 9
    conv_stride = 2
    pool_kernel_size = 2
    pool_stride = 2
    channel_1 = 32
    pad_1 = 4
    channel_2 = 64
    pad_2 = 8
    H_1 = int(1+(image_dims[0]-conv_kernel_size+2*pad_1)/conv_stride)
    H_1max = int(H_1/2)
    H_2 = int(1+(H_1max-conv_kernel_size+2*pad_2)/conv_stride)
    H_2max = int(H_2/2)

    # Define sequential 3D convolutional neural network model
    self.model = nn.Sequential(
        nn.Conv2d(in_channels=image_channels,
                  out_channels=channel_1,
                  kernel_size=conv_kernel_size,
                  stride=conv_stride,
                  padding=pad_1,
                  dilation=1,
                  groups=1,
                  bias=True),
        nn.ReLU(),
        nn.LayerNorm(H_1, eps=1e-05, elementwise_affine=True),
        nn.MaxPool2d(kernel_size=pool_kernel_size,
                      stride=pool_stride,
                      padding=0,
                      dilation=1,
                      return_indices=False,
                      ceil_mode=False),
        nn.Conv2d(in_channels=channel_1,
                  out_channels=channel_2,
                  kernel_size=conv_kernel_size,
                  stride=conv_stride,
                  padding=pad_2,
                  dilation=1,
                  groups=1,
                  bias=True),
        nn.ReLU(),
        nn.LayerNorm(H_2, eps=1e-05, elementwise_affine=True),
        nn.MaxPool2d(kernel_size=pool_kernel_size,
                      stride=pool_stride,
                      padding=0,
                      dilation=1,
                      return_indices=False,
                      ceil_mode=False),
        Flatten(),
        nn.Linear(channel_2*H_2max*H_2max, 32, bias=True),
        nn.Linear(32, 64, bias=True),
        nn.Linear(64, output_dim, bias=True)
      )

  def forward(self,x):
    y_pred = self.model(x)
    return y_pred