import numpy as np
import time
import random
import torch
import torch.nn as nn
import statistics
from feed_forward_cnn import FeedForwardCNN
from synth_vis_state_est_data_generator import SynthVisStateEstDataGenerator
from matplotlib import pyplot as plt

# Device specification
device = torch.device('cpu')
# device = torch.device('cuda')

# Setup
# seed(0)
num_samples = 10
seq_length = 100
image_size = 128
image_dims = np.array((image_size, image_size))

# Read dataset
dataset_name = 'dataset_N10_T100'
dataset_path = 'datasets/'+dataset_name+'.pkl'
backpropkf_dataset = SynthVisStateEstDataGenerator(b_load_data=True, path_to_file=dataset_path, image_dims=image_dims, b_show_figures=False)

# # Generate testing dataset
# backpropkf_dataset = SynthVisStateEstDataGenerator(b_load_data=None, path_to_file=None, num_timesteps=seq_length, num_simulations=num_samples, image_dims=image_dims, b_show_figures=False)

# Format data into one large dataset
new_dataset = []
ind = 0
for n in range(len(backpropkf_dataset)):
    for t in range(len(backpropkf_dataset[n])):
        temp = backpropkf_dataset[n][t]
        new_dataset.append(temp)
minibatch_size = 1
test_loader = torch.utils.data.DataLoader(dataset=new_dataset,
                                            batch_size=minibatch_size,
                                            shuffle=False)

# Load trained model
image_channels = 3
z_dim = 2
output_covariance = False
model_number = '1556733699'
epoch_number = -1
if epoch_number>=0:
    model_name = 'eric_models/'+model_number+'_'+str(epoch_number)+'_feed_forward'
else:
    model_name = 'eric_models/'+model_number+'_feed_forward'
print('loading model: ',model_name)
model = torch.load(model_name)
model = model.to(device=device)  # move model to speicified device

# Construct loss function
loss_function = torch.nn.MSELoss(reduction='sum')

# Testing on sequences
losses = []
errors = []
start_time = time.time()
image_pause = 0.01
image = None
image2 = None
for i, (x, y) in enumerate(test_loader):

    # Format data
    x_ = x.float().permute(0,3,1,2).to(device)
    y_z = y[:, 0:2].float().to(device)

    # print(y)

    # Forward pass
    z = model(x_)
    loss = loss_function(z, y_z)

    # Display overlayed testing image
    img = x[0,:,:,:].data.numpy()
    # z_ = z.cpu()
    z_image = z[0].data.numpy()
    # z_image = backpropkf_dataset.xy_to_rc(z_image)
    # print('z_image is: {}'.format(z_image))
    dot_color = (255, 255, 255)  # [R B G] color of tracked dot
    img2 = backpropkf_dataset.draw_circle(xy=z_image, rad=int(1), color=dot_color, img=img)
    if t >= 0:
        plt.title("Time = {:.2f} seconds".format(i))
    if image2 is None:
        image2 = plt.imshow(img2)
        plt.show(block=False)
    else:
        image2.set_data(img2)
        plt.pause(image_pause)
        plt.draw()

    # # Display original image
    # if t >= 0:
    #     plt.title("Time = {:.2f} seconds".format(t))
    # if image is None:
    #     image = plt.imshow(img)
    #     plt.show(block=False)
    # else:
    #     image.set_data(img)
    # plt.pause(image_pause)
    # plt.draw()

    # Print loss
    if t % 1 == 0:
        losses.append(loss.item())
        error = torch.norm(z-y_z)
        errors.append(error)
        print('iteration {}/{}, loss = {}'.format(i,int(seq_length*num_samples/minibatch_size-1),loss.item()))

# Finish up
print('elapsed time: {}'.format(time.time() - start_time))
print('testing mean RMS error: {}'.format(np.mean(np.sqrt(losses))))
print('testing std  RMS error: {}'.format(np.std(np.sqrt(losses))))