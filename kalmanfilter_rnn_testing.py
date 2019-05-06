import numpy as np
import math
from matplotlib import pyplot as plt
import time
import torch
import torch.utils.data
import torch.nn as nn

from synth_vis_state_est_data_generator import SynthVisStateEstDataGenerator
from feed_forward_cnn import FeedForwardCNN
from kalmanfilter_rnn import KalmanFilterRNN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Data Parameters
num_samples = 10
seq_length = 100
image_size = 128
image_dims = np.array((image_size, image_size))
minibatch_size = 1 # Do not change this.

# Load Dataset
backpropkf_test_dataset = SynthVisStateEstDataGenerator(b_load_data=True,
                                                       path_to_file="datasets/dataset_N10_T100.pkl",
                                                       num_simulations=num_samples,
                                                       num_timesteps=seq_length,
                                                       image_dims=image_dims)
print('Length of dataset: {}'.format(len(backpropkf_test_dataset)))
test_loader = torch.utils.data.DataLoader(dataset=backpropkf_test_dataset,
                                            batch_size=minibatch_size,
                                            shuffle=True)
print('Length of test_loader: {}'.format(len(test_loader)))

# Load
model_name = 'kfrnn_model_1556919710_epoch_30.ckpt' # 30 epochs, learning_rate=0.005, minibatch_size=4
model = torch.load('models/'+model_name).to(device)

# MSE Loss
criterion = nn.MSELoss()

# Test
total_step = len(test_loader)
rms_errors = []
start_time = time.time()
image_pause = 0.3*backpropkf_test_dataset.dt
image = None
image2 = None
for i, minibatch in enumerate(test_loader):
    images = torch.stack([minibatch[ii][0] for ii in range(len(minibatch))]).float().to(device)
    μ0s = torch.cat([minibatch[0][1][:, 2:5], minibatch[0][1][:, 0:2]], 1).float().to(device) # make the column order (velocities, states)
    positions = torch.stack([minibatch[ii][1][:, 0:2] for ii in range(len(minibatch))]).float().to(device)
    velocities = torch.stack([minibatch[ii][1][:, 2:4] for ii in range(len(minibatch))]).float().to(device)
    
    # Forward pass
    outputs = model(images, μ0s, output_belief_states=False)
    loss = criterion(outputs, positions)
    # Compute RMS Error
    rms_errors.append(math.sqrt(loss.item()))

    # Display overlayed testing image
    for t in range(images.shape[0]):
        img = images[t,0,:,:].data.cpu().numpy().astype(np.uint8)
        pos_image = outputs[t,0,:].data.cpu().numpy()
        dot_color = (255, 255, 255)  # [R B G] color of tracked dot
        img2 = backpropkf_test_dataset.draw_circle(xy=pos_image, rad=int(1), color=dot_color, img=img)
        if t >= 0:
            plt.title("Seq = {}/{}, Time = {:.2f} seconds".format(i+1, len(test_loader), t*backpropkf_test_dataset.dt))
        if image2 is None:
            image2 = plt.imshow(img2)
            plt.show(block=False)
        else:
            image2.set_data(img2)
            plt.pause(image_pause)
            plt.draw()

rms_error_mean = np.mean(rms_errors)
rms_error_std = np.std(rms_errors)
print("RMS Error (Mean): {:.4f}".format(rms_error_mean))
print("RMS Error  (Std): {:.4f}".format(rms_error_std))
    
