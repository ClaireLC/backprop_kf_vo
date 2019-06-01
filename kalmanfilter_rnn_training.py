import numpy as np
import time
import torch
import torch.utils.data
import torch.nn as nn
from matplotlib import pyplot as plt

from synth_vis_state_est_data_generator import SynthVisStateEstDataGenerator
from feed_forward_cnn import FeedForwardCNN
from kalmanfilter_rnn import KalmanFilterRNN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Data Parameters
num_samples = 100
seq_length = 100
image_size = 128
image_dims = np.array((image_size, image_size))
minibatch_size = 4

# Hyperparameters
num_epochs = 50
learning_rate = 5e-3

# Pretrained CNN
#trained_cnn_path = None
trained_cnn_path = "models/1556733699_feed_forward" # image_size=128, output_covariance=False

# Load Dataset
backpropkf_dataset = SynthVisStateEstDataGenerator(b_load_data=True,
                                                   path_to_file="datasets/dataset_N100_T100.pkl",
                                                   num_simulations=num_samples,
                                                   num_timesteps=seq_length,
                                                   image_dims=image_dims)
print('Length of dataset: {}'.format(len(backpropkf_dataset)))
train_loader = torch.utils.data.DataLoader(dataset=backpropkf_dataset,
                                            batch_size=minibatch_size,
                                            shuffle=True)
print('Length of train_loader: {}'.format(len(train_loader)))

## Backprop KF model
#model = KalmanFilterRNN(backpropkf_dataset.k,
#                        backpropkf_dataset.b,
#                        backpropkf_dataset.m,
#                        backpropkf_dataset.dt,
#                        image_size,
#                        device,
#                        trained_cnn_path=trained_cnn_path,
#                        end_to_end_flag=True).to(device)

# Piecewise KF model
model = KalmanFilterRNN(backpropkf_dataset.k,
                        backpropkf_dataset.b,
                        backpropkf_dataset.m,
                        backpropkf_dataset.dt,
                        image_size,
                        device,
                        trained_cnn_path=trained_cnn_path,
                        end_to_end_flag=False).to(device)

# MSE Loss, Adam Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
total_step = len(train_loader)
losses = []
start_time = time.time()
for epoch in range(num_epochs):
    # minibatch dim (N, img, (x0, y0, vx0, vy0))
    for i, minibatch in enumerate(train_loader):
        images = torch.stack([minibatch[ii][0] for ii in range(len(minibatch))]).float().to(device)
        μ0s = torch.cat([minibatch[0][1][:, 2:5], minibatch[0][1][:, 0:2]], 1).float().to(device) # make the column order (velocities, states)
        positions = torch.stack([minibatch[ii][1][:, 0:2] for ii in range(len(minibatch))]).float().to(device)
        velocities = torch.stack([minibatch[ii][1][:, 2:4] for ii in range(len(minibatch))]).float().to(device)
        
        # Forward pass
        outputs = model(images, μ0s, output_belief_states=False)
        loss = criterion(outputs, positions)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % len(train_loader) == 0:
        #if (i+1) % 5 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        model_name = 'models/kfrnn_model_' + str(int(start_time)) + '_epoch_' + str(epoch+1) + '.ckpt'
        torch.save(model, model_name)
print("--- %s seconds ---" % (time.time() - start_time))
#model_name = 'models/kfrnn_model_' + str(int(start_time)) + '_last_epoch.ckpt'
#torch.save(model, model_name)
loss_history_name = 'models/kfrnn_loss_history_' + str(int(start_time)) + '.ckpt'
torch.save(losses, loss_history_name)
fig, axs = plt.subplots()
axs.plot(range(1, len(losses) + 1), losses)
axs.set_xlabel('epoch')
axs.set_ylabel('MSE training loss')
axs.grid(True)
fig.tight_layout()
# Display and save image
fig_name = 'models/kfrnn_loss_history_' + str(int(start_time)) + '.png'
plt.savefig(fig_name, format="png")
plt.show()
