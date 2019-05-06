import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from feed_forward_cnn import FeedForwardCNN
from synth_vis_state_est_data_generator import SynthVisStateEstDataGenerator

# Device specification
# device = torch.device('cpu')
device = torch.device('cuda')

# Setup
# seed(0)
num_samples = 100
seq_length = 100
num_obstacles = 10
image_size = 128
image_dims = np.array((image_size, image_size))

# Read dataset
dataset_name = 'dataset_N100_T100'
dataset_path = 'datasets/'+dataset_name+'.pkl'
backpropkf_dataset = SynthVisStateEstDataGenerator(b_load_data=True, path_to_file=dataset_path, image_dims=image_dims, b_show_figures=False)

# # Generate testing dataset
# backpropkf_dataset = SynthVisStateEstDataGenerator(b_load_data=None, path_to_file=None, num_timesteps=seq_length, num_simulations=num_samples, image_dims=image_dims, b_show_figures=False)

# Format data into one large dataset
new_dataset = []
for t in range(len(backpropkf_dataset)):
    for n in range(len(backpropkf_dataset[t])):
        temp = backpropkf_dataset[t][n]
        new_dataset.append(temp)
print('Length of dataset: {}'.format(len(new_dataset)))
minibatch_size = 4
train_loader = torch.utils.data.DataLoader(dataset=new_dataset,
                                            batch_size=minibatch_size,
                                            shuffle=True)
print('Length of train_loader: {}'.format(len(train_loader)))

# Construct feed forward CNN model
image_channels = 3
z_dim = 2
output_covariance = False
model = FeedForwardCNN(image_channels, image_dims, z_dim, output_covariance)
print(model)
model = model.to(device=device)  # move model to speicified device

# Construct loss function and optimizer
loss_function = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Training
epochs = 100
losses = []
errors = []
start_time = time.time()
loss_file = 'eric_models/'+str(int(start_time))+'_loss.txt'
loss_save = open(loss_file, "a")
loss_save.write('epoch, iteration, loss, error\n')
for e in range(epochs):
    for i, (x,y) in enumerate(train_loader):

        # # Extract training pairs from sequence of sequences (from Haruki)
        # x = torch.stack([minibatch[ii][0] for ii in range(len(minibatch))]).float().permute(0,1,4,2,3).squeeze(1).to(device)
        # y_z = torch.stack([minibatch[ii][1][:, 0:2] for ii in range(len(minibatch))]).float().to(device)

        # Format data
        x = x.float().permute(0,3,1,2).to(device)
        y_z = y[:, 0:2].float().to(device)

        # Forward pass
        y_prediction = model(x)

        # Compute loss
        loss = loss_function(y_prediction, y_z)

        # Backward pass()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if i % 10 == 0:
            print('epoch {}/{}, iteration {}/{}, loss = {}'.format(e,(epochs-1),i,int(seq_length*num_samples/minibatch_size-1),loss.item()))
            losses.append(loss.item())
            current_error = torch.norm(y_prediction-y_z)
            errors.append(current_error)
            out_text = '{}, {}, {}, {}\n'.format(e, i, loss.item(), current_error)
            loss_save.write(out_text)
    # Save current model file after epoch
    model_name = 'eric_models/'+str(int(start_time))+'_'+str(e)+'_feed_forward'
    torch.save(model, model_name)

# Finish up
print('elapsed time: {}'.format(time.time() - start_time))
model_name = 'eric_models/'+str(int(start_time))+'_feed_forward'
torch.save(model, model_name)
print('saved model: '+model_name)
loss_save.close()