#!/usr/bin/env python3

import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.distributions as tdist
from torch.autograd import Variable

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Physics Parameters
k = 4.
b = 0.1
m = 1.
dt = 0.1
x0 = torch.tensor([0.0, 0.0, 1.0, 1.0])

# Data Parameters (for generating synthetic data. Placeholder for Adam's dataset)
seq_length = 100
num_examples = 100
minibatch_size = 16

# Hyperparameters
num_epochs = 500
learning_rate = 0.005

# Synthetic Dataset (Placeholder for Adam's dataset)
class SyntheticDataset(torch.utils.data.Dataset):
    
    
    def __init__(self, k, b, m, dt, x0, T, num_examples):
        super(SyntheticDataset, self).__init__()
        # x0: (4, ) tensor
        # state = (2D velocity, 2D position) 
        self.A = torch.tensor([[-b/m, 0., -k/m,   0.], 
                               [0., -b/m,   0., -k/m],
                               [1.,   0.,   0.,   0.],
                               [0.,   1.,   0.,   0.]])
        self.A = self.A*dt + torch.eye(4)
        self.A.requires_grad = False
        # measurement = 2D position
        self.C = torch.tensor([[0., 0., 1., 0.],
                               [0., 0., 0., 1.]])
        self.C.requires_grad = False
        # dynamics noise: IID zero mean Gaussian (only applied to velocity)
        self.Bw = torch.tensor([[1., 0.],
                               [0., 1.],
                               [0., 0.],
                               [0., 0.]])
        self.Bw.requires_grad = False
        self.Q = torch.eye(2)
        self.mvnormal_process = tdist.MultivariateNormal(torch.zeros(2), self.Q)
        self.Q.requires_grad = False
        self.L = torch.tensor([1.1, 0.15, 1.1])
        self.L.requires_grad = False
        L = torch.tensor([[torch.exp(self.L[0]), 0.0],
                          [self.L[1], torch.exp(self.L[2])]])
        self.R = L @ L.t()
        self.R.requires_grad = False
        self.mvnormal_measurement = tdist.MultivariateNormal(torch.zeros(2), self.R)
        self.T = T
        self.num_examples = num_examples
        
        # Data generation
        self.data = []
        for ii in range(self.num_examples):
            xs = [x0]
            zs = []
            ys = []
            for t in range(self.T):
                x_output = self.A @ xs[-1] + self.Bw @ self.mvnormal_process.sample()
                xs.append(x_output)
                z_output = self.C @ x_output + self.mvnormal_measurement.sample()
                zs.append(z_output)
                y_output = self.C @ x_output
                ys.append(y_output)
            xs.pop(0)
            self.data.append((torch.stack(zs, 0), torch.stack(ys, 0)))
    
    
    def __getitem__(self, index):
        return self.data[index]
    
    
    def __len__(self):
        return self.num_examples

# Kalman Filter RNN
class KalmanFilterRNN(nn.Module):
    
    
    def __init__(self, k, b, m, dt, x0, device, end_to_end_flag=False):
        # x0: (4, ) tensor
        # state = (2D velocity, 2D position)
        super(KalmanFilterRNN, self).__init__()
        self.A = torch.tensor([[-b/m, 0., -k/m,   0.], 
                               [0., -b/m,   0., -k/m],
                               [1.,   0.,   0.,   0.],
                               [0.,   1.,   0.,   0.]])
        self.A = (self.A*dt + torch.eye(4)).to(device)
        self.A.requires_grad = False
        # measurement = 2D position
        self.C = torch.tensor([[0., 0., 1., 0.],
                               [0., 0., 0., 1.]]).to(device)
        self.C.requires_grad = False
        # dynamics noise: IID zero mean Gaussian (only applied to velocity)
        self.Bw = torch.tensor([[1., 0.],
                               [0., 1.],
                               [0., 0.],
                               [0., 0.]]).to(device)
        self.Bw.requires_grad = False
        self.Q = torch.eye(2).to(device)
        self.Q.requires_grad = False
        # initial belief state (ground truth and identity matrix)
        self.μ0 = x0.to(device)
        self.μ0.requires_grad = False
        self.Σ0 = torch.eye(4).to(device)
        self.Σ0.reuires_grad = False
        self.end_to_end_flag = end_to_end_flag
        if not self.end_to_end_flag:
            self.L_hat = nn.Parameter(torch.zeros(3))
        
        
    def process_L_hat_single(self, L_hat_single):
        # L_hat_single: (3, ) tensor
        # L: (2, 2) tensor
        L = torch.zeros(2,2).to(device)
        L[0, 0] = torch.exp(L_hat_single[0])
        L[1, 0] = L_hat_single[1]
        L[1, 1] = torch.exp(L_hat_single[2])
        R = torch.matmul(L, L.t())
        return R
    
    
    def process_L_hat_batch(self, L_hat):
        # L_hat: (N, 3) tensor
        # L: (N, 2, 2) tensor
        N = L_hat.size(0)
        L_hat_tuple = L_hat.unbind(0)
        R_list = [self.process_L_hat_single(L_hat_single) for L_hat_single in L_hat_tuple]
        R_tensor_2d = torch.stack(R_list)
        R = R_tensor_2d.view(N, 2, 2)
        return R
    
    
    def kf_update(self, μ_input, Σ_input, z, L_hat=None):
        # μ_input: (N, 4) tensor
        # Σ_input: (N, 4, 4) tensor
        # L_hat: (N, 3) tensor
        # z: (N, 2) tensor
        # μ_output: (N, 4) tensor
        # Σ_output: (N, 4, 4) tensor
        if self.end_to_end_flag:
            assert (L_hat is not None), "L_hat needs to be specified!"
        else:
            assert (L_hat is None), "L_hat cannot be specified!"
        μ_predicted = self.A @ μ_input.unsqueeze(-1)
        Σ_predicted = self.A @ Σ_input @ self.A.t() + self.Bw @ self.Q @ self.Bw.t() # (4, 4) tensor
        if self.end_to_end_flag:
            R = self.process_L_hat_batch(L_hat) # (N, 2, 2) tensor
        else:
            R = self.process_L_hat_single(self.L_hat) # (2, 2) tensor
        K = Σ_predicted @ self.C.t() @ (self.C @ Σ_predicted @ self.C.t() + R).inverse() # (N, 4, 2) tensor
        μ_output = (μ_predicted + K @ (z.unsqueeze(-1) - self.C @ μ_predicted)).squeeze(-1) # (N, 4) tensor
        Σ_output = (torch.eye(4).to(device) - K @ self.C) @ Σ_predicted # (N, 4, 4) tensor
        return (μ_output, Σ_output)
    

    def forward(self, z, output_belief_states, L_hat=None):
        # L_hat: (T, N, 3) tensor
        # z: (T, N, 2) tensor
        # μs_output: (T, N, 4) tensor
        # Σs_output: (T, N, 4, 4) tensor
        # y_hats_output: (T, N, 2) tensor
        if self.end_to_end_flag:
            assert (L_hat is not None), "L_hat needs to be specified!"
        else:
            assert (L_hat is None), "L_hat cannot be specified!"
        T = z.size(0)
        N = z.size(1)
        assert (T == z.size(0) and N == z.size(1))
        μs = [self.μ0]
        Σs = [self.Σ0]
        y_hats = []
        for t in range(T):
            # μ_output: (N, 4) tensor
            # Σ_output: (N, 4, 4) tensor
            if self.end_to_end_flag:
                (μ_output, Σ_output) = self.kf_update(μs[-1], Σs[-1], z[t], L_hat[t])
            else:
                (μ_output, Σ_output) = self.kf_update(μs[-1], Σs[-1], z[t])
            # y_hat: (N, 4) tensor
            y_hat = (self.C @ μ_output.unsqueeze(-1)).squeeze(-1)
            μs.append(μ_output)
            Σs.append(Σ_output)
            y_hats.append(y_hat)
        μs.pop(0)
        Σs.pop(0)
        μs_output = torch.stack(μs, 0)
        Σs_output = torch.stack(Σs, 0)
        y_hats_output = torch.stack(y_hats, 0)
        if output_belief_states:
            return (μs_output, Σs_output, y_hats_output)
        else:
            return y_hats_output
        

# Training
model = KalmanFilterRNN(k, b, m, dt, x0, device, end_to_end_flag=False).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

synthetic_dataset = SyntheticDataset(k, b, m, dt, x0, seq_length, num_examples)
train_loader = torch.utils.data.DataLoader(dataset=synthetic_dataset,
                                           batch_size=minibatch_size,
                                           shuffle=True)

total_step = len(train_loader)
losses = []
errors = []
start_time = time.time()
for epoch in range(num_epochs):
    for i, (zs, ys) in enumerate(train_loader):
        ys = ys.transpose(0, 1).to(device) # (T, N, 2) tensor
        zs = zs.transpose(0, 1).to(device) # (T, N, 2) tensor
        
        # Forward pass
        outputs = model(zs, output_belief_states=False)
        loss = criterion(outputs, ys)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 5 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            losses.append(loss.item())
            errors.append(torch.norm(list(model.parameters())[0].data - synthetic_dataset.L.to(device)).item())
print("--- %s seconds ---" % (time.time() - start_time))
torch.save(model.state_dict(), 'kfrnn_params.ckpt')
torch.save(losses, 'kfrnn_loss_history.ckpt')
torch.save(errors, 'kfrnn_error_history.ckpt')