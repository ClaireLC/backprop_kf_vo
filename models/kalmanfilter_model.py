import numpy as np
import torch
import torch.nn as nn

from .feed_forward_cnn_model import FeedForwardCNN

# Kalman Filter
class KalmanFilter(nn.Module):
    
    def __init__(self, k, b, m, dt, device):
        # For training backwardKF end to end
        # x0: (4, ) tensor
        # state = (2D velocity, 2D position)
        super(KalmanFilterRNN, self).__init__()
        self.device = device
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
        #self.μ0 = x0.to(device)
        #self.μ0.requires_grad = False
        self.Σ0 = torch.eye(4).to(device)
        self.Σ0.reuires_grad = False
        
        
    def process_L_hat_single(self, L_hat_single):
        # L_hat_single: (3, ) tensor
        # L: (2, 2) tensor
        L = torch.zeros(2,2).to(self.device)
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
        
        # Training end to end
        assert (L_hat is not None), "L_hat needs to be specified!"
      
        μ_predicted = self.A @ μ_input.unsqueeze(-1)
        Σ_predicted = self.A @ Σ_input @ self.A.t() + self.Bw @ self.Q @ self.Bw.t() # (4, 4) tensor
        
        R = self.process_L_hat_batch(L_hat) # (N, 2, 2) tensor
        
        K = Σ_predicted @ self.C.t() @ (self.C @ Σ_predicted @ self.C.t() + R).inverse() # (N, 4, 2) tensor
        μ_output = (μ_predicted + K @ (z.unsqueeze(-1) - self.C @ μ_predicted)).squeeze(-1) # (N, 4) tensor
        Σ_output = (torch.eye(4).to(self.device) - K @ self.C) @ Σ_predicted # (N, 4, 4) tensor

        return (μ_output, Σ_output)
    

    def forward(self, z_and_L_hat_list, μ0):
        # L_hat: (T, N, 3) tensor
        # z_and_L_hat_list: [(N, 5) tensor, ...] of length T
        # μ0: (N, 6) tensor

        # μs_output: (T, N, 4) tensor
        # Σs_output: (T, N, 4, 4) tensor
        # y_hats_output: (T, N, 2) tensor
        T = len(z_and_L_hat_list)
        
        μs = [μ0]
        Σs = [self.Σ0]
        y_hats = []

        for t in range(T):
            # μ_output: (N, 4) tensor
            # Σ_output: (N, 4, 4) tensor
            
            # z: (N, 2) tensor
            # L_hat: (N, 3) tensor
            z_and_L_hat = z_and_L_hat_list[t]
            z = z_and_L_hat[:, 0:2]
            L_hat = z_and_L_hat[:, 2:5]
            (μ_output, Σ_output) = self.kf_update(μs[-1], Σs[-1], z, L_hat)
            
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
        
        return y_hats_output
