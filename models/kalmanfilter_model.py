import numpy as np
import torch
import torch.nn as nn

from .feed_forward_cnn_model import FeedForwardCNN

# Kalman Filter
class KalmanFilter(nn.Module):
  
  def __init__(self, device):
    # For training backwardKF end to end
    # x0: (4, ) tensor
    # state = (2D velocity, 2D position)
    super(KalmanFilter, self).__init__()
    self.device = device
  
    # Jacobian of robot dynamics
    #self.A = torch.tensor([[-b/m, 0.,   0., -k/m,   0.], 
    #           [0., -b/m,   0.,   0., -k/m],
    #           [1.,   0., -b/m,   0.,   0.],
    #           [0.,   1.,   0.,   0.,   0.],
    #           [0.,   0.,   1.,   0.,   0.]])
    #self.A = (self.A*dt + torch.eye(5)).to(device)
    #self.A.requires_grad = False

    # measurement = 2D position
    self.C = torch.tensor([[0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 1.]]).to(device)
    self.C.requires_grad = False

    # C for loss
    self.Cy = torch.tensor([[1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.]]).to(device)
    self.Cy.requires_grad = False

    # dynamics noise: IID zero mean Gaussian (only applied to velocity)
    #self.Bw = torch.tensor([[1., 0.],
    #                       [0., 1.],
    #                       [0., 0.],
    #                       [0., 0.],
    #                       [0., 0.]]).to(device)
    #self.Bw.requires_grad = False

    # State transition uncertainty
    self.Q = torch.eye(5).to(device)
    self.Q.requires_grad = False

    # initial belief state (ground truth and identity matrix)
    #self.μ0 = x0.to(device)
    #self.μ0.requires_grad = False

    # Initial state covariance
    self.Σ0 = torch.eye(5).to(device)
    self.Σ0.reuires_grad = False
    
  def A_calc(self, μ, dt):
    """
    Calculate A matrix
    μ: (N,5) state
    dt: (N, 1)
    A: (N, 5, 5)
    """
    # Extract batch size from μ
    N = μ.shape[0]

    # Parse μ into its columns
    x     = μ[:,0].cpu().detach().numpy() # (N,)
    y     = μ[:,1].cpu().detach().numpy() # (N,)
    theta = μ[:,2].cpu().detach().numpy() # (N,)
    v     = μ[:,3].cpu().detach().numpy() #(N,)
    dt    = dt.cpu().detach().numpy()

    A = np.zeros((N,5,5))
    A[:,0,2] = -1 * v * np.sin(theta) * dt
    A[:,0,3] = np.cos(theta) * dt
    A[:,1,2] = v * np.cos(theta) * dt
    A[:,1,3] = np.sin(theta) * dt
    A[:,2,4] = dt

    A_torch = torch.from_numpy(A).to(self.device)
    A_torch.requires_grad = False

    return A_torch

  def update_mu(self, μ, dt):
    # Extract batch size from μ
    N = μ.shape[0]

    # Parse μ into its columns
    x     = μ[:,0].cpu().detach().numpy() # (N,)
    y     = μ[:,1].cpu().detach().numpy() # (N,)
    theta = μ[:,2].cpu().detach().numpy() # (N,)
    v     = μ[:,3].cpu().detach().numpy() #(N,)
    omega = μ[:,4].cpu().detach().numpy() # (N,)
    dt    = dt.cpu().detach().numpy()

    mu_next = np.zeros((N,5))
    mu_next[:,0] = v * dt * np.cos(theta) + x
    mu_next[:,1] = v * dt * np.sin(theta) + y
    mu_next[:,2] = theta + omega * dt
    mu_next[:,3] = v
    mu_next[:,4] = omega
    
    mu_next_torch = torch.from_numpy(mu_next).to(self.device)
    mu_next_torch.requires_grad = False

    return mu_next_torch

  def process_L_hat_single(self, L_hat_single):
    # L_hat_single: (3, ) tensor
    # R: (2, 2) tensor
    L = torch.zeros(2,2).to(self.device)
    L[0, 0] = torch.exp(L_hat_single[0])
    L[1, 0] = L_hat_single[1]
    L[1, 1] = torch.exp(L_hat_single[2])
    R = torch.matmul(L, L.t())
    return R
  
  def process_L_hat_batch(self, L_hat):
    # L_hat: (N, 3) tensor
    # R: (N, 2, 2) tensor
    N = L_hat.size(0)
    L_hat_tuple = L_hat.unbind(0)
    R_list = [self.process_L_hat_single(L_hat_single) for L_hat_single in L_hat_tuple]
    R = torch.stack(R_list)
    return R
  
  def kf_update(self, μ_input, Σ_input, z, dt, L_hat=None):
    # μ_input: (N, 5) tensor
    # Σ_input: (N, 5, 5) tensor
    # L_hat: (N, 3) tensor
    # z: (N, 2) tensor
    # dt: (N, 3)
    # μ_output: (N, 5) tensor
    # Σ_output: (N, 5, 5) tensor

    # Extract batch size from μ_input
    N = μ_input.shape[0]
    
    # Add batch dim to self.C
    # Shape goes from (2,5) to (N,2,5) tensor
    C = self.C.unsqueeze(0).repeat(N,1,1)

    # Training end to end
    assert (L_hat is not None), "L_hat needs to be specified!"
   
    # A is (N, 5, 5)
    A = self.A_calc(μ_input, dt)

    # mu is (N, 5, 1)
    μ_predicted = self.update_mu(μ_input, dt).unsqueeze(-1).float()

    # (N, 5, 5) + (5, 5) = (N, 5, 5) tensor
    Σ_predicted = A.float() @ Σ_input.float() @ A.permute(0, 2, 1).float() + self.Q
    
    R = self.process_L_hat_batch(L_hat) # (N, 2, 2) tensor
    
    # K is (N, 5, 2)
    K = Σ_predicted @ C.permute(0,2,1) @ (C @ Σ_predicted @ C.permute(0,2,1) + R).inverse() # (N, 5, 2) tensor

    μ_output = (μ_predicted + K @ (z.unsqueeze(-1) - C @ μ_predicted)).squeeze(-1) # (N, 5) tensor
    Σ_output = (torch.eye(5).to(self.device) - K @ C) @ Σ_predicted # (N, 5, 5) tensor

    return (μ_output, Σ_output)
  

  def forward(self, z_and_L_hat_list, μ0, times):
    # L_hat: (T, N, 3) tensor
    # z_and_L_hat_list: [(N, 5) tensor, ...] of length T
    # μ0: (N, 5) tensor
    # times: (T, N,) tensor

    # μs_output: (T, N, 5) tensor
    # Σs_output: (T, N, 4, 4) tensor
    # y_hats_output: (T, N, 2) tensor
    T = len(z_and_L_hat_list)

    μs = [μ0]
    Σs = [self.Σ0]
    y_hats = []

    prev_times= times[0,:]
    for t in range(T):
      # μ_output: (N, 5) tensor
      # Σ_output: (N, 5, 5) tensor
      
      # z: (N, 2) tensor
      # L_hat: (N, 3) tensor
      z_and_L_hat = z_and_L_hat_list[t]
      z = z_and_L_hat[:, 0:2]
      L_hat = z_and_L_hat[:, 2:5]
      dt = times[t] - prev_times
      prev_times = times[t]
      (μ_output, Σ_output) = self.kf_update(μs[-1], Σs[-1], z, dt, L_hat)
      
      # y_hat: (N, 3) tensor
      y_hat = (self.Cy @ μ_output.unsqueeze(-1)).squeeze(-1)
      μs.append(μ_output)
      Σs.append(Σ_output)
      y_hats.append(y_hat)

    μs.pop(0)
    Σs.pop(0)
    μs_output = torch.stack(μs, 0)
    Σs_output = torch.stack(Σs, 0)
    y_hats_output = torch.stack(y_hats, 0)
    
    return y_hats_output
