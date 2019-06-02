import numpy as np
import torch
import torch.nn as nn

from models.feed_forward_cnn_model import FeedForwardCNN

# Kalman Filter RNN
class KalmanFilterRNN(nn.Module):
    
    
    def __init__(self, k, b, m, dt, image_size, device, trained_cnn_path=None, end_to_end_flag=False):
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
        self.end_to_end_flag = end_to_end_flag
        if not self.end_to_end_flag:
            self.L_hat = nn.Parameter(torch.zeros(3))
        self.image_size = image_size
        if trained_cnn_path is None:
            self.feedforward_net = FeedForwardCNN(image_channels=3, 
                                                  image_dims=np.array((self.image_size, self.image_size)),
                                                  z_dim=2,
                                                  output_covariance=end_to_end_flag);
        else:
            # This will not work unless the saved model uses the same image_size and end_to_end_flag
            self.feedforward_net = torch.load(trained_cnn_path)
        
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
        Σ_output = (torch.eye(4).to(self.device) - K @ self.C) @ Σ_predicted # (N, 4, 4) tensor
        return (μ_output, Σ_output)
    

    def forward(self, o, μ0, output_belief_states):
        # L_hat: (T, N, 3) tensor
        # o: (T, N, self.image_size, self.image_size, 3) tensor
        # μ0: (N, 4) tensor
        # μs_output: (T, N, 4) tensor
        # Σs_output: (T, N, 4, 4) tensor
        # y_hats_output: (T, N, 2) tensor
        T = o.size(0)
        N = o.size(1)
        μs = [μ0]
        Σs = [self.Σ0]
        y_hats = []
        for t in range(T):
            # process o through the feedforward network.
            o_ = o[t,:].transpose(1,3)

            # μ_output: (N, 4) tensor
            # Σ_output: (N, 4, 4) tensor
            if self.end_to_end_flag:
                # z: (N, 2) tensor
                # L_hat: (N, 3) tensor
                z_and_L_hat = self.feedforward_net(o_)
                z = z_and_L_hat[:, 0:2]
                L_hat = z_and_L_hat[:, 2:5]
                (μ_output, Σ_output) = self.kf_update(μs[-1], Σs[-1], z, L_hat)
            else:
                # z: (N, 2) tensor
                z = self.feedforward_net(o_)
                (μ_output, Σ_output) = self.kf_update(μs[-1], Σs[-1], z)
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
