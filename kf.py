"""
Extended kalman filter class for KITTI/ouija dynamics model
Performs kalman filter update
"""
import numpy as np
from numpy.linalg import inv
from statistics import mean

class KF():

  def __init__(self):
    self.C = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
  
    # Covariance matrices
    self.R = np.identity(5) * 1e10 # Process noise
    self.Q = np.identity(2) * 1e10 # Observation noise

  def A_calc(self, x, y, theta, v, omega, dt):
    """
    Calculating the A matrix given current state
    """
    # Initialize 5x5 A matrix
    A = np.zeros((5,5))
    A[0,0] = 1
    A[1,1] = 1
    A[2,2] = 1
    A[3,3] = 1
    A[4,4] = 1
  
    A[0,2] = -1 * v * np.sin(theta) * dt
    A[0,3] = np.cos(theta) * dt
    A[1,2] = v * np.cos(theta) * dt
    A[1,3] = np.sin(theta) * dt
    A[2,4] = dt
  
    return(A)

  def update_mu(self, mu, dt):
    """
    Update state estimate with dynamics equations
    """
    x = mu[0]
    y = mu[1]
    theta = mu[2]
    v = mu[3]
    omega = mu[4]
  
    mu_next = np.zeros(5)
    mu_next[0] = v * dt * np.cos(theta) + x
    mu_next[1] = v * dt * np.sin(theta) + y
    mu_next[2] = theta + omega * dt
    mu_next[3] = v
    mu_next[4] = omega
    
    return mu_next
    
  def step(self, mu, sig, z, dt):
    """
    One update step of EKF
    mu: 5-dimension state (x,y,theta,v,omega)
    sig: 5x5 covariance matrix
    z: 2-dimension observation (for_vel, ang_vel)
    dt: time different between current and previous steps
    """
    # Parse state 
    x = mu[0]
    y = mu[1]
    theta = mu[2]
    v = mu[3]
    omega = mu[4]
  
    A = self.A_calc(x,y,theta,v,omega,dt)
  
    # Update
    mu_next_p = self.update_mu(mu,dt)
    sig_next_p = A @ sig @ A.transpose() + self.R
    K = sig_next_p @ self.C.T @ inv(self.C @ sig_next_p @ self.C.T + self.Q)
    mu_next = mu_next_p + K @ (z - self.C @ mu_next_p)
    sig_next = (np.identity(5) - K @ self.C) @ sig_next_p
    return mu_next, sig_next

