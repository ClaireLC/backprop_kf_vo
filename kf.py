import numpy as np
from numpy.linalg import inv

def A(x, y, theta, v, omega, dt):
  A = np.zeros(5,5)
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
  
def kf_step(mu, sig, z, dt):
  x = mu[0]
  y = mu[1]
  theta = mu[2]
  v = mu[3]
  omega = mu[4]
  
  A = A(x,y,theta,v,omega)
  C = np.array([[0, 0, 0, 1, 0], [0, 0, 0, , 1]])

  mu_next_p = A @ mu
  sig_next_p = A @ sig @ A.transpose()
  K = sig_next_p @ C.transpose() @ inv(C @ sig_next_p @ C.transpose())
  mu_next = mu_next_p + K @ (z - C @ mu_next_p)
  sig_next = (np.identity(5) - K @ C) @ sig_next_p

  return mu_next, sig_next
  
def main():
  print("Kalman filter")
  
  # Load data
  seq_dir = "/mnt/disks/dataset/dataset_post/sequences/"
  poses_dir = "/mnt/disks/dataset/dataset/poses/"
  oxts_dir = "/mnt/disks/dataset/dataset_post/oxts/"
  dataset = KittiDataset(seq_dir, poses_dir, oxts_dir, transform=None)

  # Filter loop
  mu = np.zeros(5)
  sig = np.zeros((5,5))
  prev_time = 0.0 

  mu_list = []
  for i in range(1, len(dataset)):
    curr_time = dataset[i]["curr_time"]
    dt = curr_time - prev_time
    prev_time = curr_time

    z = dataset[i]["vel"]
    mu_next, sig_next = kf_step(mu, sig, z, dt)
    mu_list.append(mu_next)

if __name__ == "__main__":
  main()
