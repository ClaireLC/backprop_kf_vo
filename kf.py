import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import plot_seq
import csv

# For testing dynamics updates
def test(x,y,theta,z,dt,for_acc):
  # Check state updates
  #theta += np.pi/2 # transform theta to world frame
  v = z[0]
  omega = z[1]

  mu_next = np.zeros(5)
  #mu_next[0] = v * dt * np.cos(theta) + x
  #mu_next[1] = v * dt * np.sin(theta) + y
  #mu_next[2] = theta + omega * dt
  #mu_next[3] = v
  ##mu_next[3] = v + dt * for_acc
  #mu_next[4] = omega
  
  # Check A matrix
  A = A_calc(x,y,theta,z[0],z[1],dt)
  #B = np.asarray([0,0,0,dt,0]) temp = np.zeros((5,1))
  #temp[3][0] = for_acc
  mu_next = A @ np.asarray([x,y,theta,z[0],z[1]]) #+ B @ temp
  return mu_next
  
# Calculating the A matrix given current state
def A_calc(x, y, theta, v, omega, dt):
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
  
# Kalman Filter update step
def kf_step(mu, sig, z, dt):
  # Parse state 
  x = mu[0]
  y = mu[1]
  theta = mu[2]
  v = mu[3]
  omega = mu[4]

  print("dt {}".format(dt))
  print("obs {}".format(z))
  print("state {}".format(mu))

  A = A_calc(x,y,theta,v,omega,dt)
  C = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

  # Covariance matrices
  R = np.identity(5) * 1e10 # State transition uncertainty
  Q = np.identity(2) * 1e10 # Measurement uncertainty

  # Update
  mu_next_p = A @ mu
  sig_next_p = A @ sig @ A.transpose() + R
  K = sig_next_p @ C.transpose() @ inv(C @ sig_next_p @ C.transpose() + Q)
  mu_next = mu_next_p + K @ (z - C @ mu_next_p)
  sig_next = (np.identity(5) - K @ C) @ sig_next_p
  print("sig {}".format(sig_next))
  print("new state {}".format(mu_next))
  return mu_next, sig_next
  
def main():
  print("Kalman filter")

  # Number of timesteps to calculate
  MAX = 50

  # Sequence number
  sequence = 1

  # Open ground truth pose file
  # Save x,y,theta into lists
  file_path = "./dataset/poses/" + str(sequence).zfill(2) + ".txt"
  x = []
  y = []
  theta = []
  x_ind = 3
  y_ind = 11
  with open(file_path, "r") as fid:
    for i, line in enumerate(fid):
      row = [float(s) for s in line.split(" ")]
      x.append(row[x_ind])
      y.append(row[y_ind])
      if np.arcsin(row[0]) > 0:
        theta.append(np.arccos(row[0]))
      else:
        theta.append(np.arccos(row[0]) * -1)
  
  # Get timestamps of sequence
  times_file_path = "./dataset/" + str(sequence).zfill(2) + "/times.txt"
  oxts_dir = "./dataset/" + str(sequence).zfill(2) + "/oxts/data/"
  # timestamp array
  times = []
  with open(times_file_path, "r") as fid:
    for i, line in enumerate(fid):
      times.append(float(line))

  # Get oxts data
  vels = []
  acc = []
  for i in range(MAX):
    oxts_file_path = oxts_dir + str(i).zfill(10) + ".txt"
    fid = open(oxts_file_path) 
    line = fid.readlines()
    oxts_data = [float(s) for s in line[0].split(" ")]
    for_vel = oxts_data[8]
    ang_vel = oxts_data[19]
    for_acc = oxts_data[14]
    #ang_acc = oxts_data[]
    vels.append(np.asarray([for_vel,ang_vel]))
    acc.append(for_acc)

  # Get inferred velocities from cnn result txt files
  path = "./cnn_results/" + str(sequence).zfill(2) + "-results.txt" 
  vel_hat = []
  with open(path, mode="r") as csv_fid:
    reader = csv.reader(csv_fid, delimiter=",")
    for i, row in enumerate(reader):
      if i != 0:
        vel_hat.append([float(row[1]),float(row[2])])

  # Initial conditions
  sig_next = np.identity(5)
  prev_time = 0.0 
  z = vels[0]
  x_in = x[0]
  y_in = y[0]
  theta_in = theta[0] + np.pi/2
  mu_next = np.array([x_in,y_in,theta_in,z[0],z[1]])

  mu_next_est = np.array([x_in,y_in,theta_in,z[0],z[1]])
  sig_next_est = np.identity(5)

  # Lists to save results for plotting
  mu_list = []
  x_list = []
  y_list = []

  # Test coords
  x_list_est = []
  y_list_est = []

  sigmas = []
  indexes = []
  for i in range(1, MAX):
    print("\n Iteration {}".format(i))
    curr_time = times[i]
    dt = curr_time - prev_time
    prev_time = curr_time

    # Run filter with ground truth velocities
    z_true = vels[i]
    mu_next, sig_next = kf_step(mu_next, sig_next, z_true, dt)
    mu_list.append(mu_next)
    x_list.append(mu_next[0])
    y_list.append(mu_next[1])
    sigmas.append(np.trace(sig_next))
    indexes.append(i)
    print("Ground truth x {} y {} theta {} velocities {}".format(x[i],y[i],theta[i] + np.pi/2, z))

    # Run filter with inferred velocities
    z_est = vel_hat[i]
    mu_next_est, sig_next_est = kf_step(mu_next_est, sig_next_est, z_est, dt)
    x_list_est.append(mu_next_est[0])
    y_list_est.append(mu_next_est[1])

  # Plot
  line_true = plt.plot(x[0:MAX], y[0:MAX], label="true")
  line_kf_true = plt.plot(x_list, y_list, label="kf with true vels")
  line_kf_est = plt.plot(x_list_est, y_list_est, label="kf with estimated vels")

  plt.setp(line_true, color='r', ls='-', marker='.')
  plt.setp(line_kf_true, color='g', ls='-', marker='.')
  plt.setp(line_kf_est, color='b', ls='-', marker='.')

  plt.legend()
  #plt.figure()
  #plt.plot(indexes,sigmas)
  plt.show()
  
if __name__ == "__main__":
  main()
