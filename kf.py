import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import plot_seq
import csv
import argparse

from preprocessing.process_ouija_data import processOuijaData

# Calculating the A matrix given current state
def A_calc(x, y, theta, v, omega, dt, dataset):
  # Initialize 5x5 A matrix
  A = np.zeros((5,5))
  A[0,0] = 1
  A[1,1] = 1
  A[2,2] = 1
  A[3,3] = 1
  A[4,4] = 1

  if dataset == "kitti":
    A[0,2] = -1 * v * np.sin(theta) * dt
    A[0,3] = np.cos(theta) * dt
    A[1,2] = v * np.cos(theta) * dt
    A[1,3] = np.sin(theta) * dt
    A[2,4] = dt
  elif dataset == "ouija":
    A[0,2] = -1 * v * np.sin(theta) * dt
    A[0,3] = np.cos(theta) * dt
    A[1,2] = v * np.cos(theta) * dt
    A[1,3] = np.sin(theta) * dt
    A[2,4] = dt

  return(A)

# Update state estimate with dynamics equations
def update_mu(mu,dt):
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
  
# Kalman Filter update step
def kf_step(mu, sig, z, dt, dataset):
  # Parse state 
  x = mu[0]
  y = mu[1]
  theta = mu[2]
  v = mu[3]
  omega = mu[4]

  A = A_calc(x,y,theta,v,omega,dt, dataset)
  C = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

  # Covariance matrices
  R = np.identity(5) * 1e10 # State transition uncertainty
  Q = np.identity(2) * 1e10 # Measurement uncertainty

  # Update
  mu_next_p = update_mu(mu,dt)
  sig_next_p = A @ sig @ A.transpose() + R
  K = sig_next_p @ C.transpose() @ inv(C @ sig_next_p @ C.transpose() + Q)
  mu_next = mu_next_p + K @ (z - C @ mu_next_p)
  sig_next = (np.identity(5) - K @ C) @ sig_next_p
  return mu_next, sig_next
  
def main(dataset, sequence):
  print("Kalman filter")

  x = [] # Ground truth x position
  y = [] # Ground truth y position
  vels = [] # Ground truth velocity tuples (forward, angular)
  thetas = [] # Ground truth angles
  times = [] # Timestamps
  vel_hat = [] # Inferred velocity tuples (forward, angular)

  if dataset == "kitti":
    # Open ground truth pose file
    # Save x,y,theta into lists
    file_path = "./dataset/poses/" + str(sequence).zfill(2) + ".txt"
    x_ind = 3
    y_ind = 11
    with open(file_path, "r") as fid:
      for i, line in enumerate(fid):
        row = [float(s) for s in line.split(" ")]
        x.append(row[x_ind])
        y.append(row[y_ind])
        if np.arcsin(row[0]) > 0:
          thetas.append(np.arccos(row[0]))
        else:
          thetas.append(np.arccos(row[0]) * -1)
    
    # Get timestamps of sequence
    times_file_path = "./dataset/" + str(sequence).zfill(2) + "/times.txt"
    with open(times_file_path, "r") as fid:
      for i, line in enumerate(fid):
        times.append(float(line))

    # Get oxts data
    oxts_dir = "./dataset/" + str(sequence).zfill(2) + "/oxts/data/"
    acc = []
    for i in range(len(times)):
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
    with open(path, mode="r") as csv_fid:
      reader = csv.reader(csv_fid, delimiter=",")
      for i, row in enumerate(reader):
        if i != 0:
          vel_hat.append([float(row[0]),float(row[1])])
    
  if dataset == "ouija":
    # Get ground truth data
    file_path = "../test_traj_" + str(sequence).zfill(1) + "/data.txt"
    time_ind = 0
    x_ind = 1
    y_ind = 2
    x_vel_ind = 8
    theta_vel_ind = 10
    with open(file_path, "r") as fid:
      reader = csv.reader(fid, delimiter=",")
      next(reader)
      for row in reader:
        x.append(float(row[x_ind]))
        y.append(float(row[y_ind]))
        vels.append((float(row[x_vel_ind]),float(row[theta_vel_ind])))
        times.append(float(row[time_ind])) 

    # Get forward and angular velocities from Optitrack data
    data_processor = processOuijaData(file_path)
    opti_for_vels, opti_ang_vels = data_processor.get_vels()
    thetas = data_processor.get_theta()

    # Get inferred velocities from cnn result txt files
    path = "../test_traj_" + str(sequence).zfill(1) + "/results.txt"
    with open(path, mode="r") as csv_fid:
      reader = csv.reader(csv_fid, delimiter=",")
      next(reader)
      for i, row in enumerate(reader):
          #vel_hat.append([float(row[0]), 1*float(row[1])])
          vel_hat.append([(float(row[0])-7)/5, 1*float(row[1])])

  # Number of timesteps to calculate
  MAX = len(vel_hat)

  # Initial conditions
  sig_next = np.identity(5)
  prev_time = times[0]
  #z = [0,0]
  z = vels[0]
  x_in = x[0]
  y_in = y[0]
  
  if dataset == "kitti":
    theta_in = thetas[0] + np.pi/2
  elif dataset == "ouija":
    theta_in = thetas[0]

  mu_next = np.array([x_in,y_in,theta_in,z[0],z[1]])

  mu_next_est = np.array([x_in,y_in,theta_in,z[0],z[1]])
  sig_next_est = np.identity(5)

  # Lists to save results for plotting
  mu_list = []
  x_list = []
  y_list = []

  # KF with inferred velocities
  x_list_est = []
  y_list_est = []

  sigmas = []
  indexes = []
  for i in range(1, MAX):
    print("\n Iteration {}".format(i))
    curr_time = times[i]
    dt = curr_time - prev_time
    prev_time = curr_time

    # Observed velocities
    z_true = vels[i] # z from joy data / oxts
    #z_true = (opti_for_vels[i], opti_ang_vels[i]) # z from optitrack data

    # Run filter with ground truth velocities
    mu_next, sig_next = kf_step(mu_next, sig_next, z_true, dt, dataset)
    mu_list.append(mu_next)
    x_list.append(mu_next[0])
    y_list.append(mu_next[1])
    #sigmas.append(np.trace(sig_next))
    #indexes.append(i)
    #print("Ground truth x {} y {} theta {} velocities {}".format(x[i],y[i],theta[i] + np.pi/2, z))

    # Run filter with inferred velocities
    z_est = vel_hat[i]
    mu_next_est, sig_next_est = kf_step(mu_next_est, sig_next_est, z_est, dt, dataset)
    x_list_est.append(mu_next_est[0])
    y_list_est.append(mu_next_est[1])


  # Plot
  # If dataset is ouija, switch x and y to match mocap fram
  if dataset == "kitti":
    line_true = plt.plot(x[0:MAX], y[0:MAX], label="Actual")
    line_kf_true = plt.plot(x_list, y_list, label="KF using true velocities as observations")
    line_kf_est = plt.plot(x_list_est, y_list_est, label="KF using inferred vels")
  elif dataset == "ouija":
    line_true = plt.plot(y[0:MAX], x[0:MAX], label="Actual")
    line_kf_true = plt.plot(y_list, x_list, label="KF using true velocities as observations")
    line_kf_est = plt.plot(y_list_est, x_list_est, label="KF using inferred vels")
    plt.xlim(5, -5)
    plt.axis("equal")

  plt.setp(line_kf_est, color='b', ls='-', marker='.')
  plt.setp(line_true, color='r', ls='-', marker='.')
  plt.setp(line_kf_true, color='g', ls='-', marker='.')

  plt.legend()
  plt.ylabel("y (m)")
  plt.xlabel("x (m)")
  plt.title("{} Trajectory {}".format(dataset,sequence))
  #plt.figure()
  #plt.plot(indexes,sigmas)
  plt.show()
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", help="dataset type", choices=["ouija", "kitti"])
  parser.add_argument("--traj_num", help="trajectory number")
  args = parser.parse_args()
  dataset = args.dataset
  traj_num = args.traj_num
  main(dataset, traj_num)
