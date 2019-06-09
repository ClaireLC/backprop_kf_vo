import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import plot_seq
import csv
import argparse
from statistics import mean

from preprocessing.process_ouija_data import processOuijaData

class KF():

  def __init__(self, dataset, sequence):
    # Load data
    self.x = [] # Ground truth x position
    self.y = [] # Ground truth y position
    self.vels = [] # Ground truth velocity tuples (forward, angular)
    self.thetas = [] # Ground truth angles
    self.times = [] # Timestamps
    self.vel_hat = [] # Inferred velocity tuples (forward, angular)
    self.opti_vels = [] # Optitrack velocities (forward, angular)

    if dataset == "kitti":
      # Open ground truth pose file
      # Save x,y,theta into lists
      file_path = "./dataset/poses/" + str(sequence).zfill(2) + ".txt"
      x_ind = 3
      y_ind = 11
      with open(file_path, "r") as fid:
        for i, line in enumerate(fid):
          row = [float(s) for s in line.split(" ")]
          self.x.append(row[x_ind])
          self.y.append(row[y_ind])
          if np.arcsin(row[0]) > 0:
            self.thetas.append(np.arccos(row[0]) + np.pi/2)
          else:
            self.thetas.append(np.arccos(row[0]) * -1 + np.pi/2)
      
      # Get timestamps of sequence
      times_file_path = "./dataset/" + str(sequence).zfill(2) + "/times.txt"
      with open(times_file_path, "r") as fid:
        for i, line in enumerate(fid):
          self.times.append(float(line))

      # Get oxts data
      oxts_dir = "./dataset/" + str(sequence).zfill(2) + "/data/"
      for i in range(len(self.times)):
        oxts_file_path = oxts_dir + str(i).zfill(10) + ".txt"
        fid = open(oxts_file_path) 
        line = fid.readlines()
        oxts_data = [float(s) for s in line[0].split(" ")]
        for_vel = oxts_data[8]
        ang_vel = oxts_data[19]
        self.vels.append(np.asarray([for_vel,ang_vel]))

      # Get inferred velocities from cnn result txt files
      path = "./cnn_results/" + model_name + "/kitti_" + str(sequence) + ".txt"
      with open(path, mode="r") as csv_fid:
        reader = csv.reader(csv_fid, delimiter=",")
        for i, row in enumerate(reader):
          if i != 0:
            self.vel_hat.append([float(row[0]),float(row[1])])
      
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
          self.x.append(float(row[x_ind]))
          self.y.append(float(row[y_ind]))
          self.vels.append((float(row[x_vel_ind]),float(row[theta_vel_ind])))
          self.times.append(float(row[time_ind])) 

      # Get forward and angular velocities from Optitrack data, and thetas
      data_processor = processOuijaData(file_path)
      self.opti_vels = data_processor.get_vels()
      self.thetas = data_processor.get_theta()

      # Get inferred velocities from cnn result txt files
      path = "../test_traj_" + str(sequence).zfill(1) + "/results.txt"
      with open(path, mode="r") as csv_fid:
        reader = csv.reader(csv_fid, delimiter=",")
        next(reader)
        for i, row in enumerate(reader):
            #vel_hat.append([float(row[0]), 1*float(row[1])])
            self.vel_hat.append([(float(row[0])-7)/5, 1*float(row[1])])

  # Calculating the A matrix given current state
  def A_calc(self, x, y, theta, v, omega, dt, dataset):
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
  def update_mu(self, mu, dt):
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
  def kf_step(self, mu, sig, z, dt, dataset):
    # Parse state 
    x = mu[0]
    y = mu[1]
    theta = mu[2]
    v = mu[3]
    omega = mu[4]
  
    A = self.A_calc(x,y,theta,v,omega,dt, dataset)
    C = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
  
    # Covariance matrices
    R = np.identity(5) * 1e10 # State transition uncertainty
    Q = np.identity(2) * 1e10 # Measurement uncertainty
  
    # Update
    mu_next_p = self.update_mu(mu,dt)
    sig_next_p = A @ sig @ A.transpose() + R
    K = sig_next_p @ C.transpose() @ inv(C @ sig_next_p @ C.transpose() + Q)
    mu_next = mu_next_p + K @ (z - C @ mu_next_p)
    sig_next = (np.identity(5) - K @ C) @ sig_next_p
    return mu_next, sig_next

def main(dataset, sequence, model_name):
  print("Kalman filter")

  kf = KF(dataset, sequence)

  # Number of timesteps to calculate
  MAX = len(kf.vel_hat)

  # Initial conditions
  prev_time = kf.times[0]
  mu_init = np.array([kf.x[0],kf.y[0],kf.thetas[0],kf.vels[0][0],kf.vels[0][1]])
  sig_init = np.identity(5)
  mu_next     =  mu_init
  mu_next_est =  mu_init
  sig_next = sig_init
  sig_next_est = sig_init

  # Lists to save results for plotting
  x_list = []
  y_list = []
  theta_list = []

  # KF with inferred velocities
  x_list_est = []
  y_list_est = []
  theta_list_est = []

  sigmas = []
  indexes = []

  ############################ Filter loops  ####################################
  for i in range(1, MAX):
    #print("\n Iteration {}".format(i))
    curr_time = kf.times[i]
    dt = curr_time - prev_time
    prev_time = curr_time

    # Observed velocities
    if dataset == "kitti":
      z_true = kf.vels[i] # z from joy data / oxts
    elif dataset == "ouija":
      z_true = kf.opti_vels[i] # z from optitrack data

    # Run filter with ground truth velocities
    mu_next, sig_next = kf.kf_step(mu_next, sig_next, z_true, dt, dataset)
    x_list.append(mu_next[0])
    y_list.append(mu_next[1])
    theta_list.append(mu_next[2])
    #sigmas.append(np.trace(sig_next))
    #indexes.append(i)
    #print("Ground truth x {} y {} theta {} velocities {}".format(x[i],y[i],theta[i] + np.pi/2, z))

    # Run filter with inferred velocities
    z_est = kf.vel_hat[i]
    mu_next_est, sig_next_est = kf.kf_step(mu_next_est, sig_next_est, z_est, dt, dataset)
    x_list_est.append(mu_next_est[0])
    y_list_est.append(mu_next_est[1])
    theta_list_est.append(mu_next_est[2])

  ################################ Calculate error  #####################################
  prev_time = kf.times[0]
  mu_next     =  mu_init
  mu_next_est =  mu_init
  sig_next = sig_init
  sig_next_est = sig_init

  segment_length = 10
  dist_traveled = 0
  
  trans_errs = []
  rot_errs = []
  est_trans_errs = []
  est_rot_errs = []


  for i in range(1, MAX):
    #print("\n Iteration {}".format(i))
    curr_time = kf.times[i]
    dt = curr_time - prev_time
    prev_time = curr_time

    # Calculate distance traveled since last error calc
    dist = np.linalg.norm([kf.x[i] - kf.x[i-1], kf.y[i] - kf.y[i-1]])
    dist_traveled += dist

    # Observed velocities
    if dataset == "kitti":
      z_true = kf.vels[i] # z from joy data / oxts
    elif dataset == "ouija":
      z_true = kf.opti_vels[i] # z from optitrack data

    # Run filter with ground truth velocities
    mu_next, sig_next = kf.kf_step(mu_next, sig_next, z_true, dt, dataset)

    # Run filter with inferred velocities
    z_est = kf.vel_hat[i]
    mu_next_est, sig_next_est = kf.kf_step(mu_next_est, sig_next_est, z_est, dt, dataset)

    # If segment_length distance has been traveled since last reset
    # Calculate translational and rotational error
    # Reset initial conditions to ground truth
    if dist_traveled >= segment_length:
      # Calculate translational error
      # Filter with ground truth velocities
      trans_errs.append(np.linalg.norm([mu_next[0] - kf.x[i], mu_next[1] - kf.y[i]]))
      rot_errs.append(abs(mu_next[2] - kf.thetas[i]))

      # Filter with inferred velocities
      est_trans_errs.append(np.linalg.norm([mu_next_est[0] - kf.x[i], mu_next_est[1] - kf.y[i]]))
      est_rot_errs.append(abs(mu_next_est[2] - kf.thetas[i]))

      mu_init = np.array([kf.x[i],kf.y[i],kf.thetas[i],kf.vels[i][0],kf.vels[i][1]])
      sig_init = np.identity(5)
      mu_next     =  mu_init
      mu_next_est =  mu_init
      sig_next = sig_init
      sig_next_est = sig_init
      dist_traveled = 0

  # Calculate average errors
  trans_err_avg = mean(trans_errs)
  rot_err_avg = mean(rot_errs)
  est_trans_err_avg = mean(est_trans_errs)
  est_rot_err_avg = mean(est_rot_errs)

  print("Average filter error")
  print("using true velocities: translational {} rotational {}".format(trans_err_avg, rot_err_avg))
  print("using inferred velocities: translational {} rotational {}".format(est_trans_err_avg, est_rot_err_avg))

  ################################## Plot #####################################
  # If dataset is ouija, switch x and y to match mocap frame
  if dataset == "kitti":
    line_true = plt.plot(kf.x[0:MAX], kf.y[0:MAX], label="Actual")
    line_kf_true = plt.plot(x_list, y_list, label="KF using true velocities as observations")
    line_kf_est = plt.plot(x_list_est, y_list_est, label="KF using inferred vels")
    plt.ylabel("y (m)")
    plt.xlabel("x (m)")
  elif dataset == "ouija":
    line_true = plt.plot(kf.y[0:MAX], kf.x[0:MAX], label="Actual")
    line_kf_true = plt.plot(y_list, x_list, label="KF using true velocities as observations")
    line_kf_est = plt.plot(y_list_est, x_list_est, label="KF using inferred vels")
    plt.xlim(5, -5)
    plt.axis("equal")
    plt.ylabel("x in mocap frame (m)")
    plt.xlabel("y in mocap frame (m)")

  plt.setp(line_kf_est, color='b', ls='-', marker='.')
  plt.setp(line_true, color='r', ls='-', marker='.')
  plt.setp(line_kf_true, color='g', ls='-', marker='.')

  plt.legend()
  plt.title("{} Trajectory {}".format(dataset,sequence))

  # Save plot
  fig_name = "./figs/{}_{}_{}_traj_est.png".format(model_name,dataset,sequence)
  plt.savefig(fig_name, format="png")
  #plt.figure()
  #plt.plot(indexes,sigmas)
  plt.show()
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", help="dataset type", choices=["ouija", "kitti"])
  parser.add_argument("--traj_num", help="trajectory number")
  parser.add_argument("--model_name", help="name of model")
  args = parser.parse_args()
  dataset = args.dataset
  traj_num = args.traj_num
  model_name = args.model_name

  main(dataset, traj_num, model_name)
