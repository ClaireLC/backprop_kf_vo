"""
Compute piecewise KF for inferrence results from feed forward network
Calculate error
Plot results
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import csv
import argparse
from statistics import mean
import os
from tqdm import tqdm

from preprocessing.ouija_optitrack import ouijaOptitrack
from kf import KF
import kitti_ground_truth, plot_results

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='kitti', help="dataset type", choices=["ouija", "kitti"])
parser.add_argument("--traj_num", nargs='+', help="trajectory number. Can be multiple separated with space")
parser.add_argument("--model_name", nargs='+', help="name of model. Can be multiple separated with space")
parser.add_argument("--save_plot", default=False, type=bool, help="Saves plot if set to True")
args = parser.parse_args()
dataset = args.dataset
traj_nums = args.traj_num
model_names = args.model_name
save_plot = args.save_plot

err_save_dir = "./traj_error/"

def get_ground_truth(dataset, traj_num_str, model_name):
  """
  Get ground truth state information
  """
  # Load data
  x = [] # Ground truth x position
  y = [] # Ground truth y position
  theta = [] # Ground truth angles
  for_vel = [] # Ground truth forward velocity
  ang_vel = [] # Ground truth angular velocity
  times = [] # Timestamps

  if dataset == "kitti":
    # Get ground truth poses of specified sequence
    x, y, theta = kitti_ground_truth.get_poses(traj_num_str)
    # Get oxts data
    for_vel, ang_vel = kitti_ground_truth.get_vels(traj_num_str, len(x))
    # Get timestamps of sequence
    times = kitti_ground_truth.get_times(str(traj_num_str))

  if dataset == "ouija":
    # Get x, y, theta, forward and angular velocities from Optitrack data
    data_processor = ouijaOptitrack(file_path)
    times = data_processor.get_times()
    x, y = data_processor.get_xy()
    theta = data_processor.get_theta()
    for_vel, ang_vel = data_processor.get_vels()

  return x, y, theta, for_vel, ang_vel, times

def get_joystick_vels(traj_num_str):
  """
  Get joystick forward and angular commands for Ouijabot
  """
  for_vel_joy = []
  ang_vel_joy = []

  file_path = "../test_traj_" + traj_num_str.zfill(1) + "/data.txt"
  for_vel_ind = 8
  ang_vel_ind = 10
  with open(file_path, "r") as fid:
    reader = csv.reader(fid, delimiter=",")
    next(reader)
    for row in reader:
      for_vel_joy.append(float(row[for_vel_ind]))
      ang_vel_joy.append(float(row[ang_vel_ind]))

  return for_vel_joy, ang_vel_joy

def get_inferred(dataset, traj_num_str, model_name):
  """
  Get inferred velocites from cnn_results directory
  """
  for_vel_inf = [] # Inferred forward velocity
  ang_vel_inf = [] # Inferred angular velocity 

  if dataset == "kitti":
    path = "./cnn_results/" + model_name + "/kitti_" + traj_num_str + ".txt"
  if dataset == "ouija":
    # Get inferred velocities from cnn result txt files
    path = "../test_traj_" + traj_num_str.zfill(1) +"/results.txt"

  with open(path, mode="r") as csv_fid:
    reader = csv.reader(csv_fid, delimiter=",")
    next(reader)
    for i, row in enumerate(reader):
      for_vel_inf.append(float(row[0]))
      ang_vel_inf.append(float(row[1]))

  return for_vel_inf, ang_vel_inf
  
def compute_kf(mu_init, sig_init, times, observations):
  """
  Compute steps of the filter given inital conditions, times, and observations
  mu_init: initial 5 dimensional state
  sig_init: intial 5x5 covariance matrix
  times: list of timestamps in sequence
  observations: 2d list of observations [[for_vel, ang_vel],...,] over sequence
  """
  # Kalman filter update class
  kf = KF()

  # Initial conditions
  prev_time = times[0]
  mu_next   =  mu_init
  sig_next  = sig_init

  # Lists of mu and sigmas at every timestep
  mus    = [] 
  sigmas = []

  ############################ Filter loop  ####################################
  for i in range(1, len(observations)):
    #print("\n Iteration {}".format(i))
    curr_time = times[i]
    dt = curr_time - prev_time
    prev_time = curr_time

    # Run filter with inferred velocities
    z = observations[i-1]
    mu_next, sig_next = kf.step(mu_next, sig_next, z, dt)

    mus.append(mu_next)
    sigmas.append(sig_next)

  return np.array(mus), np.array(sigmas)

def compute_err(x, y, theta, for_vel, ang_vel, times, obs, seq_length):
  """
  Compute error of filter across sequences with seq_length number of timesteps
  x: list of ground truth x coords
  y: list of ground truth y coords
  theta: list of ground truth angles
  for_vel: list of ground truth forward velocities
  ang_vel: list of ground truth angular velocities
  times: list of timesteps
  seq_length: number of timesteps of sequence to calculate error over
  """
  trans_err = 0
  rot_err = 0

  sig_init = np.identity(5)

  for start_ind in tqdm(range(len(times) - seq_length + 1)):
    # Set initial state at start_ind
    #print("start ind {}".format(start_ind))
    mu_init = np.array([
                        x[start_ind],
                        y[start_ind],
                        theta[start_ind],
                        for_vel[start_ind],
                        ang_vel[start_ind],
                       ])

    #print("mu init {}".format(mu_init))

    curr_seq_x = x[start_ind:start_ind + seq_length]
    curr_seq_y = y[start_ind:start_ind + seq_length]
    curr_seq_theta = theta[start_ind:start_ind + seq_length]

    # Compute kalman filter over sequence
    mu, sig = compute_kf(
                        mu_init,
                        sig_init,
                        times[start_ind:start_ind + seq_length],
                        obs[start_ind:start_ind + seq_length],
                        )
    # Compute total distance traveled over sequence
    dist = 0
    for i in range(len(mu)):
      curr_mu = mu[i]
      #print(curr_mu)
      #print(curr_seq_x[i])
      #print(curr_seq_y[i])
      dist += np.linalg.norm([
                              curr_mu[0] - curr_seq_x[i+1],
                              curr_mu[1] - curr_seq_y[i+1],
                            ])
      #print(dist)
      #print()
  
    # Distance from start to end point
    dist = np.linalg.norm([
                            mu[0,0] - mu[-1,0] ,
                            mu[0,1] - mu[-1,1] ,
                          ])

    #print("Total distance traveled over {} steps: {}".format(len(mu),dist))

    # Get final error, averaged over distance
    abs_trans_err = np.linalg.norm([
                                    mu[-1,0] - curr_seq_x[-1],
                                    mu[-1,1] - curr_seq_y[-1],
                                  ])
    abs_rot_err = abs(mu[-1,2] - curr_seq_theta[-1])

    trans_err += (abs_trans_err / dist)
    rot_err   += (abs_rot_err / dist)

  # Average error over number of sequences
  trans_err /= (len(times) - seq_length + 1)
  rot_err   /= (len(times) - seq_length + 1)

  return trans_err, rot_err

def main(dataset, traj_nums, model_names):
  print("Kalman filter")

  sig_init = np.identity(5)

  # Plot trajectories 
  for plt_idx in range(len(model_names)):
    # save error results to csv
    # write header
    err_save_fname = err_save_dir + model_names[plt_idx] + ".txt"
    with open(err_save_fname, mode="w+") as fid:
      writer = csv.writer(fid, delimiter=",")
      writer.writerow(["traj_num", "trans_err_gt", "rot_err_gt", "trans_err_inf", "rot_err_gt"])
  
    for traj_num_str in traj_nums:
    #plt.figure(figsize=(5 * len(model_name), 5))
      # Get ground truth
      x_gt, y_gt, theta_gt, for_vel_gt, ang_vel_gt, times = get_ground_truth(dataset, traj_num_str, model_names[plt_idx])

      # Compute Kalman filter over entire trajectory with ground truth velocities
      mu_init = np.array([
                          x_gt[0],
                          y_gt[0],
                          theta_gt[0],
                          for_vel_gt[0],
                          ang_vel_gt[0],
                         ])
      obs_gt = np.stack((for_vel_gt, ang_vel_gt), axis = 1)
      mu_gt, sig_gt = compute_kf(mu_init, sig_init, times, obs_gt)
      
      # Compute Kalman filter over entire trajectory with inferred velocities
      # Get inferred velocities
      for_vel_inf, ang_vel_inf = get_inferred(dataset, traj_num_str, model_names[plt_idx])
      obs_inf = np.stack((for_vel_inf, ang_vel_inf), axis = 1)
      mu_inf, sig_inf = compute_kf(mu_init, sig_init, times, obs_inf)

      # Plot entire trajectories
      traj_dict = {
                  "actual": {
                                    "x": x_gt,
                                    "y": y_gt,
                                    "color": "r",
                                  },
                  "KF with ground truth velocities":     {
                                    "x": mu_gt[:,0],
                                    "y": mu_gt[:,1],
                                    "color": "b",
                                  },
                  "KF with inferred velocities":     {
                                    "x": mu_inf[:,0],
                                    "y": mu_inf[:,1],
                                    "color": "g",
                                  },
                  }
      plot_results.plot_traj(traj_dict, dataset, traj_num_str, model_names[plt_idx], save_plot)

      # Calculate error 
      seq_length = 100 # Number of timesteps of sequence
      #seq_length = len(x_gt)

      # KF with ground truth velocities
      print("Computing error over trajectory with ground truth velocities")
      trans_err_gt, rot_err_gt = compute_err(
                                            x_gt,
                                            y_gt,
                                            theta_gt, 
                                            for_vel_gt,
                                            ang_vel_gt,
                                            times,
                                            obs_gt,
                                            seq_length
                                            )

      print(trans_err_gt, rot_err_gt)

      # KF with ground inferred velocities
      print("Computing error over trajectory with inferred velocities")
      trans_err_inf, rot_err_inf = compute_err(
                                            x_gt,
                                            y_gt,
                                            theta_gt, 
                                            for_vel_gt,
                                            ang_vel_gt,
                                            times,
                                            obs_inf,
                                            seq_length
                                            )

      print(trans_err_inf, rot_err_inf)

      # Save errors to csv
      with open(err_save_fname, mode="a+") as fid:
        writer = csv.writer(fid, delimiter=",")
        writer.writerow([traj_num_str, trans_err_gt, rot_err_gt, trans_err_inf, rot_err_gt])

      
    plt.show(block=False)
  plt.show()

if __name__ == "__main__":
  main(dataset, traj_nums, model_names)
