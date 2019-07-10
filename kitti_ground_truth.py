"""
Functions for getting KITTI dataset ground truth poses and velocities
from text files in ./dataset/ directory.
"""
import numpy as np

def get_poses(traj_num_str):
  """
  Get ground truth pose
  Return lists: x, y, and theta
  """
  x = []
  y = []
  theta = []

  # Get x, y
  file_path = "./dataset/poses/" + traj_num_str.zfill(2) + ".txt"
  x_ind = 3
  y_ind = 11
  with open(file_path, "r") as fid:
    for i, line in enumerate(fid):
      row = [float(s) for s in line.split(" ")]
      x.append(row[x_ind])
      y.append(row[y_ind])

      # Get theta from pose transformation matrix
      # Add 90 degrees to transpose to world frame
      if np.arcsin(row[0]) > 0:
        theta.append(np.arccos(row[0]) + np.pi/2)
      else:
        theta.append(np.arccos(row[0]) * -1 + np.pi/2)

  return x, y, theta

def get_vels(traj_num_str, num_samples):
  """
  Get ground truth forward and angular velocities
  Return lists: for_vel, ang_vel
  """
  for_vel = []
  ang_vel = []
  # Get oxts data
  oxts_dir = "./dataset/" + traj_num_str.zfill(2) + "/data/"
  for i in range(num_samples):
    oxts_file_path = oxts_dir + str(i).zfill(10) + ".txt"
    fid = open(oxts_file_path) 
    line = fid.readlines()
    oxts_data = [float(s) for s in line[0].split(" ")]
    for_vel.append(oxts_data[8])
    ang_vel.append(oxts_data[19])
  return for_vel, ang_vel

def get_times(traj_num_str):
  """
  Get timestamps of each frame in trajectory
  Return list of timestamps
  """
  times = []
  # Get timestamps of sequence
  times_file_path = "./dataset/" + traj_num_str.zfill(2) + "/times.txt"
  with open(times_file_path, "r") as fid:
    for i, line in enumerate(fid):
      times.append(float(line))
  return times

