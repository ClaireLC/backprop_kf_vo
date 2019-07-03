"""
End-to-end model results
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
model_name = args.model_name
model_name = model_name[0]
save_plot = args.save_plot

def get_ground_truth(dataset, traj_num_str):
  """
  Get ground truth x and y coords
  """
  # Load data
  x = [] # Ground truth x position
  y = [] # Ground truth y position
  thetas = [] # Ground truth angles
  times = [] # Timestamps

  if dataset == "kitti":
    # Get ground truth poses of specified sequence
    x, y, thetas = kitti_ground_truth.get_poses(traj_num_str)
    # Get oxts data
    for_vel, ang_vel = kitti_ground_truth.get_vels(traj_num_str, len(x))
    # Get timestamps of sequence
    times = kitti_ground_truth.get_times(traj_num_str)
  if dataset == "ouija":
    file_path = "../test_traj_" + traj_num_str.zfill(1) + "/data.txt"
    # Get x, y, theta, forward and angular velocities from Optitrack data
    data_processor = ouijaOptitrack(file_path)
    times = data_processor.get_times()
    x, y = data_processor.get_xy()
    thetas = data_processor.get_theta()
  return x, y

def get_inferred(dataset, traj_num_str):
  """
  Get inferred poses from cnn_results directory
  """
  x_inf = [] # Inferred forward velocity
  y_inf = [] # Inferred angular velocity 
  if dataset == "kitti":
    path = "./cnn_results/" + model_name + "/kitti_" + traj_num_str + ".txt"
    with open(path, mode="r") as csv_fid:
      reader = csv.reader(csv_fid, delimiter=",")
      for i, row in enumerate(reader):
        if i != 0:
          x_inf.append(float(row[0]))
          y_inf.append(float(row[1]))
  
  return x_inf, y_inf

def main(dataset, traj_nums, model_name):
  print("End-to-end results")

  for traj_num_str in traj_nums:
    x, y         = get_ground_truth(dataset, traj_num_str)
    x_inf, y_inf = get_inferred(dataset, traj_num_str)
    traj_dict = {
                "ground truth": {
                                  "x": x,
                                  "y": y,
                                  "color": "r",
                                },
                "inferred":     {
                                  "x": x_inf,
                                  "y": y_inf,
                                  "color": "b",
                                },
                }
    plot_results.plot_traj(traj_dict, dataset, traj_num_str, model_name, save_plot)
    plt.show(block=False)
  plt.show()

if __name__ == "__main__":
  main(dataset, traj_nums, model_name)
