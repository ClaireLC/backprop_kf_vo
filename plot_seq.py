import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

def plot_seq(dataset_type, traj_num):
  # To save x and y coordinates
  x = []
  y = []
  x_vel = []
  y_vel = []

  # KITTI: Open seq_num.txt file
  if dataset_type == "kitti":
    file_path = "./dataset/poses/" + traj_name.zfill(2) + ".txt"
    x_ind = 3
    y_ind = 7
    with open(file_path, "r") as fid:
      for i, line in enumerate(fid):
        row = [float(s) for s in line.split(" ")]
        x.append(row[x_ind])
        y.append(row[y_ind])
  
  elif dataset_type == "ouija":
    file_path = "../test_traj_" + traj_name.zfill(1) + "/data.txt"
    x_ind = 0
    y_ind = 1
    x_vel_ind = 7
    y_vel_ind = 8
    with open(file_path, "r") as fid:
      reader = csv.reader(fid, delimiter=",")
      next(reader)
      for row in reader:
        x.append(float(row[x_ind]))
        y.append(float(row[y_ind]))
        x_vel.append(float(row[x_vel_ind]))
        y_vel.append(float(row[y_vel_ind]))

  plt.plot(x,y)
  plt.plot(x[0],y[0],'ro')
  plt.figure()
  plt.plot(range(len(x_vel)),x_vel)  
  plt.plot(range(len(y_vel)),y_vel)  
  plt.show()

def main(dataset_type, traj_name):
  print("Plotting trajectories")
  plot_seq(dataset_type, traj_name)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", help="Datset type",  choices=["kitti", "ouija"])
  parser.add_argument("--traj_num", help="Name of trajectory (sequence number for KITTI)")
  args = parser.parse_args()
  
  dataset_type = args.dataset
  traj_name = args.traj_num

  main(dataset_type, traj_name)
