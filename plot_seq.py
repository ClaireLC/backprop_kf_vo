import csv
import argparse
import matplotlib.pyplot as plt

from preprocessing.process_ouija_data import processOuijaData

def plot_seq(dataset_type, traj_num):
  # To save x and y coordinates
  x = []
  y = []
  x_vel = []
  theta_vel = []

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

    # Get oxts data
    oxts_dir = "./dataset/" + traj_name.zfill(2) + "/oxts/data/"
    acc = []
    for i in range(len(x)):
      oxts_file_path = oxts_dir + str(i).zfill(10) + ".txt"
      fid = open(oxts_file_path) 
      line = fid.readlines()
      oxts_data = [float(s) for s in line[0].split(" ")]
      x_vel.append(oxts_data[8])
      theta_vel.append(oxts_data[19])


    # Parse inferrence results
    x_vel_est = []
    theta_vel_est = []
    est_path = "./cnn_results/" + traj_name.zfill(2) + "-results.txt"
    with open(est_path, "r") as fid:
      reader = csv.reader(fid, delimiter=",")
      next(reader)
      for row in reader:
        x_vel_est.append(float(row[0]))
        theta_vel_est.append(float(row[1]))
  
  elif dataset_type == "ouija":
    file_path = "../test_traj_" + traj_name.zfill(1) + "/data.txt"
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
        x_vel.append(float(row[x_vel_ind]))
        theta_vel.append(float(row[theta_vel_ind]))
      
    # Get forward and angular velocities from Optitrack data
    data_processor = processOuijaData(file_path)
    for_vels, ang_vels = data_processor.get_vels()
  
    # Parse inferrence results
    x_vel_est = []
    theta_vel_est = []
    x_vel_scaler = 1/20
    theta_vel_scaler = 1
    est_path = "../test_traj_" + traj_name.zfill(1) + "/results.txt"
    with open(est_path, "r") as fid:
      reader = csv.reader(fid, delimiter=",")
      next(reader)
      for row in reader:
        #x_vel_est.append(float(row[0]))
        #theta_vel_est.append(float(row[1]))
        x_vel_est.append((float(row[0]) - 7)/5)
        theta_vel_est.append(float(row[1]) * theta_vel_scaler)


  # Plot trajectory
  # If plotting ouija, switch x and y to match mocap frame
  if dataset_type == "kitti":
    plt.plot(x,y,'.')
    plt.plot(x[0],y[0],'ro')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
  elif dataset_type == "ouija":
    plt.plot(y,x,'.')
    plt.plot(y[0],x[0],'ro')
    plt.xlim(5, -5)
    plt.axis("equal")
    plt.xlabel("y in mocap frame (m)")
    plt.ylabel("x in mocap frame (m)")
  plt.title("Robot trajectory, {} Trajectory {}".format(dataset_type, traj_name))

  # Plot forward velocities
  plt.figure()
  if dataset_type == "kitti":
    plt.plot(range(len(x_vel)),x_vel, label="Ground truth")  
    plt.plot(range(len(x_vel_est)),x_vel_est, label="Inferred")
  elif dataset_type == "ouija":
    plt.plot(range(len(x_vel)),x_vel, label="From Joystick")  
    plt.plot(range(len(x_vel_est)),x_vel_est, label="Inferred")
    plt.plot(range(len(for_vels)), for_vels, label="From Optitrack")
  plt.legend()
  plt.xlabel("timestep (frame number)")
  plt.ylabel("velocity (m/s)")
  plt.title("Forward velocity, {} Trajectory {}".format(dataset_type, traj_name))

  plt.figure()
  if dataset_type == "kitti":
    plt.plot(range(len(theta_vel)),theta_vel, label="Ground truth")  
    plt.plot(range(len(theta_vel_est)),theta_vel_est, label="Inferred")
  elif dataset_type == "ouija":
    plt.plot(range(len(theta_vel)),theta_vel, label="From Joystick")  
    plt.plot(range(len(theta_vel_est)),theta_vel_est, label="Inferred")
    plt.plot(range(len(ang_vels)), ang_vels, label="From Optitrack")
  plt.legend()
  plt.title("Angular velocity, {} Trajectory {}".format(dataset_type, traj_name))
  plt.xlabel("timestep (frame number)")
  plt.ylabel("velocity (rad/s)")

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
