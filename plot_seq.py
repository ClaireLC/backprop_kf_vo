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
    # These two are only used by ouija so set to None
    for_vels, ang_vels = None, None

    # Get x, y
    file_path = "./dataset/poses/" + traj_num.zfill(2) + ".txt"
    x_ind = 3
    y_ind = 7
    with open(file_path, "r") as fid:
      for i, line in enumerate(fid):
        row = [float(s) for s in line.split(" ")]
        x.append(row[x_ind])
        y.append(row[y_ind])

    # Get oxts data
    oxts_dir = "./dataset/" + traj_num.zfill(2) + "/data/"
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
    est_path = "./cnn_results/" +  model_name + "/kitti_" + traj_num + ".txt"
    with open(est_path, "r") as fid:
      reader = csv.reader(fid, delimiter=",")
      next(reader)
      for row in reader:
        x_vel_est.append(float(row[0]))
        theta_vel_est.append(float(row[1]))
  
  elif dataset_type == "ouija":
    file_path = "../test_traj_" + traj_num.zfill(1) + "/data.txt"
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
    est_path = "../test_traj_" + traj_num.zfill(1) + "/results.txt"
    with open(est_path, "r") as fid:
      reader = csv.reader(fid, delimiter=",")
      next(reader)
      for row in reader:
        #x_vel_est.append(float(row[0]))
        #theta_vel_est.append(float(row[1]))
        x_vel_est.append((float(row[0]) - 7)/5)
        theta_vel_est.append(float(row[1]) * theta_vel_scaler)

  return (x, y, x_vel, x_vel_est, theta_vel, theta_vel_est, for_vels, ang_vels)


def plot_figures(dataset_type, traj_num, x, y, x_vel, x_vel_est, theta_vel, theta_vel_est, for_vels=None, ang_vels=None):
  # Plot trajectory
  # If plotting ouija, switch x and y to match mocap frame
  """
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
  plt.title("Robot trajectory, {} Trajectory {}".format(dataset_type, traj_num))
  """

  # Plot forward velocities
  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
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
  plt.title("Forward velocity, {} Trajectory {}".format(dataset_type, traj_num))

  plt.subplot(1, 2, 2)
  if dataset_type == "kitti":
    plt.plot(range(len(theta_vel)),theta_vel, label="Ground truth")  
    plt.plot(range(len(theta_vel_est)),theta_vel_est, label="Inferred")
  elif dataset_type == "ouija":
    plt.plot(range(len(theta_vel)),theta_vel, label="From Joystick")  
    plt.plot(range(len(theta_vel_est)),theta_vel_est, label="Inferred")
    plt.plot(range(len(ang_vels)), ang_vels, label="From Optitrack")
  plt.legend()
  plt.title("Angular velocity, {} Trajectory {}".format(dataset_type, traj_num))
  plt.xlabel("timestep (frame number)")
  plt.ylabel("velocity (rad/s)")

  #plt.show(block=False)


def main(dataset_type, traj_name, model_name):
  print("Plotting trajectories")
  
  for traj in traj_name:
    data = plot_seq(dataset_type, traj)
    plot_figures(dataset_type, traj, *data)
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="kitti", help="Datset type",  choices=["kitti", "ouija"])
  parser.add_argument("--traj_num", nargs='+', help="Name of trajectory (sequence number for KITTI)")
  parser.add_argument("--model_name", help="name of model")
  args = parser.parse_args()
  
  dataset_type = args.dataset
  traj_name = args.traj_num
  model_name = args.model_name

  main(dataset_type, traj_name, model_name)
