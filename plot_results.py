"""
Functions for plotting results
Trajectories
Velocities
"""
import csv
import argparse
import matplotlib.pyplot as plt
import os

def plot_traj(trajectories_dict, dataset, traj_num, model_name, save_plot):
  """
  Plot trajectories in trajectories_dict
  If plotting ouija, switch x and y to match mocap frame
  trajectories_dict: {'description':
                         {'x': [],                         # List of x coords
                          'y': [],                         # List of y coords
                          'color': 'matplotlib color name' # Color string
                         }
                     }
  """ 
  plt.figure()

  if dataset== "kitti":
    for description, coords in trajectories_dict.items():
      x = coords["x"]
      y = coords["y"]
      plt.plot(x,y, ls="-", marker=".", color=coords["color"], label=description)
      plt.plot(x[0],y[0], marker='o', color=coords["color"])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
  elif dataset == "ouija":
    for description, coords in trajectories_dict.items():
      plt.plot(y,x, ls="-", marker=".", color=coords["color"], label=description)
      plt.plot(y[0],x[0], marker="o", color=coords["color"])
    plt.xlim(5, -5)
    plt.axis("equal")
    plt.xlabel("y in mocap frame (m)")
    plt.ylabel("x in mocap frame (m)")
  plt.legend()
  plt.title("{} Trajectory {}".format(dataset, traj_num))

  if save_plot:
      # Save plot
      os.makedirs("./figs/{}".format(model_name), exist_ok=True)
      fig_name = "./figs/{}/{}_{}_traj.png".format(model_name, dataset, traj_num)
      plt.savefig(fig_name, format="png")

def plot_vel(vels_dict, dataset, traj_num, model_name, save_plot):
  """
  Plot forward and angular velocities in vels_dict
  in two side-by-side plots 
  vels_dict: {'description (ie. ground truth)':
                         {'for_vel': [],                   # List of x coords
                          'ang_vel': [],                   # List of y coords
                          'color': 'matplotlib color name' # Color string
                         }
                     }
  """ 
  # Plot forward velocities
  plt.figure(figsize=(10, 4))

  plt.subplot(1, 2, 1)
  for description, vels in vels_dict.items():
    for_vel = vels["for_vel"]
    plt.plot(range(len(for_vel)), for_vel, color=vels["color"], label=description)
  plt.legend()
  plt.xlabel("Timestep (frame number)")
  plt.ylabel("Velocity (m/s)")
  plt.title("Forward velocity, {} Trajectory {}".format(dataset, traj_num))

  # Plot angular velocities
  plt.subplot(1, 2, 2)
  for description, vels in vels_dict.items():
    ang_vel = vels["ang_vel"]
    plt.plot(range(len(ang_vel)), ang_vel, color=vels["color"], label=description)
  plt.legend()
  plt.title("Angular velocity, {} Trajectory {}".format(dataset, traj_num))
  plt.xlabel("Timestep (frame number)")
  plt.ylabel("Velocity (rad/s)")

  if save_plot:
      # Save plot
      os.makedirs("./figs/{}".format(model_name), exist_ok=True)
      fig_name = "./figs/{}/{}_{}_vel.png".format(model_name, dataset_type, traj_num)
      plt.savefig(fig_name, format="png")

