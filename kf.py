import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import plot_seq

def test(x,y,theta,z,dt,for_acc):
  # Check state updates
  #theta += np.pi/2 # transform theta to world frame
  v = z[0]
  omega = z[1]

  mu_next = np.zeros(5)
  #mu_next[0] = v * dt * np.cos(theta+np.pi/2) + x
  #mu_next[1] = v * dt * np.sin(theta+np.pi/2) + y
  ##mu_next[2] = theta + omega * dt
  #mu_next[2] = theta + omega * dt
  ##mu_next[3] = v
  #mu_next[3] = v + dt * for_acc
  #mu_next[4] = omega
  
  # Check A matrix
  A = A_calc(x,y,theta,z[0],z[1],dt)
  #B = np.asarray([0,0,0,dt,0]) temp = np.zeros((5,1))
  #temp[3][0] = for_acc
  mu_next = A @ np.asarray([x,y,theta,z[0],z[1]]) #+ B @ temp
  return mu_next
  

def A_calc(x, y, theta, v, omega, dt):
  A = np.zeros((5,5))
  A[0,0] = 1
  A[1,1] = 1
  A[2,2] = 1
  A[3,3] = 1
  A[4,4] = 1

  #theta += np.pi/2
  A[0,2] = -1 * v * np.sin(theta) * dt
  A[0,3] = np.cos(theta) * dt
  A[1,2] = v * np.cos(theta) * dt
  A[1,3] = np.sin(theta) * dt
  A[2,4] = dt

  return(A)
  
def kf_step(mu, sig, z, dt):
  # parse state 
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

  mu_next_p = A @ mu
  #print("intermediate m {}".format(mu_next_p))
  sig_next_p = A @ sig @ A.transpose() + np.identity(5) * 1e10
  K = sig_next_p @ C.transpose() @ inv(C @ sig_next_p @ C.transpose() + (np.identity(2) * 1e-100))
  mu_next = mu_next_p + K @ (z - C @ mu_next_p)
  sig_next = (np.identity(5) - K @ C) @ sig_next_p
  print("sig {}".format(sig_next))
  print("new state {}".format(mu_next))
  return mu_next, sig_next
  
def main():
  print("Kalman filter")
  MAX = 1100
  # Open ground truth pose file
  # Save x,y,theta into lists
  file_path = "../dataset/poses/" + str(1).zfill(2) + ".txt"
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
  
  # Get timestamps
  times_file_path = "./01/times.txt"
  oxts_dir = "./01/oxts/data/"
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

  # Initial conditions
  sig_next = np.identity(5)
  prev_time = 0.0 
  z = vels[0]
  x_in = x[0]
  y_in = y[0]
  theta_in = theta[0] + np.pi/2
  mu_next = np.array([x_in,y_in,theta_in,z[0],z[1]])
  mu_next_test = np.array([x_in,y_in,theta_in,z[0],z[1]])

  # Lists to save results for plotting
  mu_list = []
  x_list = []
  y_list = []

  # Test coords
  x_list_test = []
  y_list_test = []

  sigmas = []
  indexes = []
  for i in range(1, MAX):
    print("\n Iteration {}".format(i))
    curr_time = times[i]
    dt = curr_time - prev_time
    prev_time = curr_time

    # FILTER
    z = vels[i]
    mu_next, sig_next = kf_step(mu_next, sig_next, z, dt)
    mu_list.append(mu_next)
    x_list.append(mu_next[0])
    y_list.append(mu_next[1])
    sigmas.append(np.trace(sig_next))
    indexes.append(i)
    print("Ground truth x {} y {} theta {}".format(x[i],y[i],theta[i] + np.pi/2))
    #mu_next[0] = x[i]
    #mu_next[1] = y[i]
    #mu_next[2] = theta[i] + np.pi/2
    #mu_next[3] = z[0]
    #mu_next[4] = z[1]

    # TEST
    #z = vels[i]
    #z[0] = mu_next_test[3]
    #z[1] = mu_next_test[4]
    #print("x prev: est {} actual {}".format(x_in, x[i-1]))
    #print("theta prev: {}".format(theta_in))
    #print("vels {}".format(z))

    mu_next_test = test(x_in,y_in,theta_in,z,dt,acc[i])
    x_list_test.append(mu_next_test[0])
    y_list_test.append(mu_next_test[1])
    #print("x next: est {} actual {}".format(mu_next[0], x[i]))

    # Feed calculated results back into model
    x_in = mu_next_test[0]
    y_in = mu_next_test[1]
    theta_in = mu_next_test[2]

    # Feed ground truth results into model
    #x_in = x[i]
    #y_in = y[i]
    #theta_in = theta[i]

  # Plot
  line1 = plot_seq.plot_seq(1, MAX)
  #line1 = plt.plot(x,y)
  line2 = plt.plot(x_list, y_list)
  line3 = plt.plot(x_list_test, y_list_test)
  plt.setp(line1, color='r', ls='-', marker='.')
  plt.setp(line2, color='g', ls='-', marker='.')
  plt.setp(line3, color='b', ls='-', marker='.')
  plt.figure()
  plt.plot(indexes,sigmas)
  plt.show()
  
if __name__ == "__main__":
  main()
