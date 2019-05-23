import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_seq(seq_num):
  # Open seq_num.txt file
  file_path = "../dataset/poses/" + str(seq_num).zfill(2) + ".txt"

  x = []
  y = []
  x_ind = 3
  y_ind = 7
  with open(file_path, "r") as fid:
    for i, line in enumerate(fid):
      row = [float(s) for s in line.split(" ")]
      x.append(row[x_ind])
      y.append(row[y_ind])

  plt.plot(x,y)  
  plt.plot(0,0,'ro')
  plt.show()

def main():
  print("Plotting trajectories")
  plot_seq(1)

if __name__ == "__main__":
  main()
