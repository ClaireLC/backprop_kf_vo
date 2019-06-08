import numpy as np
import csv
from scipy.spatial.transform import Rotation as R

class processOuijaData():
  """ 
  Processing Ouijabot raw data
  """
  def __init__(self, file_path):
    """
    file_path: path to raw datafile
    """
    # Load text file from filepath
    # and parse values into lists
    
    # World frame is mocap frame
    # x axis forward, y axis left, z axis up
    self.positions = [] # List of (x,y,z) positions in world frame at each timestep
    self.vels= [] # List of (x,y,z) velocities in world frame at each timestep
    self.quats = [] # List of (x,y,z,w) quaternions in world frame at each timestep
    self.times = [] # Timestamps at each timestep

    with open(file_path, "r") as csv_file:
      reader = csv.reader(csv_file, delimiter=',')

      # Extract indices of each column from header (first line)
      header = next(reader)
      indices = {}
      for i, col_name in enumerate(header):
        indices[col_name] = i
      
      for row in reader:
        x = float(row[indices["x"]])
        y = float(row[indices["y"]])
        z = float(row[indices["z"]])
        vel_x = float(row[indices["vel_x"]])
        vel_y = float(row[indices["vel_y"]])
        vel_z = float(row[indices["vel_z"]])
        qx = float(row[indices["quat_x"]])
        qy = float(row[indices["quat_y"]])
        qz = float(row[indices["quat_z"]])
        qw = float(row[indices["quat_w"]])

        self.times.append(float(row[indices["time"]]))
        self.positions.append((x,y,z))
        self.vels.append((vel_x,vel_y,vel_z))
        self.quats.append((qx,qy,qz,qw))
  
 
  def get_theta(self):
    """
    Calculates angle of robot in mocap world frame
    Rotation is represented as a rotation vector [theta_x,theta_y,theta_z]
    """
    thetas = []
    for i in range(len(self.quats)):
      r = R.from_quat(list(self.quats[i]))
      thetas.append(r.as_rotvec()[2]) # z element of rotation vector
    return thetas

  def get_vels(self):
    """
    Calculates egocentric forward and angular velocity from raw data
    Returns a list of (for_vel, ang_vel) tuples
    """
    # Threshold for determining if angle has changed
    theta_thresh = 0.0 # 0.057 degrees
    dist_thresh = 0.0

    # Get rotation vectors at each timestep
    thetas = self.get_theta()
    
    # List to hold velocity tuples (forward, angular)
    vels = []
  
    for i in range(1,len(self.times)):
      dt = self.times[i] - self.times[i-1]
 
      # If frame timestamp is the same, append previous for, ang vels
      #print(i)
      if dt == 0:
        vels.append(vels[-1])
        #print(i, for_vel, ang_vel, "\n")
        continue

      # Calculate forward velocity
      x_curr = self.positions[i][0] 
      x_prev = self.positions[i-1][0] 
      y_curr = self.positions[i][1] 
      y_prev = self.positions[i-1][1] 
      dist = np.linalg.norm([x_curr - x_prev, y_curr - y_prev])
      if abs(dist) < dist_thresh:
        dist = 0
      for_vel = dist / dt

      # Calculate angular velocity
      curr_theta_z = thetas[i]
      prev_theta_z = thetas[i-1]
      #print(curr_theta_z)
      d_theta_z = curr_theta_z - prev_theta_z
      #print(dist,dt,d_theta_z)

      if abs(d_theta_z) < theta_thresh:
        d_theta_z = 0

      prev_theta_z = curr_theta_z

      ang_vel = d_theta_z / dt

      vels.append((for_vel, ang_vel))

    return vels

def main():
  data = processOuijaData("/Users/claire/Documents/Stanford/1_Spr/test_traj_4/data.txt")

if __name__ == "__main__":
  main()
