import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
import matplotlib.pyplot as plt

class KittiDataset(Dataset):
  """
  KITTI VO trajectories with ground truth poses and forward/angular velocities
  Sample format:
    {
    "curr_img": current image,
    "diff_img": difference image,
    "pose": 4x4 transformation matrix,
    "for_vel": forward velocity,
    "ang_vel": angular velocity,
    }
  """

  def __init__(self, seq_dir, poses_dir, oxts_dir):
    """
    Args:
      seq_dir: path to root directory of preprocessed trajectory sequences
      poses_dir: path to root directory of ground truth poses
      oxts_dir: path to root directory of ground truth GPS/IMU data, which contain velocities
    """
    
    self.seq_dir = seq_dir
    self.oxts_dir = oxts_dir
    self.poses_dir = poses_dir

    self.seq_len = {
      0: 4540,
      1: 1100,
      2: 4660,
      3: 800,
      4: 270,
      5: 2760,
      6: 1100,
      7: 1100,
      8: 4070,
      9: 1590,
      10: 1200
      }

    # Generate global frame ranges for each sequence
    self.seq_ranges = {}
    prev = 0
    for i in range(11):
      start = prev + 1
      end = prev + 2 * self.seq_len[i]
      self.seq_ranges[i] = (start,end)  
      prev = end
    print(self.seq_ranges)

  def __len__(self):
    total_len = 0
    for key, val in self.seq_len.items():
      total_len += val
    # Now, double total length because we consider images from each camera as separate data points
    total_len *= 2
    return total_len
    
  def __getitem__(self, idx):
    # Decompose index into sequence number and frame number
    idx += 1
    seq_num = None
    cam_num = None
    frame_num = None
    for key, val in self.seq_ranges.items():
      if ((idx >= val[0]) and (idx <= val[1])):
        seq_num = key
  
    # Use sequence number to determine camera and frame number
    if ((idx >= self.seq_ranges[seq_num][0]) and (idx <= self.seq_ranges[seq_num][0] + self.seq_len[seq_num] - 1)):
      frame_num = idx - self.seq_ranges[seq_num][0] + 1
      cam_num = "image_2"
    else:
      frame_num = idx - self.seq_ranges[seq_num][0] - self.seq_len[seq_num] + 1
      cam_num = "image_3"
    
    print(seq_num, frame_num, cam_num)

    # Get current and difference images
    frame_digits = 6
    seq_digits = 2
  
    seq_num_str = str(seq_num).zfill(seq_digits)
    frame_num_str = str(frame_num).zfill(frame_digits) + ".png"

    curr_im_path = self.seq_dir + seq_num_str + "/" + cam_num + "/current/" + frame_num_str
    diff_im_path = self.seq_dir + seq_num_str + "/" + cam_num + "/diff/" + frame_num_str
    curr_im = io.imread(curr_im_path)
    diff_im = io.imread(diff_im_path)

    # Get ground truth pose from seq_num.txt file
    poses_file_path = self.poses_dir + seq_num_str + ".txt"
    fid = open(poses_file_path)
    pose_str = None
    for i, line in enumerate(fid):
      if i == frame_num - 1:
        pose_str = line
      if i > frame_num - 1:
        break
    pose = [float(s) for s in pose_str.split(" ")] 
    #print(pose)

    # Get ground truth velocities from oxts frame_num.txt file
    oxts_file_digits = 10
    oxts_file_str = str(frame_num).zfill(oxts_file_digits)
    oxts_file_path = self.oxts_dir + seq_num_str + "/data/" + oxts_file_str + ".txt"
    print(oxts_file_path)
    
    for_vel_line_num = 7
    ang_vel_line_num = 18

    for_vel = None
    ang_vel = None

    fid = open(oxts_file_path)
    pose_str = None
    line = fid.readlines()
    oxts_data = [float(s) for s in line[0].split(" ")]
    for_vel = oxts_data[for_vel_line_num]
    ang_vel = oxts_data[ang_vel_line_num]
    print(oxts_data)
    #print(for_vel, ang_vel)

    # Format sample
    sample= {
            "curr_im": curr_im,
            "diff_im": diff_im,
            "pose": pose,
            "for_vel": for_vel,
            "ang_vel": ang_vel,
            }
    #print(sample)

    return
    
def main():
  seq_dir = "/mnt/disks/dataset/dataset_post/sequences/"
  poses_dir = "/mnt/disks/dataset/dataset/poses/"
  oxts_dir = "/mnt/disks/dataset/dataset_post/oxts/"
  dataset = KittiDataset(seq_dir, poses_dir, oxts_dir)
  print(len(dataset))

  sample = dataset[0]
  #sample = dataset[46380 - 1]

if __name__ == "__main__":
  main()

