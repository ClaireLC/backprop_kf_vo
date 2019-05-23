import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
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
    "vel": [forward velocity, angular velocity]
    }
  """

  def __init__(self, seq_dir, poses_dir, oxts_dir, transform=None, train=True, val_idx=9):
    """
    Args:
      seq_dir: path to root directory of preprocessed trajectory sequences
      poses_dir: path to root directory of ground truth poses
      oxts_dir: path to root directory of ground truth GPS/IMU data, which contain velocities
      trnasform: optional transform to be applied on a sample
      train: when True, creates training set, skipping val_idx sequence.
        When false, creates validation set, containing only val_idx sequences
      val_idx: The index that specifies which sequence should be the validation set. range 0-9
    """

    self.seq_dir = seq_dir
    self.oxts_dir = oxts_dir
    self.poses_dir = poses_dir
    self.transform = transform

    self.seq_len = {
      0: 4540,
      1: 1100,
      2: 4660,
      3: 270,
      4: 2760,
      5: 1100,
      6: 1100,
      7: 4070,
      8: 1590,
      9: 1200
      }

    # Generate train or validation set
    self.dataset = []
    for key, val in self.seq_len.items():
      # Skip the validation set when creating training set
      # Skip everything else when creating validation set
      if (train and key == val_idx) or (not train and key != val_idx):
          continue
      # All frames are 1 indexed
      val += 1
      for frame_num in range(1, val):
        self.dataset.append((key, frame_num, "image_2"))
      for frame_num in range(1, val):
        self.dataset.append((key, frame_num, "image_3"))
    #print(len(self.dataset))

  def __len__(self):
    return len(self.dataset)


  def __getitem__(self, idx):
    # Directly index into dataset list to get data sample information
    seq_num, frame_num, cam_num = self.dataset[idx]

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
    #print(oxts_file_path)

    for_vel_line_num = 8
    ang_vel_line_num = 19

    for_vel = None
    ang_vel = None

    fid = open(oxts_file_path)
    pose_str = None
    line = fid.readlines()
    oxts_data = [float(s) for s in line[0].split(" ")]
    for_vel = oxts_data[for_vel_line_num]
    ang_vel = oxts_data[ang_vel_line_num]
    #print(oxts_data)
    #print(for_vel, ang_vel)

    # Get timestamp
    times_path = self.seq_dir + seq_num_str + "/times.txt"
    #print(times_path)
    with open(times_path, "r") as fid:
      for i, line in enumerate(fid):
        if i == frame_num:
          curr_time = float(line)
          break

    # Format sample
    sample= {
            "curr_im": curr_im,
            "diff_im": diff_im,
            "pose": np.asarray(pose),
            "vel": np.asarray([for_vel,ang_vel]),
            "curr_time": np.asarray(curr_time),
            }

    if self.transform:
      sample = self.transform(sample)

    #print(curr_time)
    return sample

class ToTensor(object):
  """ Convert ndarrays in sample to Tensors. """

  def __call__(self, sample):
    curr_im = sample["curr_im"]
    diff_im = sample["diff_im"]
    pose    = sample["pose"]
    vel = sample["vel"]
    curr_time = sample["curr_time"]

    # Swap image axes because
    # numpy image: H x W x C
    # torch image: C x H x W
    curr_im = curr_im.transpose((2,0,1))
    diff_im = diff_im.transpose((2,0,1))

    return {
            "curr_im": torch.from_numpy(curr_im),
            "diff_im": torch.from_numpy(diff_im),
            "pose":    torch.from_numpy(pose),
            "vel":     torch.from_numpy(vel),
            "curr_time": torch.from_numpy(curr_time),
            }

class SubsetSampler(Sampler):
  def __init__(self, mask):
    self.mask = mask

  def __iter__(self):
    return iter(range(self.mask))

  def __len__(self):
    return self.mask

def main():
  seq_dir = "/mnt/disks/dataset/dataset_post/sequences/"
  poses_dir = "/mnt/disks/dataset/dataset/poses/"
  oxts_dir = "/mnt/disks/dataset/dataset_post/oxts/"
  dataset_1 = KittiDataset(seq_dir, poses_dir, oxts_dir, transform=transforms.Compose([ToTensor()]))
  dataset_2 = KittiDataset(seq_dir, poses_dir, oxts_dir, transform=transforms.Compose([ToTensor()]), train=False)
  print(len(dataset_1), len(dataset_2))

  sample = dataset_1[20601-1]
  sample = dataset_1[21140-1]
  sample = dataset_1[0]
  #print(sample["curr_time"])

if __name__ == "__main__":
  main()

