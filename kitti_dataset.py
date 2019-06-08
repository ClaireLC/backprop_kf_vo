import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
from skimage import io
import matplotlib.pyplot as plt
import time

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

  def __init__(self, seq_dir, poses_dir, oxts_dir, transform=None, mode=None):
    """
    Args:
      seq_dir: path to root directory of preprocessed trajectory sequences
      poses_dir: path to root directory of ground truth poses
      oxts_dir: path to root directory of ground truth GPS/IMU data, which contain velocities
      trnasform: optional transform to be applied on a sample
      train: when True, creates training set, else creates validation set
    """

    self.seq_dir = seq_dir
    self.oxts_dir = oxts_dir
    self.poses_dir = poses_dir
    self.transform = transform

    start_time = time.time()
    # Load existing parsed data if possible. Otherwise create and store them.
    self.dataset = None
    if mode == "infer" and os.path.isfile("inorder_dataset.npy"):
      print("Loading inorder_dataset.npy, expected wait time: ", 0)
      self.dataset = np.load("inorder_dataset.npy", allow_pickle=True)
      print("Done, it took time: ", time.time() - start_time)

    elif mode == "train" and os.path.isfile("train_dataset.npy"):
      print("Loading train_dataset.npy, expected wait time: ", 0)
      self.dataset = np.load("train_dataset.npy", allow_pickle=True)
      print("Done, it took time: ", time.time() - start_time)

    elif mode == "val" and os.path.isfile("val_dataset.npy"):
      print("Loading val_dataset.npy, expected wait time: ", 0)
      self.dataset = np.load("val_dataset.npy", allow_pickle=True)
      print("Done, it took time: ", time.time() - start_time)

    else:
      self.process_dataset(mode)
      self.hydrate_dataset()
      if mode == "train":
        np.save("train_dataset", np.asarray(self.dataset))
      elif mode == "val":
        np.save("val_dataset", np.asarray(self.dataset))
      elif mode == "infer":
        np.save("inorder_dataset", np.asarray(self.dataset))


  def __len__(self):
    return len(self.dataset)


  def __getitem__(self, idx):
    # Directly index into dataset list to get data sample information
    return self.dataset[idx]


  def hydrate_dataset(self):
    """
    Takes in a processed dataset, a list of tuples (return type of process_dataset)
    Returns a list that contains all the images and relevant data in the desired format
    __getitem__ can directly index into this list.
    """
    data = []
    for datapoint in self.dataset:
      data.append(self.format_datapoint(datapoint))
    self.dataset = data


  def format_datapoint(self, datapoint):
    """
    Takes in a datapoint which is currently formated as
    datapoint = curr_im_path, diff_im_path, velocity, seq_num_str

    Output a datapoint in the format of 
    {
      "curr_im": curr_im,
      "diff_im": diff_im,
      "vel": velocity,
    }
    And then this dict is put through self.tansform
    """
    curr_im_path, diff_im_path, velocity, seq_num_str = datapoint

    curr_im = io.imread(curr_im_path)
    diff_im = io.imread(diff_im_path)

    # Currently the cnn doesn't use this. Commented out to speed up dataloader
    #pose = self.get_groudtruth_poses(seq_num_str)

    # Currently the cnn doesn't use this. Commented out to speed up DataLoader
    # curr_time = self.get_timestamp(self, seq_num_str)

    # Format sample
    sample= {
            "curr_im": curr_im,
            "diff_im": diff_im,
            "vel": velocity,
            #"pose": np.asarray(pose),
            #"curr_time": np.asarray(curr_time),
            }

    if self.transform:
      sample = self.transform(sample)

    return sample 


  def process_dataset(self, mode):
    """
    Creates a list of tuples. Each tuple corresponds to a data sample and contains information
    that helps retrieve images and data

    The list is stored in self.dataset
    Tuple format: (curr_im_path, diff_im_path, velocity, seq_num_str)
    --> (current image path, diff image path, [forward velocity, angular velocity],
         sequence number padded with zeros in front)
    """
    # Sequences and how many data samples each sequence contains. Numbers should be x2 for two cameras
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

    # Update self.dataset to contain list of (seq_num, frame_num, cam_num)
    self.create_data_tuples(mode)

    # Clean up the data inside dataset
    formated_dataset = []
    for sample in tqdm(self.dataset):
        seq_num, frame_num, cam_num = int(sample[0]), int(sample[1]), sample[2]

        # Pad sequence number with zeros in front
        seq_digits = 2
        seq_num_str = str(seq_num).zfill(seq_digits)

        # Get images
        curr_im_path, diff_im_path = self.get_image_paths(seq_num_str, frame_num, cam_num)
        # Get velocity: [for_vel, ang_vel]
        velocity = self.get_velocity(seq_num_str, frame_num)

        formated_dataset.append((curr_im_path, diff_im_path, velocity, seq_num_str))

    self.dataset = formated_dataset


  def create_data_tuples(self, mode):
    """
    Processes or loads all the data according to its sequence number, frame number and camera number
    and stores this information in a tuple of three.
    train: Indicates whether the dataset should be be train or val
    self.dataset in the end is a list of all the tuples corresponding to the train or val data samples
    """
    # Store / load the train val dataset
    self.dataset = []

    if mode == "infer":
      if os.path.isfile("inorder_dataset.npy"):
        self.dataset = np.load("inorder_dataset.npy", )
      else:
        # Generate train or validation set
        for key, val in self.seq_len.items():
          # All frames are 1 indexed
          val += 1
          for frame_num in range(1, val):
            self.dataset.append((key, frame_num, "image_2"))
          for frame_num in range(1, val):
            self.dataset.append((key, frame_num, "image_3"))

        # Save the data distribution in order
        self.dataset = np.asarray(self.dataset)
        np.save("inorder_dataset", self.dataset)
    else:
      # If split data exists
      if os.path.isfile("train_val_split.npy"):
        self.dataset = np.load("train_val_split.npy")
      else:
        # Generate train or validation set
        for key, val in self.seq_len.items():
          # All frames are 1 indexed
          val += 1
          for frame_num in range(1, val):
            self.dataset.append((key, frame_num, "image_2"))
          for frame_num in range(1, val):
            self.dataset.append((key, frame_num, "image_3"))

        # Shuffle and save the data distribution
        self.dataset = np.asarray(self.dataset)
        np.random.shuffle(self.dataset)
        np.save("train_val_split", self.dataset)

      # Split the data that's created or loaded
      train_dataset, val_dataset = np.split(self.dataset, [int(0.9 * len(self.dataset))])
      if mode == "train":
        self.dataset = train_dataset
      elif mode == "val":
        self.dataset = val_dataset


  def get_groudtruth_poses(self, seq_num_str):
    """
    Gets ground truth pose from seq_num.txt file
    """
    poses_file_path = self.poses_dir + seq_num_str + ".txt"
    fid = open(poses_file_path)
    pose_str = None
    for i, line in enumerate(fid):
      if i == frame_num - 1:
        pose_str = line
      if i > frame_num - 1:
        break
    pose = [float(s) for s in pose_str.split(" ")]
    return pose


  def get_timestamp(self, seq_num_str):
    """
    Gets timestamp
    """
    times_path = self.seq_dir + seq_num_str + "/times.txt"
    with open(times_path, "r") as fid:
      for i, line in enumerate(fid):
        if i == frame_num:
          curr_time = float(line)
          return curr_time


  def get_image_paths(self, seq_num_str, frame_num, cam_num):
    """
    Input: seq_num zero padded, frame_num, cam_num
    Output: current image path, diff image path
    """
    frame_digits = 6
    frame_num_str = str(frame_num).zfill(frame_digits) + ".png"

    curr_im_path = self.seq_dir + seq_num_str + "/" + cam_num + "/current/" + frame_num_str
    diff_im_path = self.seq_dir + seq_num_str + "/" + cam_num + "/diff/" + frame_num_str

    return curr_im_path, diff_im_path


  def get_velocity(self, seq_num_str, frame_num):
    """
    Returns [forward velocity, angular velocity]
    """
    oxts_file_digits = 10
    oxts_file_str = str(frame_num).zfill(oxts_file_digits)
    oxts_file_path = self.oxts_dir + seq_num_str + "/data/" + oxts_file_str + ".txt"
    print(oxts_file_path)

    for_vel_line_num = 8
    ang_vel_line_num = 19

    with open(oxts_file_path, 'r') as f:
      line = f.readline()
      oxts_data = [float(vel) for vel in line.split(" ")]

    for_vel = oxts_data[for_vel_line_num]
    ang_vel = oxts_data[ang_vel_line_num]

    return np.asarray([for_vel, ang_vel])


class ToTensor(object):
  """ Convert ndarrays in sample to Tensors. """

  def __call__(self, sample):
    curr_im = sample["curr_im"]
    diff_im = sample["diff_im"]
    vel = sample["vel"]
    #pose    = sample["pose"]
    #curr_time = sample["curr_time"]

    # Swap image axes because
    # numpy image: H x W x C
    # torch image: C x H x W
    curr_im = curr_im.transpose((2,0,1))
    diff_im = diff_im.transpose((2,0,1))

    return {
            "curr_im": torch.from_numpy(curr_im),
            "diff_im": torch.from_numpy(diff_im),
            "vel":     torch.from_numpy(vel),
            #"pose":    torch.from_numpy(pose),
            #"curr_time": torch.from_numpy(curr_time),
            }

class SubsetSampler(Sampler):
  def __init__(self, mask):
    self.mask = mask

  def __iter__(self):
    return iter(range(self.mask))

  def __len__(self):
    return self.mask

class SequenceSampler(Sampler):
  """
  Samples a specified sequence and camera id trajectory from the dataset
  """
  def __init__(self, sequence_num, camera_num):
    """
    sequence_num: sequence 0-9
    camera_num: camera image 2 or 3 (left or right camera)
    """
    # Sequences and how many data samples each sequence contains. Numbers should be x2 for two cameras
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

    # Calculate start and end indexes in dataset for given sequence num and camera num
    self.start_ind = 0

    for i in range(sequence_num):
      self.start_ind += self.seq_len[i] * 2

    # Offset for second camera
    self.start_ind += (camera_num - 2) * self.seq_len[sequence_num]

    self.end_ind = self.start_ind + self.seq_len[sequence_num] - 1
    print("start {} end {}".format(self.start_ind, self.end_ind))

  def __iter__(self):
    return iter(range(self.start_ind, self.end_ind + 1))

  def __len__(self):
    return self.end_ind - self.start_ind + 1

def main():
  seq_dir = "/mnt/disks/dataset/dataset_post/sequences/"
  poses_dir = "/mnt/disks/dataset/dataset/poses/"
  oxts_dir = "/mnt/disks/dataset/dataset_post/oxts/"
  dataset_1 = KittiDataset(seq_dir, poses_dir, oxts_dir, transform=transforms.Compose([ToTensor()]), mode="train")
  dataset_2 = KittiDataset(seq_dir, poses_dir, oxts_dir, transform=transforms.Compose([ToTensor()]), mode="val")
  dataset_3 = KittiDataset(seq_dir, poses_dir, oxts_dir, transform=transforms.Compose([ToTensor()]), mode="infer")
  #print(len(dataset_1), len(dataset_2))

  ##sample = dataset_1[20601-1]
  #sample = dataset_1[21140-1]
  #sample = dataset_1[0]
  #print(sample)
  #print(sample["curr_time"])


if __name__ == "__main__":
  main()

