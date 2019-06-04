import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
from skimage import io
import matplotlib.pyplot as plt

save_dir = "/mnt/disks/dataset/"

class KittiDatasetSeq(Dataset):
  """
  KITTI VO trajectories with ground truth poses and forward/angular velocities

    {
    "curr_img": current image,
    "diff_img": difference image,
    "pose": 4x4 transformation matrix,
    "vel": [forward velocity, angular velocity]
    }

  Sample format:
  [(image, state, time), ...]
  image : actual image stacked with diff image
  state : (x, y, theta, v_for, v_ang)
  """

  def __init__(self, seq_dir, poses_dir, oxts_dir, seq_length=100, mode=None):
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
    self.seq_length = seq_length

    self.transform = transforms.Compose([ToTensor()])

    # Load existing parsed data if possible. Otherwise create and store them.
    self.dataset = None
    if mode == "infer" and os.path.isfile(save_dir + "inorder_dataset_seq.npy"):
      print("Loading inorder_dataset_seq.npy ...")
      self.dataset = np.load(save_dir + "inorder_dataset_seq.npy", allow_pickle=True)
      print("Done")
    elif mode == "train" and os.path.isfile(save_dir + "train_dataset_seq.npy"):
      print("Loading train_dataset_seq.npy")
      self.dataset = np.load(save_dir + "train_dataset_seq.npy", allow_pickle=True)
      print("Done")
    elif mode == "val" and os.path.isfile(save_dir + "val_dataset_seq.npy"):
      print("Loading val_dataset_seq.npy")
      self.dataset = np.load(save_dir + "val_dataset_seq.npy", allow_pickle=True)
      print("Done")
    else:
      print("Creating dataset... May take a while")
      self.process_dataset(mode)
      print("Putting data into sequence...")
      self.put_data_into_sequence()
      print("Saving...")
      if mode == "train":
        np.save(save_dir + "train_dataset_seq", np.asarray(self.dataset))
      elif mode == "val":
        np.save(save_dir + "val_dataset_seq", np.asarray(self.dataset))
      elif mode == "infer":
        np.save(save_dir + "inorder_dataset_seq", np.asarray(self.dataset))
      print("Done")


  def __len__(self):
    return len(self.dataset)


  def __getitem__(self, idx):
    # Return a list of tuples
    return self.transform(self.dataset[idx])

  def put_data_into_sequence(self):
    """
    Takes the result of process_dataset and puts every thing into the format of [(image, state, time), ...] for each datapoint
    The length of the list is specified by self.seq_length (eg 100)
    """
    data = []
    # Base index keeps track of which video sequence we're in, so index i below can always start with 0 for each video sequence
    base_idx = 0
    # Loop through each video sequence
    for vid_seq_num in tqdm(self.seq_len):
      # Compute the number of frames in the video sequence. * 2 for two cameras
      num_frames = self.seq_len[vid_seq_num] * 2
      # Loop through the frames seq_length at a time
      for i in tqdm(range(0, num_frames, self.seq_length)):
        if i + self.seq_length < num_frames:
          sequence_data = self.dataset[base_idx + i : base_idx + i + self.seq_length]
          sequence_data_formated = []

          for datapoint in sequence_data:
            formated_datapoint = self.format_datapoint(datapoint)
            sequence_data_formated.append(formated_datapoint)

          data.append(sequence_data_formated)
      base_idx += num_frames
      break

    self.dataset = data


  def format_datapoint(self, datapoint):
    """
    Takes in a datapoint which is currently formated as
    datapoint = curr_im_path, diff_im_path, velocity, x, y, theta, cur_time, seq_num_str

    Output a datapoint in the format of (image, state, time)
    image : cur_image and diff_image put side by side into one image
    state : (x, y, theta, v_for, v_ang)
    time : current time of the frame
    """
    curr_im_path, diff_im_path, velocity, x, y, theta, cur_time, seq_num_str = datapoint

    # Get images
    curr_im = io.imread(curr_im_path)
    diff_im = io.imread(diff_im_path)

    # Swap image axes because
    # numpy image: H x W x C
    # torch image: C x H x W
    curr_im = curr_im.transpose((2,0,1))
    diff_im = diff_im.transpose((2,0,1))

    # Combine the images together by putting them side by side
    compos_image = np.concatenate((curr_im, diff_im), 1)
    for_vel = velocity[0]
    ang_vel = velocity[1]
    state = (x, y, theta, for_vel, ang_vel)

    return (compos_image, state, cur_time)


  def process_dataset(self, mode):
    """
    Creates a list of tuples. Each tuple corresponds to a data sample and contains information
    that need to be retrieved by __getitem__

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
        # Get pose
        x, y, theta = self.get_groudtruth_poses(seq_num_str, frame_num)
        # Get current time
        cur_time = self.get_timestamp(seq_num_str, frame_num)

        formated_dataset.append((curr_im_path, diff_im_path, velocity, x, y, theta, cur_time, seq_num_str))

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
        self.init_dataset()
        np.save("inorder_dataset", self.dataset)
    else:
      # If split data exists
      if os.path.isfile("train_val_split.npy"):
        self.dataset = np.load("train_val_split.npy")
      else:
        self.init_dataset()
        np.random.shuffle(self.dataset)
        np.save("train_val_split", self.dataset)

      # Split the data that's created or loaded
      train_dataset, val_dataset = np.split(self.dataset, [int(0.9 * len(self.dataset))])
      if mode == "train":
        self.dataset = train_dataset
      elif mode == "val":
        self.dataset = val_dataset


  def init_dataset(self):
    """
    Helper for create_data_tuples.
    """
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


  def get_groudtruth_poses(self, seq_num_str, frame_num):
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

    # Get x, y, theta from pose matrix which is 3 x 4
    # The first 3 x 3 part is rotaion matrix and the last 3 x 1 is [x, y, z].T
    x_ind = 3
    y_ind = 11
    x = pose[x_ind]
    y = pose[y_ind]
    if np.arcsin(pose[0]) > 0:
      theta = np.arccos(pose[0])
    else:
      theta = np.arccos(pose[0]) * -1

    return x, y, theta


  def get_timestamp(self, seq_num_str, frame_num):
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
    # print(oxts_file_path)

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
    result = []
    for compos_image, state, cur_time in sample:
      result.append((torch.from_numpy(compos_image), torch.from_numpy(state), torch.from_numpy(cur_time)))
    return result


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
  from torch.utils.data import Dataset, DataLoader
  batch_size = 2

  seq_dir = "/mnt/disks/dataset/dataset_post/sequences/"
  poses_dir = "/mnt/disks/dataset/dataset_post/poses/"
  oxts_dir = "/mnt/disks/dataset/dataset_post/oxts/"

  dataset = KittiDatasetSeq(seq_dir, poses_dir, oxts_dir, mode="train")

  dataloader = DataLoader(dataset = dataset, batch_size = batch_size)

  for i, minibatch in enumerate(dataloader):
      print(len(minibatch))
      print(type(minibatch[0]))
      print(minibatch[0][0].shape)
      print(len(minibatch[0][1]))
      break


if __name__ == "__main__":
  main()

