import os
import time
import numpy as np
import sys
import warnings
import argparse

from skimage import data, color, io, img_as_ubyte
from skimage.transform import rescale, resize, downscale_local_mean

def preprocess_img(path_to_traj, save_path, img_name, prev_img_name):
  
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  
  curr_file_path = path_to_traj + "/frame" + img_name
  prev_file_path = path_to_traj + "/frame" + prev_img_name
  new_img_dir = save_path + "/current/"
  diff_img_dir = save_path + "/diff/"
  
  if not os.path.exists(new_img_dir):
    os.makedirs(new_img_dir)
  
  if not os.path.exists(diff_img_dir):
    os.makedirs(diff_img_dir)
  
  print("curr file path: {}".format(curr_file_path))
  print("prev file path: {}".format(prev_file_path))
  
  # Open current and previous images
  if os.path.isfile(prev_file_path):
    prev_image = io.imread(prev_file_path)
  else:
    print("ERROR: incorrect previous image file path")
    quit()
  
  if os.path.isfile(curr_file_path):
    image = io.imread(curr_file_path)
  else:
    print("ERROR: incorrect image file path")
    quit()
  
  # Downsample cropped image to 150x50 (input size reported in BKF paper)
  resized = resize(image, (50, 150),anti_aliasing=True)
  prev_resized = resize(prev_image, (50, 150),anti_aliasing=True)
  
  # Get difference image
  diff_im = image - prev_image
  diff_resized = resize(diff_im, (50, 150),anti_aliasing=True)
  
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    io.imsave(new_img_dir + img_name, img_as_ubyte(resized))
    io.imsave(diff_img_dir + img_name, img_as_ubyte(diff_resized))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--traj_num", help="Number of trajectory to preprocess")
  args = parser.parse_args()
  traj_num = args.traj_num

  # get paths to specified trajectory
  path_to_traj = "/home/clairechen/test_traj_" + traj_num.zfill(1) + "/frames"
  save_path = "/home/clairechen/test_traj_" + traj_num.zfill(1) + "/frames_post"

  # get the number of frames in this trajectory
  total_frame_num = len(os.listdir(path_to_traj))

  for i in range(1, total_frame_num):
    curr_im_name = str(i).zfill(4) + ".jpg"
    prev_im_name = str(i-1).zfill(4) + ".jpg"
    preprocess_img(path_to_traj, save_path, curr_im_name, prev_im_name)

if __name__ == "__main__":
  main()
