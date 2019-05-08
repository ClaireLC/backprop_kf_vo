import os
import time
import numpy as np
import sys
import warnings

from skimage import data, color, io, img_as_ubyte
from skimage.transform import rescale, resize, downscale_local_mean

def preprocess_img(seq_name, cam_name, img_name, prev_img_name):
  path_to_kitti = "/home/clairechen/KITTI/dataset/sequences/"
  save_path = "/home/clairechen/KITTI/dataset_post/sequences/"
  
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  
  seq_num = seq_name
  cam_name = cam_name
  img_name = img_name
  prev_img_name = prev_img_name
  
  curr_file_path = path_to_kitti + seq_num + "/" + cam_name + "/" + img_name
  prev_file_path = path_to_kitti + seq_num + "/" + cam_name + "/" + prev_img_name
  new_img_dir = save_path + seq_num + "/" + cam_name + "/current/"
  diff_img_dir = save_path + seq_num + "/" + cam_name + "/diff/"
  
  if not os.path.exists(new_img_dir):
    os.makedirs(new_img_dir)
  
  if not os.path.exists(diff_img_dir):
    os.makedirs(diff_img_dir)
  
  #print("curr file path: {}".format(curr_file_path))
  #print("prev file path: {}".format(prev_file_path))
  
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
  
  # Crop image to a 3:1 aspect ratio
  cropped = image[:,56:1184]
  prev_cropped = prev_image[:,56:1184]
  
  # Downsample cropped image to 150x50 (input size reported in BKF paper)
  resized = resize(cropped, (50, 150),anti_aliasing=True)
  prev_resized = resize(prev_cropped, (50, 150),anti_aliasing=True)
  
  # Get difference image
  diff_im = image - prev_image
  diff_resized = resize(diff_im, (50, 150),anti_aliasing=True)
  
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    io.imsave(new_img_dir + img_name, img_as_ubyte(resized))
    io.imsave(diff_img_dir + img_name, img_as_ubyte(diff_resized))


def main():
  start_img_num = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1101,
    9: 1,
    10: 1
    }
  
  last_img_num = {
    0: 4540,
    1: 1100,
    2: 4660,
    3: 800,
    4: 270,
    5: 2760,
    6: 1100,
    7: 1100,
    8: 5170,
    9: 1590,
    10: 1200
    }
  
  num_digits = 6
  
  seq_num = 3
  
  cam_name = "image_2"
  
  for cam_name in ["image_2", "image_3"]:
    for curr_im_num in range(start_img_num[seq_num], last_img_num[seq_num] + 1):
      curr_im_name = str(curr_im_num).zfill(num_digits) + ".png"
      prev_im_name = str(curr_im_num - 1).zfill(num_digits) + ".png"
      seq_num_str = str(seq_num).zfill(2)
      
      args = "args: " + seq_num_str + " " + cam_name + " " + curr_im_name + " " + prev_im_name
      print(args)
      preprocess_img(seq_num_str, cam_name, curr_im_name, prev_im_name)

if __name__ == "__main__":
  main()
