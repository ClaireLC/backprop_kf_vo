import numpy as np
import sys
import os
import warnings

from skimage import data, color, io, img_as_ubyte
from skimage.transform import rescale, resize, downscale_local_mean

path_to_kitti = "/home/clairechen/KITTI/dataset/sequences/"
save_path = "/home/clairechen/KITTI/dataset_post/sequences/"

if not os.path.exists(save_path):
  os.makedirs(save_path)

seq_num = sys.argv[1]
cam_name = sys.argv[2]
img_name = sys.argv[3]
prev_img_name = sys.argv[4]

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

