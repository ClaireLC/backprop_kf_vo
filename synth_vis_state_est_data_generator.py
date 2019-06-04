import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from random import randint
import scipy.linalg as sp_linalg
import time
from torch.utils.data import Dataset


class SynthVisStateEstDataGenerator(Dataset):

    def __init__(self, b_load_data=None, path_to_file=None, num_timesteps=1, num_simulations=1,
                 image_dims=np.array((128, 128)), b_show_figures=False):
        """
        :param b_load_data: if True - will load the data located at path_to_file, if False - generates & saves data to
                            path_to_file, if None, only generates new data
        :param path_to_file: path to where the pickle file will be saved (includes "/name.pkl")
        :param num_timesteps: How many steps each simulation is run for
        :param num_simulations: How many simulations are run
        :param image_dims: row and column pixel ranges for the image
        :param b_show_figures: whether to plot images as they are generated
        """
        self.num_timesteps = num_timesteps  # dimension "T"
        self.num_simulations = num_simulations  # dimension "N"
        self.image_dims = image_dims  # [pix] (# rows, # columns)
        # simulation parameters
        self.dt = 0.1  # [s] simulation time step
        self.pix_per_m = 10  # [pix/m] mapping from image pixels to length
        self.sim_dims = np.flip(image_dims)/self.pix_per_m  # [m] simulation width, height
        self.k = 4  # [N/m] spring constant for dot motion
        self.b = 0.1  # [Ns/m] damping coef. for dot motion
        self.m = 1  # [kg] dot mass
        self.dot_radius = int(2)  # [pix] radius of tracked dot
        self.dot_color = (255, 0, 0)  # [R B G] color of tracked dot
        # States are [x_position, y_position, x_velocity, y_velocity]
        self.x0_min = float(-self.sim_dims[0]/3)
        self.x0_max = float(self.sim_dims[0]/3)
        self.y0_min = float(-self.sim_dims[1]/3)
        self.y0_max = float(self.sim_dims[1]/3)
        self.vel_std = 6
        self.vel_mean = 0
        self.num_obstacles_range = (10, 99)  # min and max number of obstacles
        self.obs_v_ave = 0.25  # [m/s] - mean velocity of occluding obstacle velocities
        self.obs_v_std = 0.4  # standard deviation of occluding obstacle velocities
        self.obs_rad_ave = self.dot_radius  # [pix] average obstacle radius
        self.obs_rad_std = 0.75  # standard deviation of obstacle radius
        self.obs_min_rad = 2  # [pixels] minimum obstacle radius

        self.fig = None
        self.ax = None
        self.image = None
        self.b_load_data = b_load_data

        self.b_show_figures = b_show_figures  # DEBUG - this should equal b_show_figures
        self.start_time = time.time()
        if b_load_data is not None and b_load_data and path_to_file is not None:
            # Load the data
            self.data = self.load_data(path_to_file)
            if b_show_figures:
                # Visualize the loaded data
                self.visualize_all_images()
        else:
            # delete existing .pkl if applicable
            if self.b_load_data is not None and not self.b_load_data and os.path.exists(path_to_file):
                print("File already exists... overwriting now")
                os.remove(path_to_file)
            # Generate new data
            self.path_to_file = path_to_file
            self.data = self.generate_data()
            print("Data generated! ({}s elapsed)".format(int(time.time() - self.start_time)))

    def __len__(self):
        """
        :return: number of data samples
        """
        return self.num_simulations

    def __getitem__(self, idx):
        """
        :return: data[idx]  -- defines the behavior of the [] operator on the dataset
        """
        # returns a list of tuples for sequence number idx
        print(type(self.data[idx][0][0]))
        return self.data[idx]

    def load_data(self, fn):
        """
        ref for incremental loading/ saving of the pickle file:
            https://stackoverflow.com/questions/26394768/pickle-file-too-large-to-load
        """
        print("Loading data...")
        i = 0
        data = {}
        with open("{}".format(fn), 'rb') as f:
            try:
                while True:
                    loaded_data = pickle.load(f)
                    for key in loaded_data.keys():
                        data[i] = loaded_data[key]
                        i += 1
                        if i > 0 and i % 100 == 0:
                            print("Loaded {} simulation iterations so far, {}s elapsed"
                                  .format(i, int(time.time() - self.start_time)))
            except EOFError:
                pass
        self.num_simulations = len(data.keys())
        print("Data loaded! ({}s elapsed)".format(int(time.time() - self.start_time)))
        return data

    def save_data(self, fn, data):
        # save self.data and self.num_simulations (update incrementally)
        print("Saving data... ", end="")
        with open("{}".format(fn), 'ab') as f:  # Note: 'ab' appends the data
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print("Data saved!")

    def generate_data(self):
        """
        ref for incremental loading/ saving of the pickle file:
            https://stackoverflow.com/questions/26394768/pickle-file-too-large-to-load
        :return: data -- dictionary of list of tuples, each is (image, dot_position, dot_velocity)
        """
        data = dict()
        for sim_itr in range(self.num_simulations):
            if sim_itr % 50 == 0 and self.num_simulations > 50:
                print("Running simulation {} / {} ({}s elapsed)".format(sim_itr, self.num_simulations,
                                                                        int(time.time() - self.start_time)))
            init_state = self.init_conditions()
            dot_state = init_state
            obstacles = self.init_obstacles()
            cum_t = 0
            data[sim_itr] = []
            for t_itr in range(self.num_timesteps):
                # print("itr {}".format(itr))
                img = self.fill_in_img(dot_state, obstacles)  # draw our target & obstacles on img
                if self.b_show_figures:
                    self.visualize_image(img, cum_t)
                data[sim_itr].append((img, dot_state))
                cum_t = cum_t + self.dt
                dot_state = self.get_dot_state(cum_t, init_state)
                obstacles = self.update_obstacles(obstacles)

            if (self.b_load_data is not None and not self.b_load_data) and \
                    (sim_itr == self.num_simulations - 1 or (sim_itr > 0 and sim_itr % 100 == 0)):
                # Save the data every N iterations
                self.save_data(self.path_to_file, data)
                del data  # delete existing data
                data = dict()  # reinitialize data to be empty

        return data

    def init_conditions(self):
        x0 = np.random.uniform(low=self.x0_min, high=self.x0_max, size=None)
        y0 = np.random.uniform(low=self.y0_min, high=self.y0_max, size=None)
        vx0 = self.vel_std*np.random.randn()+self.vel_mean
        vy0 = self.vel_std*np.random.randn()+self.vel_mean
        return np.array((x0, y0, vx0, vy0))

    def init_obstacles(self):
        obstacles = []
        num_obstacles = np.random.randint(*self.num_obstacles_range)
        for i in range(num_obstacles):
            obstacles.append(self.create_obstacle())  # [m, m/s, pix, RGB]
        return obstacles

    def create_obstacle(self):
        col = (randint(0, 255), randint(0, 255), randint(0, 255))
        ob_state = np.concatenate((self.rc_to_xy(np.array([randint(0, self.image_dims[0]),
                                                           randint(0, self.image_dims[1])])),
                                   np.random.normal(self.obs_v_ave, self.obs_v_std, 2)))
        radius = max(self.obs_min_rad, np.round(np.random.normal(self.obs_rad_ave, self.obs_rad_std)).astype(np.int))  # [pix]
        return ob_state, radius, col  # [[m, m/s], pix, RGB]

    def fill_in_img(self, dot_state, obstacles):
        img = np.zeros((self.image_dims[0], self.image_dims[1], 3)).astype(np.uint8)  # all black image
        img = self.draw_circle(dot_state[0:2], self.dot_radius, self.dot_color, img)
        for obs in obstacles:
            img = self.draw_circle(obs[0][0:2], obs[1], obs[2], img)
        return img

    def draw_circle(self, xy, rad, color, img):
        """
        :param xy: the (x,y) position of the circle [m]
        :param rad: the radius of the circle [pixels]
        :param color: (R, B, G)
        :param img: image to be drawn on
        :return: img - drawn on image
        """
        rc = self.xy_to_rc(xy)  # [pix] convert to pixels
        for dr in range(-rad, rad + 1):
            r1 = (rad**2 - dr**2)**0.5
            r2 = (rad ** 2 - (dr - np.sign(dr)) ** 2) ** 0.5
            r_span = np.round((r1 + r2)/2).astype(np.int)
            for dc in range(-r_span, r_span + 1):
                row = rc[0] + dr
                col = rc[1] + dc
                if row < 0 or row >= img.shape[0] or col < 0 or col >= img.shape[1]:
                    continue  # truncate the drawing of the obstacle if it's partially offscreen
                else:
                    img[row, col, :] = color
        return img

    def get_dot_state(self, t, init_state):
        """
        :param t: time at which state is being queried. State is [position, velocity] (np.array of length 4)
        :return: new_state = [position, velocity]
        """
        A = np.zeros((4, 4))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 0:2] = -self.k/self.m*np.eye(2)
        A[2:4, 2:4] = -self.b/self.m*np.eye(2)
        new_state = sp_linalg.expm(A*t)@init_state
        return new_state

    def update_obstacles(self, obstacles):
        for i, obs in enumerate(obstacles):
            new_position = obs[0][0:2] + obs[0][2:4]*self.dt
            if self.b_is_pos_in_view(new_position):
                obstacles[i][0][0] = new_position[0]
                obstacles[i][0][1] = new_position[1]
            else:
                # obstacle has left the image, overwrite with new obstacle
                obstacles[i] = self.create_obstacle()
        return obstacles

    def b_is_pos_in_view(self, pos):
        return -self.sim_dims[0] / 2 <= pos[0] <= self.sim_dims[0] / 2 and \
               -self.sim_dims[1] / 2 <= pos[1] <= self.sim_dims[1] / 2

    def visualize_image(self, img, t=-1):
        if t >= 0:
            plt.title("Time = {:.2f} seconds".format(t))

        if self.fig is None:
            self.fig = plt.figure(1)
            self.image = plt.imshow(img)
            plt.show(block=False)
        else:
            self.image.set_data(img)
            self.fig.canvas.draw()

        time.sleep(self.dt/5)

    def visualize_all_images(self):
        # loop through the loaded data to display the images
        for n_itr in range(self.num_simulations):
            one_sim_data = self.data[n_itr]
            for t_itr, (image, state) in enumerate(one_sim_data):
                self.visualize_image(image, t_itr * self.dt)

    def rc_to_xy(self, rc):
        """
        :param rc: the row and column of a pixel (dim is 2)
        :return: the x and y coordinates in meters (dim is 2)
        """
        x = (rc[1] - self.image_dims[1]/2)/self.pix_per_m
        y = (-rc[0] + self.image_dims[0]/2)/self.pix_per_m
        return np.array([x, y])

    def xy_to_rc(self, xy):
        """
        :param xy_m: the x and y coordinates in meters (dim is 2)
        :return: rc - the row and column of a pixel (dim is 2)
        """
        row = np.round((-xy[1] + self.sim_dims[1]/2)*self.pix_per_m)
        col = np.round((xy[0] + self.sim_dims[0]/2)*self.pix_per_m)
        return np.array([row, col]).astype(np.int)

    def find_mean_pixel_location_by_color(self, img, color_to_find=np.array([255, 0, 0])):
        return np.vstack(np.where(np.all(img == color_to_find, axis=2))).T.mean(axis=0)


def str2bool(v):
    """
    Allows the argparser to accept "true/false" "True/False" "t/f" "1/0" etc for bool values
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    """
    Calling this script with no inputs generates data and does not save nor load any data. If this is desired, the first
    argument is a path to the file (i.e. ~/path/my_filename.pkl) and the second argument is either True or False for 
    loading data from that file or writing data to that file respectively.
    
    To display paramter info, type the following on the command line:
      $ python3 synth_vis_state_est_data_generator.py -h
    
    Example to load data from dataset1.pkl:
      $ python3 synth_vis_state_est_data_generator.py -L 
          ... --path_to_file /home/adam/Documents/backprop_kf/datasets/dataset1.pkl
          
    Example to generate and save data to dataset2.pkl:
      $ python3 synth_vis_state_est_data_generator.py -S --path_to_file 
          ... --path_to_file /home/adam/Documents/backprop_kf/datasets/dataset1.pkl -N 2000 -T 30
    """
    parser = argparse.ArgumentParser(description='Process my inputs.')
    parser.add_argument('-L', '--b_load_data', type=str2bool, nargs='?', const=True, required=False, default=None,
                        help='a boolean for if data should be loaded (empty or false if not)')
    parser.add_argument('-S', '--b_save_data', type=str2bool, nargs='?', const=True, required=False, default=None,
                        help='a boolean for if data should be saved (empty or false if not')
    parser.add_argument('-p', '--path_to_file', type=str, required=False,
                        help='path to load/save .pkl file')
    parser.add_argument('-T', '--num_timesteps', type=int, required=False, default=20,
                        help='number of timestamps for which each simulation rollout is run')
    parser.add_argument('-N', '--num_simulations', type=int, required=False, default=1,
                        help='number of simulations to run')
    parser.add_argument('-iw', type=int, required=False, default=128,
                        help='width of images to generate in pixels')
    parser.add_argument('-ih', type=int, required=False, default=128,
                        help='height of images to generate in pixels')
    parser.add_argument('-bsf', '--b_show_figures', type=str2bool, nargs='?', const=True, required=False, default=False,
                        help='a boolean for if the generated images should be rendered')
    args = parser.parse_args()

    if args.b_load_data and args.b_save_data:
        raise RuntimeError("Can only load OR save, only use 1 of these two flags")

    if args.b_load_data is not None and args.b_load_data:
        load_or_save = True
    elif args.b_save_data is not None and args.b_save_data:
        load_or_save = False
    else:
        load_or_save = None

    my_dataset = SynthVisStateEstDataGenerator(b_load_data=load_or_save,
                                               path_to_file=args.path_to_file,
                                               num_timesteps=args.num_timesteps,
                                               num_simulations=args.num_simulations,
                                               b_show_figures=args.b_show_figures,
                                               image_dims=np.array([args.ih, args.iw]))
    print("Done with test data loader")

