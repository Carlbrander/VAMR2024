import os
import numpy as np
import cv2
from Bootstrapping import bootstrapping
from VO import VisualOdometry
from io import StringIO
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from plot import Plotter
from benchmark import Benchmarker

def parse_arguments():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--ds", type=int, default=0, help="Dataset to use. Options: 0: KITTI, 1: Malaga, 2: parking")
    parser.add_argument("--use_sift", type=bool, default=True, help="Use SIFT instead of Harris")
    args = parser.parse_args()


    #harris detector parameters (from exercise 3)
    args.corner_patch_size = 9
    args.harris_kappa = 0.08
    args.num_keypoints = 1000
    args.nonmaximum_supression_radius = 5
    args.descriptor_radius = 9
    args.match_lambda = 4
    args.use_BA = False
    args.use_pose_refinement = False
    args.max_depth = 100


    args.threshold_angle = 0.1 # only for the start anyway, adapted dynamically
    args.min_baseline = 0.5 # only for the start anyway, adapted dynamically

    return args

def read_poses_kitti(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            values = list(map(float, line.split()))
            R = np.array(values[:9]).reshape(3, 3)
            t = np.array([values[3], values[-1]]).reshape(2, 1)
            poses.append((t))
    return poses

def compute_camera_positions(poses):
    camera_positions = []
    for R, t in poses:
        camera_position = t #-R.T @ t
        camera_positions.append(camera_position.flatten())
    return np.array(camera_positions)

def plot_camera_trajectory(camera_positions, title="Camera Trajectory"):
    fig, ax = plt.subplots()
    num_positions = len(camera_positions)
    colors = plt.cm.viridis(np.linspace(0, 1, num_positions))
    
    # for i in range(num_positions - 1):
    #     ax.plot(camera_positions[i:i+2][0], camera_positions[i:i+2][1], color=colors[i])
    
    camera_x_gt = [point[0] for point in camera_positions]
    camera_z_gt = [point[1] for point in camera_positions]
    ax.plot(camera_x_gt, camera_z_gt, 'k-', label='Ground Truth Trajectory')
    ax.legend()

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(title)
    ax.plot(camera_x_gt[0], camera_z_gt[0], 'go', label="Start")
    ax.plot(camera_x_gt[-1], camera_z_gt[-1], 'ro', label="End")
    ax.set_xlim(np.min(camera_x_gt), np.max(camera_x_gt))
    ax.set_ylim(np.min(camera_z_gt), np.max(camera_z_gt))
    ax.legend()
    plt.draw()
    plt.pause(1)

def dataset_setup(args):

    ds = args.ds

    if ds == 0:
        # need to set kitti_path to folder containing "05" and "poses"
        kitti_path = "data/kitti/"
        assert 'kitti_path' in locals()
        poses = read_poses_kitti(os.path.join(kitti_path, 'poses', '05.txt'))
        last_frame = 4540
        K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                      [0, 7.188560000000e+02, 1.852157000000e+02],
                      [0, 0, 1]])
    elif ds == 1:
        # Path containing the many files of Malaga 7.
        malaga_path = "data/malaga-urban-dataset-extract-07/"
        assert 'malaga_path' in locals()
        images = os.listdir(os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images'))
        left_images = images[2::2]
        #sort images by name
        left_images.sort()
        last_frame = len(left_images)
        K = np.array([[621.18428, 0, 404.0076],
                      [0, 621.18428, 309.05989],
                      [0, 0, 1]])
        args.left_images = left_images
    elif ds == 2:
        # Path containing images, depths and all...
        parking_path = "data/parking"
        assert 'parking_path' in locals()
        last_frame = 598
        with open(os.path.join(parking_path, 'K.txt'), 'r') as file:
            lines = [line.rstrip(',\n') for line in file]

        data_str = '\n'.join(lines)

        K = np.genfromtxt(StringIO(data_str), delimiter=',')

        ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
        ground_truth = ground_truth[:, [-9, -1]]
    else:
        assert False

    # Bootstrap according to instructions from project statement (frame 1 and 3)

    # need to set bootstrap_frames
    if ds == 0:
        bootstrap_frames = [0, 2]
        img0 = cv2.imread(os.path.join(kitti_path, '05', 'image_0', f'{bootstrap_frames[0]:06d}.png'), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(kitti_path, '05', 'image_0', f'{bootstrap_frames[1]:06d}.png'), cv2.IMREAD_GRAYSCALE)
        args.kitti_path = kitti_path
    elif ds == 1:
        bootstrap_frames = [0, 2]
        img0 = cv2.cvtColor(cv2.imread(os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images', left_images[bootstrap_frames[0]])), cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(cv2.imread(os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images', left_images[bootstrap_frames[1]])), cv2.COLOR_BGR2GRAY)
        args.malaga_path = malaga_path
    elif ds == 2:
        bootstrap_frames = [0, 2]
        img0 = cv2.cvtColor(cv2.imread(os.path.join(parking_path, f'images/img_{bootstrap_frames[0]:05d}.png')), cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(cv2.imread(os.path.join(parking_path, f'images/img_{bootstrap_frames[1]:05d}.png')), cv2.COLOR_BGR2GRAY)
        args.parking_path = parking_path
    else:
        assert False

    args.K = K
    args.last_frame = last_frame
    args.bootstrap_frames = bootstrap_frames
    args.gt_Rt = poses
    args.gt_camera_position = poses
    args.img0 = img0
    args.img1 = img1
    plot_camera_trajectory(args.gt_camera_position)

    return args

class History:
    def __init__(self, keypoints, landmarks, R, t):

        self.keypoints = [keypoints]
        self.landmarks = [landmarks]
        self.R = [R]
        self.t = [t]
        #initialize with empty first element
        self.triangulated_landmarks = [np.array([])]
        self.triangulated_keypoints = [np.array([])]
        #initiate hidde state history
        self.Hidden_states = []
        self.camera_position = []
        self.threshold_angles = []
        self.num_keypoints = []

def load_image(ds, i,args):

    if ds == 0:
        image = cv2.imread(os.path.join(args.kitti_path, '05', 'image_0', f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
    elif ds == 1:
        image = cv2.cvtColor(cv2.imread(os.path.join(args.malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images', args.left_images[i])), cv2.COLOR_BGR2GRAY)
    elif ds == 2:
        image = cv2.cvtColor(cv2.imread(os.path.join(args.parking_path, f'images/img_{i:05d}.png')), cv2.COLOR_BGR2GRAY)
    else:
        assert False

    return image

def getScale(gt_camera_position, t):
    scale = -gt_camera_position[1]/t[2]
    return scale

def continuous_operation(keypoints, landmarks, descriptors, R, t, args, history):

    prev_img = args.img1
    vo = VisualOdometry(args)
    benchmarker = Benchmarker(args.gt_camera_position, args.ds)
    plotter = Plotter(args.gt_camera_position, benchmarker.camera_position_bm)

    Hidden_state = []
  
    # Continuous operation
    for i in range(args.bootstrap_frames[1] + 1, args.last_frame + 1):
        # if i < 120:
        #     continue

        print(f'\n\nProcessing frame {i}\n=====================')
        
        image = load_image(args.ds, i, args)


        print(f"-5. keypoints.shape: {keypoints.shape}")

        keypoints, landmarks, descriptors, R, t, Hidden_state, history = vo.process_image(prev_img, image, keypoints, landmarks, descriptors, R, t, Hidden_state, history)
        
        # RMS, is_benchmark = benchmarker.process(history.camera_position, i)
        RMS = 0
        is_benchmark = True
        plotter.visualize_dashboard(history, image, RMS, is_benchmark)
        
        #update previous image
        prev_img = image


if __name__ == "__main__":

    #Parse Arguments
    args = parse_arguments()

    #Dataset Setup
    args = dataset_setup(args)

    #Bootstrapping
    args, keypoints, landmarks, R, t, descriptors = bootstrapping(args)

    # get scale
    scale = getScale(args.gt_camera_position[args.bootstrap_frames[1]], t)
    args.gt_camera_position = args.gt_camera_position*scale

    #Initialize History
    history = History(keypoints, landmarks, R, t)
    

    #Continuous Operation
    continuous_operation(keypoints, landmarks, descriptors, R, t, args, history)



