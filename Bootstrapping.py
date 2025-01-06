import numpy as np
import cv2
import matplotlib.pyplot as plt

#solution scripts from exercise 3 for feature detection and matching using shi-tomasi
from bootstrapping_utils.exercise_3.harris import harris
from bootstrapping_utils.exercise_3.select_keypoints import selectKeypoints
from bootstrapping_utils.exercise_3.describe_keypoints import describeKeypoints
from bootstrapping_utils.exercise_3.match_descriptors import matchDescriptors

def plot_3d_bootstrapping(points_3d, R, t):

    # Visualize in 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #add camera locations
    ax.scatter(0, 0, 0, c='g', marker='o')
    ax.scatter(t[0], t[1], t[2], c='r', marker='o')

    
    #set min max for all axis to be the same (-50,50)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(0, 50)

    #set elevation and azimuth
    ax.view_init(elev=0, azim=-90)


    plt.show()

    cv2.destroyAllWindows()

def plot_2d_bootstrapping(img0, img1, keypoints_0_inliers, keypoints_1_inliers, title):

    #visualize inlier keypoints on the two images
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.imshow(img0)
    plt.scatter(keypoints_0_inliers[:, 0], keypoints_0_inliers[:, 1], c='r', s=10)
    plt.title(title + ' in Image 0')

    plt.subplot(122)
    plt.imshow(img1)
    plt.scatter(keypoints_1_inliers[:, 0], keypoints_1_inliers[:, 1], c='r', s=10)
    plt.title(title + ' in Image 1')
    plt.show()

    cv2.destroyAllWindows()



def bootstrapping(args):
    '''
    This function takes the first two images and returns the keypoints and landmarks using
    the 8-point algorithm and patch matching between the two images.
    It then uses RANSAC to remove outliers.

    Input:
    img0 - The first image
    img1 - The second image (actually the third one from the dataset but it's just the next one with sufficient distance)
    K - The intrinsic matrix

    Output:
    keypoints - The keypoints in the first image
    landmarks - The 3D landmarks
    '''

    img0                = args.img0
    img1                = args.img1
    K                   = args.K


    if not args.use_sift:
        #get keypoints of both images using 
        harris_scores_0 = harris(img0, args.corner_patch_size, args.harris_kappa) #returns [H,W] array of harris scores
        keypoints_0 = selectKeypoints(harris_scores_0, 2*args.num_keypoints, args.nonmaximum_supression_radius) #returns [2,N] array of keypoints
        descriptors_0 = describeKeypoints(img0, keypoints_0, args.descriptor_radius) #returns [(2*r+1)**2,N] array of descriptors

        harris_scores_1 = harris(img1, args.corner_patch_size, args.harris_kappa) #returns [H,W] array of harris scores
        keypoints_1 = selectKeypoints(harris_scores_1, 2*args.num_keypoints, args.nonmaximum_supression_radius) #returns [2,N] array of keypoints
        descriptors_1 = describeKeypoints(img1, keypoints_1, args.descriptor_radius) #returns [(2*r+1)**2,N] array of descriptors

        matches = matchDescriptors(descriptors_1, descriptors_0, args.match_lambda)

        #get matched keypoints from matches
        matched_keypoints_1 = keypoints_1.T[matches != -1]
        matched_keypoints_0 = keypoints_0.T[matches[matches != -1]]

        #get matched descriptors from matches
        matched_descriptors_1 = descriptors_1[:, matches != -1]

        # Swap coordinates from (row, col) to (x, y)
        matched_keypoints_0_xy = matched_keypoints_0[:, ::-1]
        matched_keypoints_1_xy = matched_keypoints_1[:, ::-1]

    if args.use_sift:
        sift = cv2.SIFT_create()
        keypoints_0, descriptors_0 = sift.detectAndCompute(img0, None)
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)

  
        # Match descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_0, descriptors_1, k=2)

        

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # Get matched keypoints
        matched_keypoints_0_xy = np.array([keypoints_0[match.queryIdx].pt for match in good])
        matched_keypoints_1_xy = np.array([keypoints_1[match.trainIdx].pt for match in good])

        # Get matched descriptors
        matched_descriptors_1 = np.array([descriptors_1[match.trainIdx] for match in good]).T

       
  

    #Estimate the essential matrix E using the 8-point algorithm with strict threshold of 1 pixel and 99.9% confidence
    # Set the random seed for reproducibility
    np.random.seed(42)
    
    E, inliers = cv2.findEssentialMat(matched_keypoints_0_xy, matched_keypoints_1_xy, K, cv2.RANSAC, 0.999, 1.0)

    # Recover the pose of the second camera
    # Use inliers mask to filter the matched keypoints
    matched_keypoints_0_xy = matched_keypoints_0_xy[inliers.ravel() == 1]
    matched_keypoints_1_xy = matched_keypoints_1_xy[inliers.ravel() == 1]
    matched_descriptors_1 = matched_descriptors_1[:, inliers.ravel() == 1]

    # Recover the pose of the second camera using inlier points
    _, R, t, mask_pose = cv2.recoverPose(E, matched_keypoints_0_xy, matched_keypoints_1_xy, K)

    # Select inlier points
    inliers = mask_pose.ravel().astype(bool)
    keypoints_0_inliers = matched_keypoints_0_xy[inliers]
    keypoints_1_inliers = matched_keypoints_1_xy[inliers]

    # Select desctiptors of inlier points
    descriptors_1_inliers = matched_descriptors_1[:, inliers]

    # Projection matrices for the two cameras
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))

    # Triangulate points
    points_4d_hom = cv2.triangulatePoints(P0, P1, keypoints_0_inliers.T, keypoints_1_inliers.T)
    points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T

  
    # Visualize 3D points
    #plot_3d_bootstrapping(points_3d, R, t)
    ## Visualize inlier keypoints
    #plot_2d_bootstrapping(img0, img1, keypoints_0_inliers, keypoints_1_inliers, "Inlier Keypoints")

    
    #transpose keypoints_1_inliers to match the format of the keypoints expected in continuous operation
    return args, keypoints_1_inliers.T, points_3d.T, R, t, descriptors_1_inliers

    


