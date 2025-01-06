import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import plotly.graph_objects as go



#solution scripts from exercise 3 for feature detection and matching using shi-tomasi
from bootstrapping_utils.exercise_3.harris import harris
from bootstrapping_utils.exercise_3.select_keypoints import selectKeypoints
from bootstrapping_utils.exercise_3.describe_keypoints import describeKeypoints
from bootstrapping_utils.exercise_3.match_descriptors import matchDescriptors


np.random.seed(1)


class VisualOdometry:

    def __init__(self, args):
        """
        Initialize the Visual Odometry system with basic components
        """
        # Tuning parameters
        self.threshold_angle = args.threshold_angle
        self.nonmaximum_suppression_radius = args.nonmaximum_supression_radius

        self.RANSAC_iterationsCount = args.RANSAC_iterationsCount
        self.RANSAC_reprojectionError = args.RANSAC_reprojectionError
        self.RANSAC_confidence = args.RANSAC_confidence
        self.KLT_winSize = args.KLT_window_size
        self.KLT_max_level = args.KLT_max_level
        self.KLT_criteria = args.KLT_criteria

        self.num_keypoints = args.max_num_new_keypoints

        # Propagate arguments
        self.corner_patch_size = args.corner_patch_size
        self.harris_kappa = args.harris_kappa
        self.descriptor_radius = args.descriptor_radius
        self.match_lambda = args.match_lambda
        self.K = args.K

        self.use_sift = args.use_sift

        #internal image counter
        self.current_image_counter = 0

    def detect_keypoints(self, image, num_keypoints=None, nonmaximum_suppression_radius=None):
        """
        Detect keypoints and their descriptors in the given frame
        
        Args:
            image (numpy.ndarray): Input image for this iteration
        
        Returns:
            tuple: Keypoints and their descriptors
        """

        if num_keypoints == None:
            num_keypoints = self.num_keypoints

        if nonmaximum_suppression_radius == None:
            nonmaximum_suppression_radius = self.nonmaximum_suppression_radius


        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if not self.use_sift:
            harris_scores = harris(gray, self.corner_patch_size, self.harris_kappa)
            keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_suppression_radius)
            descriptors = describeKeypoints(gray, keypoints, self.descriptor_radius)

        if self.use_sift:
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)

           
            # Select N keypoints randomly
            if len(keypoints) > num_keypoints:
                keypoints = list(keypoints)  # Convert to list if it's a tuple
                descriptors = list(descriptors)  # Ensure descriptors are also mutable
                combined = list(zip(keypoints, descriptors))
                np.random.shuffle(combined)
                keypoints, descriptors = zip(*combined)  # Convert back to tuples if needed
            keypoints = keypoints[:num_keypoints]
            descriptors = descriptors[:num_keypoints]

            # Convert keypoints to numpy array
            keypoints = np.array([keypoint.pt for keypoint in keypoints]).T

            # Convert descriptors to numpy array
            descriptors = np.array(descriptors)

            descriptors = descriptors.T

            #switcch x and y coordinates of the keypoints
            keypoints = keypoints[[1, 0], :]

        return keypoints, descriptors
            
    def estimate_motion(self, keypoints_1, landmarks_1):
        """
        Estimate the motion between two frames

        Args:
            keypoints_1 (numpy.ndarray): Keypoints from this frame
            landmarks_1 (numpy.ndarray): Landmarks from this frame

        Returns:

        """
        # Estimate motion using PnP
        retval, rotation_vector, translation_vector, inliers = \
            cv2.solvePnPRansac(
                    landmarks_1.astype(np.float32).T, 
                    keypoints_1.astype(np.float32).T, 
                    self.K, 
                    distCoeffs=None,
                    iterationsCount = self.RANSAC_iterationsCount,
                    reprojectionError = self.RANSAC_reprojectionError,
                    confidence = self.RANSAC_confidence)
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        #non homogenoius transformation
        R_1 = rotation_matrix
        t_1 = translation_vector


        return R_1, t_1, inliers

    def triangulate_landmark(self, keypoint_1, keypoint_2, R_1, t_1, R_2, t_2):
        """
        Triangulate new landmarks from keypoints in two frames
        
        Args:
            keypoints_1 (numpy.ndarray): Keypoints from frame 1
            keypoints_2 (numpy.ndarray): Keypoints from frame 2
            R1 (numpy.ndarray): Rotation matrix of frame 1
            t1 (numpy.ndarray): Translation vector of frame 1
            R2 (numpy.ndarray): Rotation matrix of frame 2
            t2 (numpy.ndarray): Translation vector of frame 2
        
        Returns:
            numpy.ndarray: Triangulated landmarks
        """

        #I checked and the keypoint pairs between 1 and 2 are correct.

        assert t_1.shape == (3, 1)
        assert t_2.shape == (3, 1)
        assert R_1.shape == (3, 3)
        assert R_2.shape == (3, 3)
       
        # Pose Camera when keypoint was first observed
        P1 = self.K @ np.hstack((R_1,  t_1))
        # Current Pose Camera where keypoint is observed (and got tracked to)
        P2 = self.K @ np.hstack((R_2,  t_2))


        # Triangulate points
        landmark_homogenious = cv2.triangulatePoints(P1, P2, keypoint_1, keypoint_2)

        #transform landmarks from homogenious to euclidean coordinates
        landmark = landmark_homogenious / landmark_homogenious[3, :]
        landmark_final = landmark[:3, :]

        return landmark_final

    def track_keypoints(self, prev_image, image, keypoints_0):
        """
        Track keypoints between two frames

        Args:
            prev_image (numpy.ndarray): Previous image
            image (numpy.ndarray): Current image
            keypoints_0 (list): Keypoints from last image
            descriptors_0 (numpy.ndarray): Descriptors from last image
        
        Returns:
            list: Tracked keypoints_1
            list: Tracked landmarks_1
            list: Tracked descriptors_1
        """
     
        if len(prev_image.shape) > 2:
                prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) > 2:    
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #use KLT to track existing keypoint in this new image
        keypoints_1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_image,
        image,
        keypoints_0.T.astype(np.float32),
        None,
        winSize=(self.KLT_winSize, self.KLT_winSize),
        maxLevel=self.KLT_max_level,
        criteria=self.KLT_criteria
        )
        
        # Select good points
        st_here = st.reshape(-1)
        #get new keypoints
        
        keypoints_1 = keypoints_1.T[:,st_here == 1]
        
        return keypoints_1, st
    
    def feature_matching(self, descriptors_0, descriptors_1):

        # Match descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_0, descriptors_1, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        return good

    def track_and_update_hidden_state(self, Hidden_state):

        # Hidden_State is a list of arrays: [original keypoints, original R, original t, tracked keypoints]

        new_Hidden_state = []
        #check if Hidden_state is not just an emtpy list od lists
        if Hidden_state:

            # track the keypoints in the Hidden state
            tracked_hidden_features, st = self.track_keypoints(self.prev_image, self.image, Hidden_state[3])
            st = st.flatten()

            # delete the keypoints that were not tracked
            new_Hidden_state = [Hidden_state[0][:, st == 1], Hidden_state[1][:,:, st == 1], Hidden_state[2][:, :, st == 1], tracked_hidden_features, Hidden_state[4][:, st == 1]]

            Hidden_state = [Hidden_state[0][:, st == 1], Hidden_state[1][:,:, st == 1], Hidden_state[2][:, :, st == 1], Hidden_state[3][:, st == 1], Hidden_state[4][:, st == 1]]

            # # filter new hidden keypoints via fundamental matrix RANSAC
            # inliers_old ,inliers_new, mask = self.filter_keypoints_via_fundamental_ransac(
            #     Hidden_state[3], 
            #     new_Hidden_state[3],
            #     ransacReprojThreshold=0.5,
            #     confidence=0.99,
            #     maxIters=2000
            # )

            # # Update the Hidden state with the filtered keypoints
            # new_Hidden_state = [new_Hidden_state[0][:, mask], new_Hidden_state[1][:,:, mask], new_Hidden_state[2][:, :, mask], new_Hidden_state[3][:, mask], new_Hidden_state[4][:, mask]]
            
        

        return new_Hidden_state

    def triangulate_new_landmarks(self, Hidden_state, R_1, t_1):
        new_keypoints = []
        new_descriptors = []
        new_landmarks = []
        triangulated_idx = []
        
        if Hidden_state:
            for i in range(Hidden_state[0].shape[1]):

                # Skip triangulation if baseline is 0
                if np.linalg.norm(t_1 - Hidden_state[2][:, :, i]) == 0:
                    continue

                # Triangulate the new landmark
                landmark = self.triangulate_landmark(
                    Hidden_state[0][:, i], Hidden_state[3][:, i], Hidden_state[1][:, :, i], Hidden_state[2][:, :, i], R_1, t_1
                )

                # Calculate the angle between the two camera poses
                angle = self.calculate_angle(landmark, R_1, t_1, Hidden_state[1][:,:,i], Hidden_state[2][:, :, i])

                # Check if the angle is over the threshold
                if angle > self.threshold_angle:
                    new_keypoints.append(Hidden_state[3][:, i])
                    new_descriptors.append(Hidden_state[4][:, i])
                    new_landmarks.append(landmark.flatten())
                    triangulated_idx.append(i)


        # Convert lists to numpy arrays
        new_keypoints = np.array(new_keypoints).T
        new_landmarks = np.array(new_landmarks).T
        new_descriptors = np.array(new_descriptors).T

        return new_keypoints, new_landmarks, new_descriptors, triangulated_idx

    def calculate_angle(self, landmark,R_1, t_1, R_2, t_2):
        #get the direction vector from the first observation camera pose to the landmark
        direction_1 = landmark + R_1.T @ t_1
        #get the direction vector from the current camera pose to the landmark
        direction_2 = landmark + R_2.T @ t_2

        #normalize the vectors
        direction_1 = direction_1 / np.linalg.norm(direction_1)
        direction_2 = direction_2 / np.linalg.norm(direction_2)

        #get the angle between the two vectors
        angle = angle = np.arccos(np.dot(direction_1.flatten(), direction_2.flatten()))

        return angle

    def NMS_on_keypoints(self, new_keypoints, old_keypoints, radius):
        removal_index = set()

        # Calculate distances between new keypoints and old keypoints
        dist_old = np.linalg.norm(new_keypoints[:, :, np.newaxis] - old_keypoints[:, np.newaxis, :], axis=0)
        close_old = np.any(dist_old < radius, axis=1)
        removal_index.update(np.where(close_old)[0])

        # Calculate distances between new keypoints themselves
        dist_new = np.linalg.norm(new_keypoints[:, :, np.newaxis] - new_keypoints[:, np.newaxis, :], axis=0)
        np.fill_diagonal(dist_new, np.inf)  # Ignore self-distances
        close_new = np.any(dist_new < radius, axis=1)
        removal_index.update(np.where(close_new)[0])

        return list(removal_index)

    def add_new_landmarks(self, keypoints_1, landmarks_1, descriptors_1, R_1, t_1, Hidden_state, history):

        ### Detect new Keypoints ###
        new_keypoints, new_descriptors = self.detect_keypoints(self.image, self.num_keypoints, self.nonmaximum_suppression_radius)
        # switch rows and columns to get the correct format
        new_keypoints = new_keypoints[[1, 0], :]


        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image

        history.texts.append(f"Number of Keypoints allowed to detect: {self.num_keypoints}")
        history.texts.append(f"-3 Number of freshly Detected Keypoints: {new_keypoints.shape[1]}")
        print("Number of Keypoints allowed to detect:", self.num_keypoints)
        print("-3 Number of freshly Detected Keypoints:", new_keypoints.shape[1])


        # NMS on new keypoints with landmarks and hidden features

        if Hidden_state:
            keypoints_hidden =  Hidden_state[3]
            total_keypoints = np.concatenate((keypoints_1, keypoints_hidden), axis=1)
        else:
            total_keypoints = keypoints_1

        # NMS on new keypoints with old keypoints
        removal_index = self.NMS_on_keypoints(new_keypoints, total_keypoints, radius = self.nonmaximum_suppression_radius)

        new_keypoints = np.delete(new_keypoints, removal_index, axis=1)
        new_descriptors = np.delete(new_descriptors, removal_index, axis=1)
        print("-2 Number of Keypoints after NMS:", new_keypoints.shape[1])
        history.texts.append(f"-2 Number of Keypoints after NMS: {new_keypoints.shape[1]}")



        # Add new keypoints to the Hidden_state
        if not Hidden_state:
            R_matrices = np.repeat(R_1[:,:,np.newaxis], new_keypoints.shape[1], axis=2)
            t_vectors = np.repeat(t_1[:,:,np.newaxis], new_keypoints.shape[1], axis=2)
            Hidden_state = [new_keypoints, R_matrices, t_vectors, new_keypoints, new_descriptors]
        else:
            R_matrices = np.repeat(R_1[:,:,np.newaxis], new_keypoints.shape[1], axis=2)
            t_vectors = np.repeat(t_1[:,:,np.newaxis], new_keypoints.shape[1], axis=2)
            # Concatenate new keypoints to the Hidden_state
            Hidden_state = [np.concatenate((Hidden_state[0], new_keypoints), axis=1), 
                            np.concatenate((Hidden_state[1], R_matrices), axis=2), 
                            np.concatenate((Hidden_state[2], t_vectors), axis=2), 
                            np.concatenate((Hidden_state[3], new_keypoints), axis=1),
                            np.concatenate((Hidden_state[4], new_descriptors), axis=1)]
        
        # print cummulative number of keypoints in hidden state after adding new keypoints
        sum_hidden_state_landmarks = 0
        if Hidden_state:
            sum_hidden_state_landmarks = Hidden_state[0].shape[1]
        else:
            sum_hidden_state_landmarks = 0
        history.texts.append(f"Number of Keypoints in Hidden States after Adding new Keypoints: {sum_hidden_state_landmarks}")
        print("Number of Keypoints in Hidden States after Adding new Keypoints:", sum_hidden_state_landmarks)
        
        
        
        
      
        """#####   Triangulate new Landmarks   #####"""
        triangulated_keypoints, triangulated_landmarks, triangulated_descriptors, triangulated_idx = self.triangulate_new_landmarks(Hidden_state, R_1, t_1)
        if len(triangulated_idx) > 0:
            history.texts.append(f"Number of new landmarks that pass the Angle threshold: {len(triangulated_idx)}")
            print(f"Number of new landmarks that pass the Angle threshold: {len(triangulated_idx)}")
        else: 
            history.texts.append(f"Number of new landmarks that pass the Angle threshold: is zero")
            print(f"Number of new landmarks that pass the Angle threshold: is zero")

        # Remove the triangulated features from the Hidden state
        if triangulated_idx:
            Hidden_state = [np.delete(Hidden_state[0], triangulated_idx, axis=1), 
                            np.delete(Hidden_state[1], triangulated_idx, axis=2), 
                            np.delete(Hidden_state[2], triangulated_idx, axis=2), 
                            np.delete(Hidden_state[3], triangulated_idx, axis=1),
                            np.delete(Hidden_state[4], triangulated_idx, axis=1)]
        history.texts.append(f"Number of Hidden States after removing triangulated keypoints: {Hidden_state[0].shape[1]}")
        print("Number of Hidden States after removing triangulated keypoints:", Hidden_state[0].shape[1])
        


        ### Remove Negative Points from Landmarks we want to add next###
        triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.remove_negative_points(triangulated_landmarks, triangulated_keypoints, triangulated_descriptors, R_1, t_1)

        if len(triangulated_landmarks) > 0:
            history.texts.append(f"Number of new Landmarks after removing the negatives: {triangulated_landmarks.shape[1]}")
        else:
            history.texts.append("Number of new Landmarks after removing the negatives: is zero")
        
        
        if triangulated_landmarks.size == 0:
            return keypoints_1, landmarks_1, descriptors_1, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors
        

        # # Reduce the number of new landmarks to 3 times the number of current landmarks
        # if triangulated_landmarks.shape[1] > 4 * landmarks_1.shape[1]:
        #     history.texts.append(f"Number of new Landmarks before reducing: {triangulated_landmarks.shape[1]}")
        #     print(f"Number of new Landmarks before reducing: {triangulated_landmarks.shape[1]}")
        #     # select 2*landmarks_1.shape[1] random indices
        #     random_indices = np.random.choice(triangulated_landmarks.shape[1], int(4 * landmarks_1.shape[1]), replace=False)
        #     triangulated_landmarks = triangulated_landmarks[:, random_indices]
        #     triangulated_keypoints = triangulated_keypoints[:, random_indices]
        #     triangulated_descriptors = triangulated_descriptors[:, random_indices]
        #     history.texts.append(f"Number of new Landmarks after reducing: {triangulated_landmarks.shape[1]}")
        #     print(f"Number of new Landmarks after reducing: {triangulated_landmarks.shape[1]}")
            

        landmarks_2 = np.hstack((landmarks_1, triangulated_landmarks))
        keypoints_2 = np.hstack((keypoints_1, triangulated_keypoints))
        descriptors_2 = np.hstack((descriptors_1, triangulated_descriptors))

        history.texts.append(f"keypoints_2.shape at the end of add_new_landmarks: {keypoints_2.shape}")
        
        return keypoints_2, landmarks_2, descriptors_2, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors

    def remove_negative_points(self, landmarks, keypoints, descriptors, R_1, t_1):

        if landmarks.size == 0:
            return landmarks, keypoints, descriptors
        


        forward_vector = R_1.T @ np.array([0, 0, 1])

        # Get the camera position
        camera_position = -R_1.T @ t_1


        # Get the direction vector from the camera to the landmarks
        direction_vector = landmarks - camera_position

        # Normalize the direction vectors
        norms = np.linalg.norm(direction_vector, axis=0)  # Shape: (N,)
        norms[norms == 0] = 1.0  # Prevent division by zero
        direction_vector = direction_vector / norms  # Shape: (3, N)


        # Calculate the dot product between the direction vector and the forward vector
        dot_product = []
        for vector in direction_vector.T:
            dot_product.append(np.dot(vector, forward_vector))

        dot_product = np.array(dot_product)

        # Find indices where the angle is less than 90 degrees
        positive_indices = np.where(dot_product > 0)[0]

        # Filter landmarks, keypoints, and descriptors
        landmarks_positive = landmarks[:, positive_indices]
        keypoints_positive = keypoints[:, positive_indices]
        descriptors_positive = descriptors[:, positive_indices]

        # Filter negative landmarks (opposite of positive_indices)
        negative_indices = np.where(dot_product <= 0)[0]

        landmarks_negative = landmarks[:, negative_indices]
        keypoints_negative = keypoints[:, negative_indices]
        descriptors_negative = descriptors[:, negative_indices]

        #flip location of negative landmarks around the camera location point
        landmarks_negative = 2 * camera_position - landmarks_negative


        landmarks_positive = np.hstack((landmarks_positive, landmarks_negative))
        keypoints_positive = np.hstack((keypoints_positive, keypoints_negative))
        descriptors_positive = np.hstack((descriptors_positive, descriptors_negative))


        return landmarks_positive, keypoints_positive, descriptors_positive
    
    def filter_keypoints_via_fundamental_ransac(
        self,
        keypoints_0, 
        keypoints_1, 
        ransacReprojThreshold=1.0, 
        confidence=0.99, 
        maxIters=2000
    ):
        """
        Filters matched 2D keypoints using a fundamental-matrix RANSAC step,
        mimicking estimateFundamentalMatrix in MATLAB.

        Args:
            keypoints_0 (np.ndarray): Shape (2, N). Keypoints in the first image (row, col) or (x, y).
            keypoints_1 (np.ndarray): Shape (2, N). Corresponding keypoints in the second image.
            ransacReprojThreshold (float): RANSAC distance threshold in pixels (or normalized coords).
            confidence (float): RANSAC confidence.
            maxIters (int): Maximum RANSAC iterations.

        Returns:
            inlier_keypoints_0 (np.ndarray): Shape (2, M). Inlier subset of keypoints_0.
            inlier_keypoints_1 (np.ndarray): Shape (2, M). Inlier subset of keypoints_1.
            inlier_mask (np.ndarray): Boolean mask of shape (N,) with True for inliers.
        """
        num_points = keypoints_0.shape[1]
        if num_points < 8:
            # Not enough points to compute fundamental matrix
            inlier_mask = np.ones(num_points, dtype=bool)  # trivially keep everything
            return keypoints_0, keypoints_1, inlier_mask

        # Convert from (2, N) to (N, 2) because cv2.findFundamentalMat expects (x, y) per row
        pts0 = keypoints_0.T  # shape (N, 2)
        pts1 = keypoints_1.T  # shape (N, 2)

        # Compute fundamental matrix using RANSAC
        # Note: FM_RANSAC uses pixel coords. If you want more strict filtering, reduce ransacReprojThreshold
        F, mask = cv2.findFundamentalMat(
            pts0, 
            pts1, 
            method=cv2.FM_RANSAC, 
            ransacReprojThreshold=ransacReprojThreshold, 
            confidence=confidence, 
            maxIters=maxIters
        )
        if mask is None:
            # If F can't be found or something fails, return original
            inlier_mask = np.ones(num_points, dtype=bool)
            return keypoints_0, keypoints_1, inlier_mask

        # 'mask' is an N x 1 array of 0/1 indicating outliers/inliers
        mask = mask.ravel().astype(bool)

        inlier_keypoints_0 = keypoints_0[:, mask]
        inlier_keypoints_1 = keypoints_1[:, mask]

        return inlier_keypoints_0, inlier_keypoints_1, mask


    def process_image(self, prev_image, image, keypoints_0, landmarks_0, descriptors_0, R_0, t_0, Hidden_state, history):

        self.image = image
        self.prev_image = prev_image
        
        ###Track keypoints from last frame to this frame using KLT###
        history.texts.append(f"-6. keypoints_0.shape at the beginning of process_image: {keypoints_0.shape}")
        print(f"-6. keypoints_0.shape at the beginning of process_image: {keypoints_0.shape}")
        keypoints_1, st = self.track_keypoints(prev_image, image, keypoints_0)
        history.texts.append(f"-5. keypoints_1.shape after track_keypoints : {keypoints_1.shape}")
        print(f"-5. keypoints_1.shape after track_keypoints : {keypoints_1.shape}")
        st = st.reshape(-1)
        #remove keypoints that are not tracked
        landmarks_1 = landmarks_0[:, st == 1]
        descriptors_1 = descriptors_0[:, st == 1]



        # tracked_keypoints_0 = keypoints_0[:, st == 1]
        # tracked_keypoints_1 = keypoints_1
        # # print(tracked_keypoints_0.shape)

        # # -----------------------------------------------------
        # # 2) Fundamental Matrix RANSAC (2D-2D outlier rejection)
        # # -----------------------------------------------------
        # # Filter out outliers among the newly tracked points
        # filtered_keypoints_0, filtered_keypoints_1, fm_inliers_mask = self.filter_keypoints_via_fundamental_ransac(
        #     tracked_keypoints_0, 
        #     tracked_keypoints_1, 
        #     # ransacReprojThreshold=0.05,
        #     ransacReprojThreshold=0.5,
        #     confidence=0.99,
        #     maxIters=2000
        # )
        # history.texts.append(f"-401. number of non filtered landmarks : {landmarks_1.shape}")
        # print(f"-401. number of non filtered landmarks : {landmarks_1.shape}")
        # keypoints_1 = keypoints_1[:, fm_inliers_mask] 
        # landmarks_1 = landmarks_1[:, fm_inliers_mask]
        # descriptors_1 = descriptors_1[:, fm_inliers_mask]
        # history.texts.append(f"-401. number of filtered landmarks : {landmarks_1.shape}")
        # print(f"-401. number of filtered landmarks : {landmarks_1.shape}")



        ###estimate motion using PnP###
        R_1,t_1, inliers = self.estimate_motion(keypoints_1, landmarks_1)
        # Use inliers to filter out outliers from keypoints and landmarks
        inliers = inliers.flatten()
        keypoints_1 = keypoints_1[:, inliers]
        landmarks_1 = landmarks_1[:, inliers]
        history.texts.append(f"-4. landmarks_1.shape after inliers filtering (RANSAC) : {landmarks_1.shape}")
        print(f"-4. keypoints_1.shape after inliers filtering (RANSAC) : {keypoints_1.shape}")
        descriptors_1 = descriptors_1[:, inliers]
        history.camera_position.append(-R_1.T @ t_1)




        """#####     Track hidden features     #####"""

        # Print cummulative number of keypoints in Hidden state before tracking
        sum_hidden_state_landmarks = 0
        if Hidden_state:
            sum_hidden_state_landmarks = Hidden_state[0].shape[1]
        else:
            sum_hidden_state_landmarks = 0
        history.texts.append(f"Number of Keypoints in Hidden State before Tracking: {sum_hidden_state_landmarks}")
        print("Number of Keypoints in Hidden State before Tracking:", sum_hidden_state_landmarks)

        # # Track and Update all Hidden States
        # Hidden_state = self.track_and_update_hidden_state(Hidden_state)

        #print cummulative number of keypoints in hidden state after tracking
        sum_hidden_state_landmarks = 0
        if Hidden_state:
            sum_hidden_state_landmarks = Hidden_state[0].shape[1]
        else:
            sum_hidden_state_landmarks = 0
        history.texts.append(f"Number of Keypoints in Hidden State after Tracking: {sum_hidden_state_landmarks}")
        print("Number of Keypoints in last Hidden State after Tracking:", sum_hidden_state_landmarks)



        """#####   Triangulate new Landmarks   ######"""


        keypoints_2, landmarks_2, descriptors_2, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors = \
            self.add_new_landmarks(keypoints_1, landmarks_1, descriptors_1, R_1, t_1, Hidden_state, history)



        #keeping this in case we want to fix the bundle adjustment
        R_2 = R_1
        t_2 = t_1
        landmarks_3 = landmarks_2
        keypoints_3 = keypoints_2
        descriptors_3 = descriptors_2
     

        ###Update History###
        history.keypoints.append(keypoints_3)
        history.landmarks.append(landmarks_3)
        history.R.append(R_2)
        history.t.append(t_2)
        
        history.triangulated_landmarks.append(triangulated_landmarks)
        history.triangulated_keypoints.append(triangulated_keypoints)
        history.threshold_angles.append(self.threshold_angle)
        history.num_keypoints.append(self.num_keypoints)
        history.Hidden_states.append(Hidden_state[0].shape[1])
        

        #  #This is only for the line plot plotting:
        # history.Hidden_states = Hidden_state

        # #check the discrepancy between the number of elements in Hidden_state and the frame number
        # if len(history.Hidden_states) < self.current_image_counter:
        #     #get difference
        #     difference = self.current_image_counter - len(history.Hidden_states)

        #     # Calculate positions to insert empty lists evenly
        #     interval = len(history.Hidden_states) // (difference + 1) if difference > 0 else 0
        #     for i in range(difference):
        #         insert_pos = (i + 1) * interval + i
        #         history.Hidden_states.insert(insert_pos, [])

        self.current_image_counter += 1



        # plot_2d(image, history)
        # plot_trajectory_and_landmarks(history)
        # plot_trajectory_and_landmarks_3d(history, save_html=True)



        return keypoints_3, landmarks_3, descriptors_3, R_2, t_2, Hidden_state, history


def plot_2d(img, history):
    triangulated_keypoints = history.triangulated_keypoints[-1]
    keypoints_history = history.keypoints

    plt.figure(figsize=(25, 18))
    #make sure the image is in color
    image_plotting = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    #plot previous keypoints in yellow
    for keypoints_from_history in keypoints_history[max(-10, -len(keypoints_history)):-1]:
        for kp in keypoints_from_history.T:
            center = tuple(kp.astype(int))
            cv2.circle(image_plotting, center, 3, (0, 255, 255), -1)

    #plot current keypoints blue
    for keypoints_from_history in keypoints_history[-1].T:
        center = tuple(keypoints_from_history.astype(int))
        cv2.circle(image_plotting, center, 3, (255, 0, 0), -1)

    #plot new keypoints in red
    for kp in triangulated_keypoints.T:
        center = tuple(kp.astype(int))
        cv2.circle(image_plotting, center, 3, (0, 0, 255), -1)

    image_rgb = cv2.cvtColor(image_plotting, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.axis('off')
    plt.savefig(f"output/debug_plot2d_{len(history.camera_position):06}.png")
    plt.close()


def plot_trajectory_and_landmarks(history):
    """
    Plots the 3D camera trajectory (motion estimation) and triangulated 3D landmarks.

    Args:
        history: A data structure containing:
                 - history.camera_position: list of camera centers in world coords
                 - history.landmarks: list of np.ndarray(3, N) with 3D landmarks
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1) Plot camera trajectory
    camera_positions = np.array(history.camera_position)  # shape (num_frames, 3)
    # print(camera_positions)
    ax.plot(camera_positions[:, 0, 0], camera_positions[:, 1, 0], camera_positions[:, 2, 0],
            'bo-', label='Camera Trajectory')

    # 2) Plot all triangulated landmarks
    #    Suppose you store each frame's set of landmarks in history.landmarks
    #    We'll collect them all into one array for plotting
    all_points = []
    for frame_landmarks in history.landmarks:
        if frame_landmarks.size == 0:
            continue
        all_points.append(frame_landmarks)
    if len(all_points) > 0:
        all_points = np.hstack(all_points)  # shape (3, total_points)
        ax.scatter(all_points[0, :],
                   all_points[1, :],
                   all_points[2, :],
                   c='r', marker='.', s=2, label='Triangulated 3D Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory & Triangulated Points')
    ax.legend()
    ax.view_init(elev=25, azim=-60)  # Adjust view angle for clarity if you want
    plt.tight_layout()
    plt.savefig(f"output/trajectory_and_landmarks_{len(history.camera_position):06}.png")
    plt.close()


def compute_reprojection_errors(
    R, t, K, landmarks_3d, keypoints_2d
):
    """
    Compute reprojection errors for the given landmarks and their corresponding keypoints.

    Args:
        R: 3x3 rotation matrix (world -> camera)
        t: 3x1 translation vector (world -> camera)
        K: 3x3 camera intrinsics
        landmarks_3d: shape (3, N), 3D points in world coords
        keypoints_2d: shape (2, N), 2D measured keypoints in the image

    Returns:
        A 1D numpy array of reprojection errors for each landmark-keypoint pair.
    """
    # Transform points into camera coords
    X_cam = R @ landmarks_3d + t  # shape (3, N)

    # Project onto image plane
    X_proj = K @ X_cam  # shape (3, N)
    x_proj = X_proj[:2] / X_proj[2]  # shape (2, N) -> (u, v)

    # Compute errors
    diff = keypoints_2d - x_proj  # shape (2, N)
    errors = np.sqrt(np.sum(diff**2, axis=0))  # shape (N,)
    return errors

def plot_reprojection_errors(errors, title_suffix=""):
    """
    Plot a histogram of reprojection errors.

    Args:
        errors (np.ndarray): 1D array of reprojection errors
        title_suffix (str): Additional title info, e.g. "Frame 10"
    """
    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=30, color='g', alpha=0.7, edgecolor='black')
    plt.title(f'Reprojection Error Distribution {title_suffix}')
    plt.xlabel('Error (pixels)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f"output/reprojection_errors_{title_suffix}.png")
    plt.close()

def check_reprojection_for_current_frame(R_1, t_1, K, landmarks_3d, keypoints_2d, frame_id):
    # Compute errors
    errors = compute_reprojection_errors(R_1, t_1, K, landmarks_3d, keypoints_2d)

    # Plot or log the stats
    plot_reprojection_errors(errors, title_suffix=f"Frame_{frame_id}")

    # Optional: print mean or max error
    print(f"[Frame {frame_id}] Mean reprojection error: {np.mean(errors):.2f} px, "
          f"Max error: {np.max(errors):.2f} px")


def plot_trajectory_and_landmarks_3d(history, save_html=True):
    """
    All points in a single trace, but colored by frame index (for example).
    This creates a continuous gradient by default. You can switch to discrete if needed.
    """

    fig = go.Figure()

    # Camera trajectory
    camera_positions = np.array([pos.flatten() for pos in history.camera_position])  # shape (num_frames, 3)
    fig.add_trace(
        go.Scatter3d(
            x=camera_positions[:, 0],
            y=camera_positions[:, 1],
            z=camera_positions[:, 2],
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            line=dict(color='blue', width=2),
            name='Camera Trajectory'
        )
    )

    # Collect all points + frame indices
    all_points = []
    all_indices = []  # store frame index for each point
    for frame_idx, frame_landmarks in enumerate(history.landmarks):
        if frame_landmarks.size == 0:
            continue
        all_points.append(frame_landmarks)
        # Suppose this frame has M points => create an array of shape (M,) with value frame_idx
        all_indices.append(np.full(frame_landmarks.shape[1], frame_idx))

    if len(all_points) == 0:
        fig.show()
        return

    # Concatenate across frames
    all_points = np.hstack(all_points)   # shape (3, total_points)
    all_indices = np.concatenate(all_indices)  # shape (total_points,)

    # Plot as one trace, colored by frame index
    fig.add_trace(
        go.Scatter3d(
            x=all_points[0],
            y=all_points[1],
            z=all_points[2],
            mode='markers',
            marker=dict(
                size=3,
                color=all_indices,      # color = frame index array
                colorscale='Turbo',     # or 'Viridis', 'Jet', 'Plotly3', etc.
                showscale=True,         # add colorbar
                colorbar=dict(title="Frame Index")
            ),
            name='Triangulated 3D Points'
        )
    )

    camera = dict(
        up=dict(x=0, y=-1, z=0),  # Setting Y as the up direction
        center=dict(x=0, y=0, z=0),  # Centering the view
        eye=dict(x=-camera_positions[-1, 0], y=camera_positions[-1, 1], z=-camera_positions[-1, 2])  # Adjusting the camera's position (x=2.5, y=0.1, z=0.1)
    )

    max_range = np.array([camera_positions.max(axis=0) - camera_positions.min(axis=0),
                          np.hstack(history.landmarks).max(axis=1) - np.hstack(history.landmarks).min(axis=1)]).max()

    mid_x = (camera_positions[:, 0].max() + camera_positions[:, 0].min()) / 2
    mid_y = (camera_positions[:, 1].max() + camera_positions[:, 1].min()) / 2
    mid_z = (camera_positions[:, 2].max() + camera_positions[:, 2].min()) / 2

    fig.update_layout(
        title="3D Landmarks by Frame (Different Colors)",
        scene=dict(
            xaxis=dict(range=[mid_x - max_range/2, mid_x + max_range/2], autorange=False),
            yaxis=dict(range=[mid_y - max_range/2, mid_y + max_range/2], autorange=False),
            zaxis=dict(range=[mid_z - max_range/2, mid_z + max_range/2], autorange=False),

            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=camera,
            aspectmode='cube',
        ),
        width=900,
        height=700
    )
    # Save or show
    if save_html:
        file_name = f"output/trajectory_and_landmarks_{len(history.camera_position):06}.html"
        fig.write_html(file_name)
        print(f"Saved interactive 3D plot to {file_name}")
    else:
        fig.show()
