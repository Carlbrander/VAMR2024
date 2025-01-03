import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

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
       
        # Propagate arguments
        self.corner_patch_size = args.corner_patch_size
        self.harris_kappa = args.harris_kappa
        self.num_keypoints = args.num_keypoints
        self.nonmaximum_suppression_radius = args.nonmaximum_supression_radius
        self.descriptor_radius = args.descriptor_radius
        self.match_lambda = args.match_lambda
        self.K = args.K
        self.threshold_angle = args.threshold_angle
        self.min_baseline = args.min_baseline

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

    def match_features(self, descriptors_1, descriptors_0, keypoints_1, keypoints_0):
        """
        Match features between two frames

        Args:
            descriptors_0 (numpy.ndarray): Descriptors from last image
            descriptors_1 (numpy.ndarray): Descriptors from this image

        Returns:
            list: Good matches between the two frames
        """
        
        if not self.use_sift:
            matches = matchDescriptors(descriptors_1, descriptors_0, self.match_lambda)

            #get matched keypoints from matches
            matched_keypoints_1 = keypoints_1.T[matches != -1]
            matched_keypoints_0 = keypoints_0.T[matches[matches != -1]]

            #transform both keypoint arrays from harris representation to cv2 representation 
            # Which means Swap coordinates from (row, col) to (x, y)
            matched_keypoints_1 = matched_keypoints_1[:, ::-1]
            matched_keypoints_0 = matched_keypoints_0[:, ::-1]

        if self.use_sift:


            descriptors_0 = descriptors_0.astype(np.float32).T
            descriptors_1 = descriptors_1.astype(np.float32).T

            keypoints_0 = keypoints_0.T
            keypoints_1 = keypoints_1.T


            # Match descriptors
            bf = cv2.BFMatcher()
            knn_matches = bf.knnMatch(descriptors_0, descriptors_1, k=2)


            good = []
            for i, pair in enumerate(knn_matches):
                try:
                    m, n = pair
                    if m.distance < 0.75*n.distance:
                        good.append(m)

                except ValueError:
                    pass

            # Create a 1D array of -1
            matches_array = np.full(descriptors_1.shape[0], -1, dtype=int)
            for m in good:
                matches_array[m.trainIdx] = m.queryIdx

            matches = matches_array

            # Get matched keypoints
            matched_keypoints_0 = np.array([keypoints_0[match.queryIdx] for match in good])
            matched_keypoints_1 = np.array([keypoints_1[match.trainIdx] for match in good])

        return matched_keypoints_1.T, matched_keypoints_0.T, matches
            
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
                    iterationsCount=2000,
                    reprojectionError=8.0,
                    confidence=0.999)
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        #non homogenoius transformation
        R_1 = rotation_matrix
        t_1 = translation_vector


        return R_1, t_1, inliers

    def triangulate_landmark(self, keypoint_1, keypoint_2, R_1, t_1, R_2, t_2):
        """
        Triangulate new landmark from keypoint in two frames
        
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

        #transform landmark from homogenious to euclidean coordinates
        landmark_euclidean = landmark_homogenious / landmark_homogenious[3]
        landmark_final = landmark_euclidean[:3]

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
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # Select good points
        st_here = st.reshape(-1)
        #get new keypoints
        
        keypoints_1 = keypoints_1.T[:,st_here == 1]
        
        return keypoints_1, st

    def track_and_update_hidden_state(self, Hidden_state, R_1, t_1):
        # Track last frame keypoints in the current frame and limit the list history to the last 10 frames
        # Hidden_state[0] -> hidden states of the last frame
        # Hidden_state[1] -> hidden states of the second last frame
        # ...
        # original keypoints, original R, original t, current keypoint, descriptors
        #   2xN             , 3x3xN     , 3x1xN     , 2xN             , 2xN        

        
        new_Hidden_state = []
        #check if Hidden_state is not just an emtpy list od lists
        if Hidden_state:
            hidden_feature_original = Hidden_state[-1][0]
            R_original = Hidden_state[-1][1]
            t_original = Hidden_state[-1][2]
            hidden_features_last_frame = Hidden_state[-1][3]
            descriptor_last_frame = Hidden_state[-1][4]

            hidden_features_new_frame, st = self.track_keypoints(self.prev_image, self.image, hidden_features_last_frame)
            st = st.reshape(-1)

            new_Hidden_state = Hidden_state 
            # Add the non-tracked features to the Hidden_state[-1]
            non_tracked_indices = st == 0
            if np.any(non_tracked_indices):
                new_Hidden_state[-1][0] = Hidden_state[-1][0][:, st == 0]
                new_Hidden_state[-1][1] = Hidden_state[-1][1][:,:, st == 0]
                new_Hidden_state[-1][2] = Hidden_state[-1][2][:,:, st == 0]
                new_Hidden_state[-1][3] = Hidden_state[-1][3][:, st == 0]
                new_Hidden_state[-1][4] = Hidden_state[-1][4][:, st == 0]
            else:
                new_Hidden_state[-1] = []

            # Only use the tracked features to append to the new Hidden_state
            oK = hidden_feature_original[:, st == 1]
            oR = R_original[:, :,st == 1]
            oT = t_original[:, :, st == 1]
            cK = hidden_features_new_frame
            descriptor = descriptor_last_frame[:, st == 1]


            # Append the updated Hidden_state to the new Hidden_state
            new_Hidden_state.append([oK, oR, oT, cK, descriptor])

            # Limit the history to the last 10 frames
            if len(new_Hidden_state) > 10:
                new_Hidden_state.pop(0)
        
        return new_Hidden_state

            

    def remove_duplicate_keypoints(self, Hidden_state):

      
        newest_Hidden_state = Hidden_state[-1]
        num_keypoints = newest_Hidden_state[0].shape[1]
        
        indices_to_keep = np.ones(num_keypoints, dtype=bool)

        for candidate in Hidden_state[:-1]:
            if len(candidate) == 0:
                continue
            # Match features between the newest state and the candidate
            _, _, matches = self.match_features(
                newest_Hidden_state[6], candidate[6],
                newest_Hidden_state[3], candidate[3]
            )

            #do spatial NMS between the keypoints of the newest hidden state and the keypoints of the candidate
            for i in candidate[3].T:
                for j in newest_Hidden_state[3].T:
                    if np.linalg.norm(i - j) < self.nonmaximum_suppression_radius:
                        indices_to_keep[np.where(np.all(newest_Hidden_state[3].T == j, axis=1))] = False



          
            # Indices where there is a match
            matched_indices = np.where(matches != -1)[0]

            # Ensure matched_indices are within bounds
            matched_indices = matched_indices[matched_indices < indices_to_keep.size]

            # Update the mask to False for matched indices
            indices_to_keep[matched_indices] = False
            
        
        # Apply the mask once after the loop
        newest_Hidden_state[0] = newest_Hidden_state[0][:, indices_to_keep]
        newest_Hidden_state[3] = newest_Hidden_state[3][:, indices_to_keep]
        newest_Hidden_state[6] = newest_Hidden_state[6][:, indices_to_keep]

        Hidden_state[-1] = newest_Hidden_state

       

        return Hidden_state

    def triangulate_new_landmarks(self, Hidden_state, R_1, t_1):
        new_keypoints = []
        new_descriptors = []
        new_landmarks = []
        
        if Hidden_state:
            candidates = Hidden_state[-1]
            for i in range(candidates[0].shape[1]):
                oK = candidates[0][:, i]
                R_0 = candidates[1][:, :, i]
                t_0 = candidates[2][:, :, i]
                cK = candidates[3][:, i]
                descriptor = candidates[4][:, i]

                # baseline between the two camera poses
                baseline = np.linalg.norm(t_1 - t_0)
                if baseline < self.min_baseline:
                    continue
    
                # Triangulate new landmark
                landmark = self.triangulate_landmark(
                    oK, cK, R_0, t_0, R_1, t_1
                )
                
                # Check angle between the two camera poses and the landmark
                angle = self.calculate_angle(landmark, t_0, t_1)
                if angle < self.threshold_angle:
                    continue

                # Append the new landmark to the list
                new_keypoints.append(oK)
                new_descriptors.append(descriptor)
                new_landmarks.append(landmark)

        # Convert lists to numpy arrays
        new_keypoints = np.array(new_keypoints).T
        new_landmarks = np.array(new_landmarks).T.reshape(3,-1)
        new_descriptors = np.array(new_descriptors).T

        return new_keypoints, new_landmarks, new_descriptors

    def calculate_angle(self, landmark, t_1, t_2):
        #get the direction vector from the first observation camera pose to the landmark
        direction_1 = landmark - t_1
        #get the direction vector from the current camera pose to the landmark
        direction_2 = landmark - t_2

        #normalize the vectors
        direction_1 = direction_1 / np.linalg.norm(direction_1)
        direction_2 = direction_2 / np.linalg.norm(direction_2)

        #get the angle between the two vectors
        angle = np.arccos(np.clip(np.dot(direction_1.flatten(), direction_2.flatten()), -1.0, 1.0))

        return angle
    
    def NMS_on_keypoints(self, new_keypoints, old_keypoints, radius):


        removal_index = set()

        # Loop over each new keypoint
        for i in range(new_keypoints.shape[1]):

            # Check distance to old_keypoints
            # TODO: pass here all of the detected keypoints instead of only limited new ones
            for j in range(old_keypoints.shape[1]):
                dist_old = np.linalg.norm(new_keypoints[:, i] - old_keypoints[:, j])
                if dist_old < radius:
                    removal_index.add(i)
                    break
            else:
                # Only check new_keypoints if not removed by old_keypoints
                # TODO: this should already be done in the NMS after keypoint detection
                for j in range(new_keypoints.shape[1]):
                    if i == j:
                        continue
                    dist_new = np.linalg.norm(new_keypoints[:, i] - new_keypoints[:, j])
                    if dist_new < radius:
                        removal_index.add(i)
                        break

        return list(removal_index)

    def spatial_non_maximum_suppression(self, keypoints, landmarks, descriptors, keypoints_1, landmarks_1, descriptors_1):
        


        if landmarks.size == 0:
            return landmarks, keypoints, descriptors
        


        #define a threshold for the distance between keypoints
        threshold = 0.3 #in meters

        #initialize a list to store the indices of the keypoints to keep
        indices_to_keep = []

        for i in range(landmarks.shape[1]):
            #initialize a flag that indicates if the keypoint is close to another keypoint
            close_to_another_keypoint = False

            for j in range(landmarks_1.shape[1]):
              

                #calculate the distance between the keypoints
                distance = np.linalg.norm(landmarks[:, i] - landmarks_1[:, j])

                #check if the distance is below the threshold
                if distance < threshold:
                    close_to_another_keypoint = True
                    break

            #if the keypoint is not close to another keypoint, keep it
            if not close_to_another_keypoint:
                indices_to_keep.append(i)

        #apply the mask to the keypoints, landmarks, and descriptors
        keypoints = keypoints[:, indices_to_keep]
        landmarks = landmarks[:, indices_to_keep]
        descriptors = descriptors[:, indices_to_keep]

        return landmarks, keypoints, descriptors

    def statistical_filtering(self, keypoints, landmarks, descriptors, R_1, t_1):

        #This function checks for landmarks that are a lot further away from the camera than the average distance of all landmarks
        #and removes them from the list of landmarks

        if landmarks.size == 0:
            return landmarks, keypoints, descriptors
        
        # Get mean of all new landmarks
        mean_landmark = np.mean(landmarks, axis=1)

        # Get distance of each landmark to the mean landmark
        distances = np.linalg.norm(landmarks - mean_landmark[:, None], axis=0)

        # Get mean distance
        mean_distance = np.mean(distances)

        # Get indices of landmarks that are not too far away
        indices_to_keep = distances < 5 * mean_distance


       
        #Check if the distance of the landmark to the camera is not too small or too large
        #get the camera position
        camera_position = -R_1.T @ t_1

        camera_position = camera_position.flatten()

        #get the distance of the landmarks to the camera
        distances_to_camera = np.linalg.norm(landmarks - camera_position[:, None], axis=0)

        #get the mean distance of the landmarks to the camera
        mean_distance_to_camera = np.mean(distances_to_camera)

        #get the standard deviation of the distances to the camera
        std_distance_to_camera = np.std(distances_to_camera)

        #get the indices of the landmarks that are not too close or too far away from the camera
        indices_to_keep = np.logical_and(indices_to_keep,distances_to_camera > mean_distance_to_camera - 2 * std_distance_to_camera)

        indices_to_keep = np.logical_and(indices_to_keep,  distances_to_camera < mean_distance_to_camera + 2 * std_distance_to_camera)

        # Do an absolut filtering of the points that are too far away
        indices_to_keep = np.logical_and(indices_to_keep, distances_to_camera < 150)

        # Do an absollute filtering of the points that are too close
        indices_to_keep = np.logical_and(indices_to_keep, distances_to_camera > 0.5)



        # Apply mask
        landmarks = landmarks[:, indices_to_keep]
        keypoints = keypoints[:, indices_to_keep]
        descriptors = descriptors[:, indices_to_keep]

        return landmarks, keypoints, descriptors

    def R_to_Quaternion(self, R):

        #convert the rotation matrix to a quaternion for bundle adjustment
        #get the trace of the rotation matrix
        trace = np.trace(R)

        #get the diagonal elements of the rotation matrix
        R_diag = np.diag(R)

        #get the maximum diagonal element
        max_diag = np.max(R_diag)

        #initialize the quaternion
        q = np.zeros(4)

        if max_diag == R_diag[0]:
            
            q[0] = 1 + 2 * R[0, 0] - trace
            q[1] = R[1, 0] + R[0, 1]
            q[2] = R[2, 0] + R[0, 2]
            q[3] = R[1, 2] - R[2, 1]

        elif max_diag == R_diag[1]:

            q[0] = R[1, 0] + R[0, 1]
            q[1] = 1 + 2 * R[1, 1] - trace
            q[2] = R[2, 1] + R[1, 2]
            q[3] = R[2, 0] - R[0, 2]

        else:

            q[0] = R[2, 0] + R[0, 2]
            q[1] = R[2, 1] + R[1, 2]
            q[2] = 1 + 2 * R[2, 2] - trace
            q[3] = R[0, 1] - R[1, 0]

        #normalize the quaternion

        q = q / np.linalg.norm(q)

        return q
    
    def Quaternion_to_R(self, q):
        
        #convert the quaternion to a rotation matrix
        R = np.zeros((3, 3))

        R[0, 0] = 1 - 2 * q[2]**2 - 2 * q[3]**2
        R[0, 1] = 2 * q[1] * q[2] - 2 * q[0] * q[3]
        R[0, 2] = 2 * q[1] * q[3] + 2 * q[0] * q[2]

        R[1, 0] = 2 * q[1] * q[2] + 2 * q[0] * q[3]
        R[1, 1] = 1 - 2 * q[1]**2 - 2 * q[3]**2
        R[1, 2] = 2 * q[2] * q[3] - 2 * q[0] * q[1]

        R[2, 0] = 2 * q[1] * q[3] - 2 * q[0] * q[2]
        R[2, 1] = 2 * q[2] * q[3] + 2 * q[0] * q[1]
        R[2, 2] = 1 - 2 * q[1]**2 - 2 * q[2]**2

        return R

    def Bundle_Adjustment(self, keypoints_1, landmarks_1, descriptors_1, R_1, t_1, history):

            
        # Parameters to optimize
        window_size = min(5, len(history.R))
        R_list = history.R[-window_size:] + [R_1]
        t_list = history.t[-window_size:] + [t_1]
        landmarks_list = history.landmarks[-window_size:] + [landmarks_1]

        num_frames = len(R_list)
        K = self.K 

        # Prepare variables and indexing
        variables = []
        pose_indices = []
        landmark_indices = []
        num_landmarks = 0

        # Flatten poses into variables
        for i in range(num_frames):
            R_mat = R_list[i]
            t_vec = t_list[i].flatten()
            quaternion = self.R_to_Quaternion(R_mat)
            pose_indices.append(len(variables))
            variables.extend(quaternion)
            variables.extend(t_vec)

        # Flatten landmarks into variables
        for i, landmarks in enumerate(landmarks_list):
            for j in range(landmarks.shape[1]):
                landmark_indices.append((i, j, len(variables)))
                variables.extend(landmarks[:, j])
                num_landmarks += 1

        variables = np.array(variables)

        # Collect observations
        observations = []
        landmark_global_idx = 0
        for frame_idx, (keypoints, landmarks) in enumerate(zip(history.keypoints[-window_size:] + [keypoints_1], landmarks_list)):
            num_kp = keypoints.shape[1]
            for kp_idx in range(num_kp):
                keypoint = keypoints[:, kp_idx]
                observations.append({
                    'frame_idx': frame_idx,
                    'landmark_idx': landmark_global_idx,
                    'keypoint': keypoint
                })
                landmark_global_idx += 1

        # Cost function
        def cost_function(vars):
            residuals = []
            # Reconstruct poses
            poses = []
            for i in range(num_frames):
                idx = pose_indices[i]
                quaternion = vars[idx:idx+4]
                t_vec = vars[idx+4:idx+7]
                R_mat = self.Quaternion_to_R(quaternion)
                t_vec = t_vec.reshape((3,1))
                poses.append({'R': R_mat, 't': t_vec})

            # Reconstruct landmarks
            landmarks = []
            for i in range(num_landmarks):
                idx = landmark_indices[i][2]
                X = vars[idx:idx+3].reshape((3,1))
                landmarks.append(X)

            # Compute residuals
            for obs in observations:
                frame_idx = obs['frame_idx']
                landmark_idx = obs['landmark_idx']
                keypoint = obs['keypoint']
                pose = poses[frame_idx]
                X = landmarks[landmark_idx]

                # Project point
                X_cam = pose['R'] @ X + pose['t']
                X_proj = K @ X_cam

                u_proj = X_proj[0][0] / X_proj[2][0]
                v_proj = X_proj[1][0] / X_proj[2][0]

                residual_u = keypoint[0] - u_proj
                residual_v = keypoint[1] - v_proj

                residuals.extend([residual_u, residual_v])

            return residuals
        
        # Optimize the cost function
        res = least_squares(cost_function, variables, method='trf',ftol=1e-6,xtol=1e-6, gtol=1e-6, loss='linear', verbose=2)

        optimized_vars = res.x

        # Extract optimized poses and landmarks
        poses = []
        for i in range(num_frames):
            idx = pose_indices[i]
            quaternion = optimized_vars[idx:idx+4]
            t_vec = optimized_vars[idx+4:idx+7]
            R_mat = self.Quaternion_to_R(quaternion)
            t_vec = t_vec.reshape((3,1))
            poses.append({'R': R_mat, 't': t_vec})
        

        landmarks = []
        for i in range(num_landmarks):
            idx = landmark_indices[i][2]
            X = optimized_vars[idx:idx+3].reshape((3,1))
            landmarks.append(X)

        # Update history with optimized poses and landmarks
        for i in range(num_frames -1):
            history.R[-window_size + i] = poses[i]['R']
            history.t[-window_size + i] = poses[i]['t']
        
        optimized_R = poses[-1]['R']
        optimized_t = poses[-1]['t']

        # Update landmarks in history
        landmark_idx = 0
        for i in range(num_frames):
            landmarks_frame = landmarks_list[i]
            num_landmarks_frame = landmarks_frame.shape[1]
            optimized_landmarks_frame = np.zeros((3, num_landmarks_frame))
            for j in range(num_landmarks_frame):
                optimized_landmarks_frame[:, j] = landmarks[landmark_idx].flatten()
                landmark_idx +=1
            history.landmarks[-window_size + i] = optimized_landmarks_frame

        # Identify retained landmarks for the current frame based on reprojection error
        current_frame_landmarks = optimized_landmarks_frame
        current_frame_keypoints = history.keypoints[-1]
        retained_landmark_indices = []

        for j in range(current_frame_landmarks.shape[1]):
            X = current_frame_landmarks[:, j].reshape(3,1)
            pose = poses[-1]
            X_cam = pose['R'] @ X + pose['t']
            X_proj = K @ X_cam

            u_proj = X_proj[0][0] / X_proj[2][0]
            v_proj = X_proj[1][0] / X_proj[2][0]

            keypoint = current_frame_keypoints[:, j]
            reproj_error = np.sqrt((keypoint[0] - u_proj)**2 + (keypoint[1] - v_proj)**2)

            if reproj_error < 2.0:
                retained_landmark_indices.append(j)
            # Ensure retained_landmark_indices does not exceed the number of keypoints
            retained_landmark_indices = [idx for idx in retained_landmark_indices if idx < history.keypoints[-1].shape[1]]

        # Update keypoints and descriptors based on retained landmarks
        updated_keypoints = history.keypoints[-1][:, retained_landmark_indices]
        updated_descriptors = descriptors_1[retained_landmark_indices]

        # Optimized landmarks for current frame
        optimized_landmarks_current = history.landmarks[-1]

        return optimized_R, optimized_t, optimized_landmarks_current, updated_keypoints, updated_descriptors, history.landmarks

    def add_new_landmarks(self, keypoints_1, landmarks_1, descriptors_1, R_1, t_1, Hidden_state, history):
        print(f"-100. landmarks_1.shape: {landmarks_1.shape}")

        ### Detect new Keypoints ###
        new_keypoints, new_descriptors = self.detect_keypoints(self.image, self.num_keypoints, self.nonmaximum_suppression_radius)
        # switch rows and columns to get the correct format
        new_keypoints = new_keypoints[[1, 0], :]



        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image

        print("Number of Keypoints allowed to detect:", self.num_keypoints)
        print("Number of freshly Detected Keypoints:", new_keypoints.shape[1])

        # Remove all the newly detected keypoints based on the keypoints
        # that are already in the Hidden state and in the current frame (or at least the ones that are in the current frame)
        removal_index = self.NMS_on_keypoints(new_keypoints, keypoints_1, radius=self.nonmaximum_suppression_radius)


        #remove the newly detected keypoints that are too close to the already tracked keypoints
        new_keypoints = np.delete(new_keypoints, removal_index, axis=1)
        new_descriptors = np.delete(new_descriptors, removal_index, axis=1)

        print("Number of new keypoints after NMS with keypoints:", new_keypoints.shape[1])


        #print cummulative number of keypoints in last hidden state
        if Hidden_state:
            sum_hidden_state_landmarks = Hidden_state[-1][0].shape[1]
        else:
            sum_hidden_state_landmarks = 0
        print("Number of Keypoints in last Hidden State before Tracking:", sum_hidden_state_landmarks)

        # Track the hidden state keypoints in the new frame
        Hidden_state = self.track_and_update_hidden_state(Hidden_state, R_1, t_1)

        #print cummulative number of keypoints in hidden state
        if Hidden_state:
            sum_hidden_state_landmarks = Hidden_state[-1][0].shape[1]
        else:
            sum_hidden_state_landmarks = 0
        print("Number of Keypoints in last Hidden State after Tracking:", sum_hidden_state_landmarks)



        # Remove Duplicate Keypoints in newest Hidden state, NMS
        if Hidden_state:
            removal_index = self.NMS_on_keypoints(new_keypoints, Hidden_state[-1][3], radius=self.nonmaximum_suppression_radius)
            new_keypoints = np.delete(new_keypoints, removal_index, axis=1)
            new_descriptors = np.delete(new_descriptors, removal_index, axis=1)
        print("Number of new keypoints after NMS with Hidden State:", new_keypoints.shape[1])

        # Feature Matching of new keypoints with older Hidden States (last 10 frames)
        for candidate in Hidden_state[:-1]:
            if len(candidate) == 0:
                continue
            _, _, matches = self.match_features(new_descriptors, candidate[4], new_keypoints, candidate[3])
            # Get indices of matched keypoints
            matched_indices = np.where(matches != -1)[0]

            # Assuming R_1 and t_1 are the rotation matrix and translation vector for the current frame
            # and matched_indices contains the indices of the matched keypoints

            # Create 3x3xN and 3x1xN matrices for the matched keypoints
            R_matrices = np.repeat(R_1[:, :, np.newaxis], len(matched_indices), axis=2)
            t_vectors = np.repeat(t_1[:, :, np.newaxis], len(matched_indices), axis=2)

            # Update the Hidden_state with the new keypoints, rotation matrices, and translation vectors
            Hidden_state[-1][0] = np.hstack((Hidden_state[-1][0], new_keypoints[:, matched_indices]))
            Hidden_state[-1][1] = np.concatenate((Hidden_state[-1][1], R_matrices), axis=2)
            Hidden_state[-1][2] = np.concatenate((Hidden_state[-1][2], t_vectors), axis=2)
            Hidden_state[-1][3] = np.hstack((Hidden_state[-1][3], new_keypoints[:, matched_indices]))
            Hidden_state[-1][4] = np.hstack((Hidden_state[-1][4], new_descriptors[:, matched_indices]))

            # Remove matched keypoints from new keypoints
            new_keypoints = np.delete(new_keypoints, matched_indices, axis=1)
            new_descriptors = np.delete(new_descriptors, matched_indices, axis=1)

        print("Number of new keypoints after matching with Hidden State:", new_keypoints.shape[1])

        # Rest of the new keypoints that are not matched with the Hidden State, add them to the Hidden State[-1]
        # Check if Hidden_state is empty
        if not Hidden_state:
            # Initialize Hidden_state with the new keypoints, R, t, and descriptors
            R_matrices = np.repeat(R_1[:, :, np.newaxis], new_keypoints.shape[1], axis=2)
            t_vectors = np.repeat(t_1[:, :, np.newaxis], new_keypoints.shape[1], axis=2)
            Hidden_state.append([new_keypoints, R_matrices, t_vectors, new_keypoints, new_descriptors])
        else:
            # Ensure R and t are 3x3xN and 3x1xN matrices
            R_matrices = np.repeat(R_1[:, :, np.newaxis], new_keypoints.shape[1], axis=2)
            t_vectors = np.repeat(t_1[:, :, np.newaxis], new_keypoints.shape[1], axis=2)

            # Add the new keypoints, R, t, and descriptors to the latest frame in Hidden_state
            Hidden_state[-1][0] = np.hstack((Hidden_state[-1][0], new_keypoints))
            Hidden_state[-1][1] = np.concatenate((Hidden_state[-1][1], R_matrices), axis=2)
            Hidden_state[-1][2] = np.concatenate((Hidden_state[-1][2], t_vectors), axis=2)
            Hidden_state[-1][3] = np.hstack((Hidden_state[-1][3], new_keypoints))
            Hidden_state[-1][4] = np.hstack((Hidden_state[-1][4], new_descriptors))

        # Print cummulative number of keypoints in Hidden state
        sum_hidden_state_landmarks = 0
        if Hidden_state:
            sum_hidden_state_landmarks = Hidden_state[-1][0].shape[1]
        else:
            sum_hidden_state_landmarks = 0
        print("Number of Keypoints in last Hidden State after Adding new Keypoints:", sum_hidden_state_landmarks)


        "-----------------------------------------------------------------------------------------------------------------"
        ############ old code ################
        "-----------------------------------------------------------------------------------------------------------------"
        # Add new keypoints & descriptors to the Hidden_state
        # TODO: why are new keypoints, rotation and translation doubled?
        # TODO: Why do appear rotation and translation at all? also counts as landmakrs because len>0
        # Hidden_state.append([new_keypoints, R_1, t_1.reshape(3,1), new_keypoints, R_1, t_1.reshape(3,1), new_descriptors, self.current_image_counter]) 
        
        # ### Remove Duplicate Keypoints in newest Hidden state ###
        # Hidden_state = self.remove_duplicate_keypoints(Hidden_state)

        # #print cummulative number of keypoints in hidden state
        # sum_hidden_state_landmarks = 0
        # for candidate in Hidden_state:
        #     if len(candidate) == 0:
        #         continue
        #     sum_hidden_state_landmarks += candidate[0].shape[1]
        # print("Number of Keypoints in Hidden State after removing duplicates:", sum_hidden_state_landmarks)
        
        # #Remove hidden state that are empty lists
        # Hidden_state = [candidate for candidate in Hidden_state if len(candidate) > 0]
        # # Safely remove Hidden States that have less than 4 keypoints using list comprehension
        # Hidden_state = [candidate for candidate in Hidden_state if candidate[3].shape[1] >= 4]

        # #print cummulative number of keypoints in hidden state
        # sum_hidden_state_landmarks = 0
        # for candidate in Hidden_state:
        #     sum_hidden_state_landmarks += candidate[0].shape[1]
        # print("Number of Keypoints in Hidden State bafter removing too small candidates:", sum_hidden_state_landmarks)
                
        
        # # Check if current Hidden State has less than 4 keypoints
        # if Hidden_state and Hidden_state[-1][0].shape[1] < 4:
        #     print("Not enough keypoints to triangulate, removing newest Hidden State")
        #     Hidden_state.pop()
        #     print(f"1.1 keypoints_1.shape: {keypoints_1.shape}")
        #     return keypoints_1, landmarks_1, descriptors_1, Hidden_state, None, None, None
        
      
        ### Triangulate new Landmarks ###
        triangulated_keypoints, triangulated_landmarks, triangulated_descriptors = self.triangulate_new_landmarks(Hidden_state, R_1, t_1)
        if len(triangulated_keypoints) > 0:
            print("Number of new landmarks after triangulation that pass the Angle threshold: ", triangulated_keypoints.shape[1])
        else: 
            print("Number of new landmarks after triangulation that pass the Angle threshold: is zero")


        ### Remove Negative Points from Landmarks we want to add next###
        triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.remove_negative_points(triangulated_landmarks, triangulated_keypoints, triangulated_descriptors, R_1, t_1)

        if len(triangulated_keypoints) > 0:
            print("Number of new Landmarks after removing the negatives: ", triangulated_landmarks.shape[1])
        else:
            print("Number of new Landmarks after removing the negatives: is zero")
        
        # ### Spatial Non Maximum Suppression between within new and old Landmarks ###
        # triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.spatial_non_maximum_suppression(triangulated_keypoints, triangulated_landmarks, triangulated_descriptors, keypoints_1, landmarks_1, descriptors_1)
        # if len(triangulated_keypoints) > 0:
        #     print(f"00. Number of the triangulated_landmarks after NMS: {triangulated_landmarks.shape[1]}")
        # else:
        #     print(f"00. Number of the triangulated_landmarks after NMS: is zero")



        # ### Statistical Filtering of new Landmarks ###
        # triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.statistical_filtering(triangulated_keypoints, triangulated_landmarks, triangulated_descriptors, R_1, t_1)
        # if len(triangulated_keypoints) > 0:
        #     print(f"01. Number of the triangulated_landmarks after statistical_filtering: {triangulated_landmarks.shape[1]}")
        # else:
        #     print(f"01. Number of the triangulated_landmarks after statistical_filtering: is zero")
        

        if triangulated_landmarks.size == 0:
            return keypoints_1, landmarks_1, descriptors_1, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors
        

        # Reduce number of new points if they are too many (more than 10% of the currently tracked points)
        print(f"2. Number of the triangulated_landmarks before reducing number ('triangulated_landmarks.shape[1]'): {triangulated_landmarks.shape[1]}")
        print(f"2. landmarks_1.shape[1]: {landmarks_1.shape[1]}")

        num_points_to_keep = 75
        if triangulated_landmarks.shape[1] > num_points_to_keep:
            print("Too many new landmarks, reducing number")
            # num_points_to_keep = int(100)
            indices_to_keep = np.random.choice(triangulated_landmarks.shape[1], num_points_to_keep, replace=False)
            triangulated_landmarks = triangulated_landmarks[:, indices_to_keep]
            triangulated_keypoints = triangulated_keypoints[:, indices_to_keep]
            triangulated_descriptors = triangulated_descriptors[:, indices_to_keep]
            #update the Hidden state with the reduced number of new landmarks
            # Hidden_state[-1][0] = triangulated_keypoints
            # Hidden_state[-1][3] = triangulated_keypoints
            # Hidden_state[-1][6] = triangulated_descriptors

        print(f"3. Number of the triangulated_landmarks after reducing number ('triangulated_landmarks.shape[1]'): {triangulated_landmarks.shape[1]}")
        print(f"3. landmarks_1.shape[1]: {landmarks_1.shape[1]}")



        landmarks_2 = np.hstack((landmarks_1, triangulated_landmarks))
        keypoints_2 = np.hstack((keypoints_1, triangulated_keypoints))
        descriptors_2 = np.hstack((descriptors_1, triangulated_descriptors))

        print(f"1.2 keypoints_2.shape: {keypoints_2.shape}")
        
        return keypoints_2, landmarks_2, descriptors_2, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors
    
    def adapt_parameters(self, Hidden_state, keypoints_1, landmarks_1, descriptors_1, R_1, t_1):

        #Adapt the number of keypoints to detect dynamically based on the number of keypoints in the hidden state
        
        #linear function to adapt the number of keypoints to detect with no more at 300 keypoints
        

        landmarks_count = []
        for candidate in Hidden_state[:-1]:
            #if candidate is an empty list:
            if len(candidate) == 0:
                landmarks_count.append(0)
                continue
            landmarks_count.append(candidate[0].shape[1])  

        #get the number of keypoints in the hidden state
        sum_hidden_state_landmarks = sum(landmarks_count)

        if not self.use_sift:
            self.num_keypoints = max(10,int(-sum_hidden_state_landmarks + min(400,self.current_image_counter*200)))
        if self.use_sift:
            self.num_keypoints = max(500,int(-sum_hidden_state_landmarks + min(500,self.current_image_counter*200)))
            print(f"-6. self.num_keypoints: {self.num_keypoints}")


        #self.num_keypoints = max(1,-landmarks_1.shape[1] + 500)

        
        #Adapt the threshold angle dynamically based on the number of keypoints being tracked right now

        #this adapts the threshold angle to be higher if more keypoints are tracked (make it harder for new ones to be added)
        #and lower if less keypoints are tracked (make it easier for new ones to be added)
        if not self.use_sift:
            self.threshold_angle = round(max(0.02, landmarks_1.shape[1] / 3000), 2)
        if self.use_sift:
            self.threshold_angle = round(max(0.001, landmarks_1.shape[1] / 18000), 2)
            print(f"-101. self.threshold_angle: {self.threshold_angle}")


        # Calculate average landmark distance to camera
        distance = landmarks_1 - (-R_1.T @ t_1)
        distance = np.linalg.norm(distance, axis=0)
        avg_distance = np.mean(distance)
        self.min_baseline = max(0.5, avg_distance * 0.2)
        print(f"-102. self.min_baseline: {self.min_baseline}")

        

    def remove_negative_points(self, landmarks, keypoints, descriptors, R_1, t_1):

        if landmarks.size == 0:
            return landmarks, keypoints, descriptors
        

        forward_vector = R_1.T @ np.array([0, 0, 1])

        # Get the camera position
        camera_position = -R_1.T @ t_1

        # Get the direction vector from the camera to the landmarks
        direction_vector = landmarks - camera_position  # Shape: (3, N)

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

    def process_image(self, prev_image, image, keypoints_0, landmarks_0, descriptors_0, R_0, t_0, Hidden_state, history):
        


        #TODO: find memory leak in history
        #TODO: find possible memory leak in Hidden_state
        




        self.image = image
        self.prev_image = prev_image
    
        
        ###Track keypoints from last frame to this frame using KLT###
        print(f"-4. keypoints_0.shape: {keypoints_0.shape}")
        keypoints_1, st = self.track_keypoints(prev_image, image, keypoints_0)
        print(f"-3. keypoints_1.shape: {keypoints_1.shape}")
        st = st.reshape(-1)
        #remove keypoints that are not tracked
        landmarks_1 = landmarks_0[:, st == 1]
        descriptors_1 = descriptors_0[:, st == 1]

        ###estimate motion using PnP###
        R_1,t_1, inliers = self.estimate_motion(keypoints_1, landmarks_1)
        # Use inliers to filter out outliers from keypoints and landmarks
        inliers = inliers.flatten()
        keypoints_1 = keypoints_1[:, inliers]
        print(f"-2. landmarks_1.shape: {landmarks_1.shape}")
        landmarks_1 = landmarks_1[:, inliers]
        print(f"-1. landmarks_1.shape: {landmarks_1.shape}")
        descriptors_1 = descriptors_1[:, inliers]
        history.camera_position.append(-R_1.T @ t_1)

        ###Triangulate new Landmarks###

        # Adapt Parameter for Landmark Detection dynamically #
        self.adapt_parameters(Hidden_state, keypoints_1, landmarks_1, descriptors_1, R_1, t_1)

        keypoints_2, landmarks_2, descriptors_2, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors = \
            self.add_new_landmarks(keypoints_1, landmarks_1, descriptors_1, R_1, t_1, Hidden_state, history)
        

        ### Bundle Adjustment ###
        #if self.current_image_counter > 11:
            #print("Bundle Adjustment")
            #R_2, t_2, landmarks_3, updated_keypoints, descriptors_updated, landmarks_history = self.Bundle_Adjustment(keypoints_2, landmarks_2, descriptors_2, R_1, t_1, history)
            #print("modified R_2 from : ", t_1, "to: ", t_2, "with bundle adjustment")
#
            #keypoints_3 = updated_keypoints
            #descriptors_3 = descriptors_updated

          


        #keeping this in case we want to fix the bundle adjustment
        R_2 = R_1
        t_2 = t_1
        landmarks_3 = landmarks_2
        keypoints_3 = keypoints_2
        descriptors_3 = descriptors_2
        #update hidden state with the new camera location
        #if triangulated_keypoints.size != 0:
        #    Hidden_state[-1][1] = R_2
        #    Hidden_state[-1][2] = t_2.reshape(3, 1)
        #    Hidden_state[-1][4] = R_2
        #    Hidden_state[-1][5] = t_2.reshape(3, 1)




        ###Update History###
        history.keypoints.append(keypoints_3)
        history.landmarks.append(landmarks_3)
        history.R.append(R_2)
        history.t.append(t_2)
        
        history.triangulated_landmarks.append(triangulated_landmarks)
        history.triangulated_keypoints.append(triangulated_keypoints)
        history.threshold_angles.append(self.threshold_angle)
        history.num_keypoints.append(self.num_keypoints)
       
        #This is only for the line plot plotting:
        history.Hidden_states = Hidden_state

        #check the discrepancy between the number of elements in Hidden_state and the frame number
        if len(history.Hidden_states) < self.current_image_counter:
            #get difference
            difference = self.current_image_counter - len(history.Hidden_states)

            # Calculate positions to insert empty lists evenly
            interval = len(history.Hidden_states) // (difference + 1) if difference > 0 else 0
            for i in range(difference):
                insert_pos = (i + 1) * interval + i
                history.Hidden_states.insert(insert_pos, [])

        self.current_image_counter += 1





        return keypoints_3, landmarks_3, descriptors_3, R_2, t_2, Hidden_state, history