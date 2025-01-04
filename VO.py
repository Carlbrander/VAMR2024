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
                    reprojectionError=1.0,
                    confidence=0.999)
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        #non homogenoius transformation
        R_1 = rotation_matrix
        t_1 = translation_vector


        return R_1, t_1, inliers

    def triangulate_landmarks(self, keypoints_1, keypoints_2, R_1, t_1, R_2, t_2):
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
        landmarks_homogenious = cv2.triangulatePoints(P1, P2, keypoints_1, keypoints_2)

        #transform landmarks from homogenious to euclidean coordinates
        landmarks = landmarks_homogenious / landmarks_homogenious[3, :]
        landmarks_final = landmarks[:3, :]

        return landmarks_final

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
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01),
        )
    
        # Select good points
        st_here = st.reshape(-1)
        #get new keypoints
        
        keypoints_1 = keypoints_1.T[:,st_here == 1]
        
        return keypoints_1, st

    def track_and_update_hidden_state(self, Hidden_state, R_1, t_1):

        
        new_Hidden_state = []
        #check if Hidden_state is not just an emtpy list od lists
        if Hidden_state:
            for candidate in Hidden_state[-2:-1]:
                if len(candidate) == 0:
                    new_Hidden_state.append(candidate)
                    continue
               

                candidate_keypoints, st  = self.track_keypoints(self.prev_image, self.image, candidate[3])
                
                    
               

                st = st.reshape(-1)

                # Convert candidates to a list to allow modification
                candidate = list(candidate)

                #candidate_keypoints = candidate_keypoints[:, st == 1]
                assert candidate_keypoints.shape[1] <= candidate[3].shape[1]

                #remove keypoints that are not tracked and update the current keypoints with the updated tracked keypoints
                candidate[3] = candidate_keypoints

                #remove the keypoints in their initial oberservation if they are not tracked anymore
                candidate[0] = candidate[0][:, st == 1]
                assert candidate[0].shape[1] == candidate[3].shape[1]

                #remove the descriptors in their initial obersrvation if they are not tracked anymore
                candidate[6] = candidate[6][:, st == 1]
                assert candidate[6].shape[1] == candidate[3].shape[1]
             
                candidate[4] = R_1

                candidate[5] = t_1.reshape(3, 1)


                candidate = [np.array(candidate[0]), np.array(candidate[1]), np.array(candidate[2]), np.array(candidate[3]), np.array(candidate[4]), np.array(candidate[5]), np.array(candidate[6]), candidate[7]]

                new_Hidden_state.append(candidate)

        
        new_Hidden_state.append(Hidden_state[-1])

      

                

        return new_Hidden_state

    def remove_duplicate_keypoints(self, Hidden_state):
        newest_Hidden_state = Hidden_state[-1]
        num_keypoints = newest_Hidden_state[0].shape[1]

        indices_to_keep = np.ones(num_keypoints, dtype=bool)

        for candidate in Hidden_state[-2:-1]:
            if len(candidate) == 0:
                continue

            # Vectorized distance calculation between keypoints
            dist_matrix = np.linalg.norm(
                newest_Hidden_state[3][:, :, np.newaxis] - candidate[3][:, np.newaxis, :], axis=0
            )

            # Find indices where distance is less than the suppression radius
            close_indices = np.any(dist_matrix < self.nonmaximum_suppression_radius, axis=1)

            # Update indices to keep
            indices_to_keep[close_indices] = False
            # Indices where there is a match
            #matched_indices = np.where(matches != -1)[0]
        
            # Ensure matched_indices are within bounds
            #matched_indices = matched_indices[matched_indices < indices_to_keep.size]

            # Update the mask to False for matched indices
            #indices_to_keep[matched_indices] = False
        # Apply the mask once after the loop
        newest_Hidden_state[0] = newest_Hidden_state[0][:, indices_to_keep]
        newest_Hidden_state[3] = newest_Hidden_state[3][:, indices_to_keep]
        newest_Hidden_state[6] = newest_Hidden_state[6][:, indices_to_keep]

        Hidden_state[-1] = newest_Hidden_state

        return Hidden_state

    def triangulate_new_landmarks(self, Hidden_state):
        new_keypoints = []
        new_descriptors = []
        new_landmarks = []
        
        if Hidden_state:
            for candidate_i, candidate in enumerate(Hidden_state[-2:-1]):
                angles = []  # Reset angles for each candidate

                # Triangulate new landmarks
                landmarks = self.triangulate_landmarks(
                    candidate[0], candidate[3],
                    candidate[1], candidate[2],
                    candidate[4], candidate[5]
                )
                

                #check if baseline between the two camera poses is not too small
                baseline = np.linalg.norm(candidate[5] - candidate[2])

                if baseline < self.min_baseline:
                    continue

                for landmark in landmarks.T:
                    # Calculate bearing angle between the landmark and both camera views
                    angle = self.calculate_angle(landmark, candidate[2], candidate[5])
                    angles.append(angle)   

                # If angle > threshold, add to lists
                for idx, angle in enumerate(angles):
                    if angle >= self.threshold_angle:
                        new_keypoints.append(candidate[3][:, idx])
                        new_descriptors.append(candidate[6][:, idx])
                        new_landmarks.append(landmarks[:, idx])

        # Convert lists to numpy arrays
        new_keypoints = np.array(new_keypoints).T
        new_landmarks = np.array(new_landmarks).T
        new_descriptors = np.array(new_descriptors).T

        return new_keypoints, new_landmarks, new_descriptors

    def calculate_angle(self, landmark, t_1, t_2):
        #get the direction vector from the first observation camera pose to the landmark
        direction_1 = landmark - t_1.flatten()
        #get the direction vector from the current camera pose to the landmark
        direction_2 = landmark - t_2.flatten()

        #normalize the vectors
        direction_1 = direction_1 / np.linalg.norm(direction_1)
        direction_2 = direction_2 / np.linalg.norm(direction_2)

        #get the angle between the two vectors
        angle = np.arccos(np.clip(np.dot(direction_1, direction_2), -1.0, 1.0))

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
        # history.texts.append(f"landmarks_1.shape: {landmarks_1.shape}")

        ### Detect new Keypoints ###
        new_keypoints, new_descriptors = self.detect_keypoints(self.image, self.num_keypoints, self.nonmaximum_suppression_radius)
        # switch rows and columns to get the correct format
        new_keypoints = new_keypoints[[1, 0], :]





        if len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image

        history.texts.append(f"Number of Keypoints allowed to detect: {self.num_keypoints}")
        history.texts.append(f"Number of freshly Detected Keypoints: {new_keypoints.shape[1]}")






        #remove_indices = []
#
        ####NMS on the new keypoints only between each other ###
        #for i in range(new_keypoints.shape[1]):
        #    for j in range(new_keypoints.shape[1]):
        #        if i == j:
        #            continue
        #        dist = np.linalg.norm(new_keypoints[:, i] - new_keypoints[:, j])
        #        if dist < self.nonmaximum_suppression_radius:
        #            remove_indices.append(i)
        #            break
#
        #new_keypoints = np.delete(new_keypoints, remove_indices, axis=1)
        #new_descriptors = np.delete(new_descriptors, remove_indices, axis=1)


        # print("Number of Keypoints after NMS between all new keypoints:", new_keypoints.shape[1])



        #Remove all the newly detected keypoints based on the keypoints
        #that are already in the Hidden state and in the current frame (or at least the ones that are in the current frame)
        removal_index = self.NMS_on_keypoints(new_keypoints, keypoints_1, radius=self.nonmaximum_suppression_radius)

        #remove the newly detected keypoints that are too close to the already tracked keypoints
        new_keypoints = np.delete(new_keypoints, removal_index, axis=1)
        new_descriptors = np.delete(new_descriptors, removal_index, axis=1)

        history.texts.append(f"Number of new keypoints after NMS added to latest Hidden State: {new_keypoints.shape[1]}")


        # Add new keypoints & descriptors to the Hidden_state
        Hidden_state.append([new_keypoints, R_1, t_1.reshape(3,1), new_keypoints, R_1, t_1.reshape(3,1), new_descriptors, self.current_image_counter]) 
        
        
        #print cummulative number of keypoints in hidden state
        sum_hidden_state_landmarks = 0
        for candidate in Hidden_state:
            if len(candidate) == 0:
                continue
            sum_hidden_state_landmarks += candidate[0].shape[1]
        history.texts.append(f"Number of Keypoints in Hidden State before Tracking: {sum_hidden_state_landmarks}")

        ### Track and Update all Hidden States ###
        Hidden_state = self.track_and_update_hidden_state(Hidden_state, R_1, t_1)


        #print cummulative number of keypoints in hidden state
        sum_hidden_state_landmarks = 0
        for candidate in Hidden_state:
            if len(candidate) == 0:
                continue
            sum_hidden_state_landmarks += candidate[0].shape[1]
        history.texts.append(f"Number of Keypoints in Hidden State after Tracking: {sum_hidden_state_landmarks}")

        
        ### Remove Duplicate Keypoints in newest Hidden state ###
        Hidden_state = self.remove_duplicate_keypoints(Hidden_state)

        #print cummulative number of keypoints in hidden state
        sum_hidden_state_landmarks = 0
        for candidate in Hidden_state:
            if len(candidate) == 0:
                continue
            sum_hidden_state_landmarks += candidate[0].shape[1]
        history.texts.append(f"Number of Keypoints in Hidden State after removing duplicates: {sum_hidden_state_landmarks}")
        
        #Remove hidden state that are empty lists
        Hidden_state = [candidate for candidate in Hidden_state if len(candidate) > 0]
        # Safely remove Hidden States that have less than 4 keypoints using list comprehension
        Hidden_state = [candidate for candidate in Hidden_state if candidate[3].shape[1] >= 4]

        #print cummulative number of keypoints in hidden state
        sum_hidden_state_landmarks = 0
        for candidate in Hidden_state:
            sum_hidden_state_landmarks += candidate[0].shape[1]
        history.texts.append(f"Number of Keypoints in Hidden State bafter removing too small candidates: {sum_hidden_state_landmarks}")
                
        
        # Check if current Hidden State has less than 4 keypoints
        if Hidden_state and Hidden_state[-1][0].shape[1] < 4:
            history.texts.append(f"Not enough keypoints to triangulate, removing newest Hidden State; keypoints_1.shape: {keypoints_1.shape}")
            Hidden_state.pop()
            return keypoints_1, landmarks_1, descriptors_1, Hidden_state, None, None, None
        
      
        ### Triangulate new Landmarks ###
        triangulated_keypoints, triangulated_landmarks, triangulated_descriptors = self.triangulate_new_landmarks(Hidden_state)
        if len(triangulated_landmarks) > 0:
            history.texts.append(f"Number of new landmarks after triangulation that pass the Angle threshold: {triangulated_landmarks.shape[1]}")
        else: 
            history.texts.append(f"Number of new landmarks after triangulation that pass the Angle threshold: is zero")


        ### Remove Negative Points from Landmarks we want to add next###
        triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.remove_negative_points(triangulated_landmarks, triangulated_keypoints, triangulated_descriptors, R_1, t_1)

        if len(triangulated_landmarks) > 0:
            history.texts.append(f"Number of new Landmarks after removing the negatives: {triangulated_landmarks.shape[1]}")
        else:
            history.texts.append("Number of new Landmarks after removing the negatives: is zero")
        # ### Spatial Non Maximum Suppression between within new and old Landmarks ###
        # triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.spatial_non_maximum_suppression(triangulated_keypoints, triangulated_landmarks, triangulated_descriptors, keypoints_1, landmarks_1, descriptors_1)
        # if len(triangulated_landmarks) > 0:
        #     history.texts.append(f"Number of the triangulated_landmarks after NMS: {triangulated_landmarks.shape[1]}")
        # else:
        #     history.texts.append(f"Number of the triangulated_landmarks after NMS: is zero")

        # ### Statistical Filtering of new Landmarks ###
        # triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.statistical_filtering(triangulated_keypoints, triangulated_landmarks, triangulated_descriptors, R_1, t_1)

        if len(triangulated_landmarks) > 0:
            history.texts.append(f"Number of the triangulated_landmarks after statistical_filtering: {triangulated_landmarks.shape[1]}")
        else:
            history.texts.append(f"Number of the triangulated_landmarks after statistical_filtering: is zero")
        
        if triangulated_landmarks.size == 0:
            return keypoints_1, landmarks_1, descriptors_1, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors
        

        # Reduce number of new points if they are too many (more than 10% of the currently tracked points)
        # print(f"2. Number of the triangulated_landmarks before reducing number ('triangulated_landmarks.shape[1]'): {triangulated_landmarks.shape[1]}")
        # print(f"2. landmarks_1.shape[1]: {landmarks_1.shape[1]}")

        # if landmarks_1.shape[1] < 100:
        #     num_points_to_keep = 100
        # else:
        #     num_points_to_keep = 50
        # if triangulated_landmarks.shape[1] > num_points_to_keep:
        #     history.texts.append("Too many new landmarks, reducing number")
        #     # num_points_to_keep = int(100)
        #     indices_to_keep = np.random.choice(triangulated_landmarks.shape[1], num_points_to_keep, replace=False)
        #     triangulated_landmarks = triangulated_landmarks[:, indices_to_keep]
        #     triangulated_keypoints = triangulated_keypoints[:, indices_to_keep]
        #     triangulated_descriptors = triangulated_descriptors[:, indices_to_keep]
        #     #update the Hidden state with the reduced number of new landmarks
        #     Hidden_state[-1][0] = triangulated_keypoints
        #     Hidden_state[-1][3] = triangulated_keypoints
        #     Hidden_state[-1][6] = triangulated_descriptors

        history.texts.append(f"Number of the triangulated_landmarks after reducing number: {triangulated_landmarks.shape[1]}")

        landmarks_2 = np.hstack((landmarks_1, triangulated_landmarks))
        keypoints_2 = np.hstack((keypoints_1, triangulated_keypoints))
        descriptors_2 = np.hstack((descriptors_1, triangulated_descriptors))

        history.texts.append(f"keypoints_2.shape at the end of add_new_landmarks: {keypoints_2.shape}")
        
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
            self.num_keypoints = max(1,int(-sum_hidden_state_landmarks + min(400,self.current_image_counter*200)))
        if self.use_sift:
            self.num_keypoints = 800#max(10,int(-sum_hidden_state_landmarks + min(500,self.current_image_counter*200)))
            # print(f"-6. self.num_keypoints: {self.num_keypoints}")


        #self.num_keypoints = max(1,-landmarks_1.shape[1] + 500)

        
        #Adapt the threshold angle dynamically based on the number of keypoints being tracked right now

        #this adapts the threshold angle to be higher if more keypoints are tracked (make it harder for new ones to be added)
        #and lower if less keypoints are tracked (make it easier for new ones to be added)
        if not self.use_sift:
            self.threshold_angle = round(max(0.02, landmarks_1.shape[1] / 3000), 2)
        if self.use_sift:
            self.threshold_angle = 0.0001#round(max(0.001, landmarks_1.shape[1] / 18000), 2)
            # print(f"-101. self.threshold_angle: {self.threshold_angle}")

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

    def process_image(self, prev_image, image, keypoints_0, landmarks_0, descriptors_0, R_0, t_0, Hidden_state, history):

        self.image = image
        self.prev_image = prev_image
        
        ###Track keypoints from last frame to this frame using KLT###
        history.texts.append(f"-6. keypoints_0.shape at the beginning of process_image: {keypoints_0.shape}")
        keypoints_1, st = self.track_keypoints(prev_image, image, keypoints_0)
        history.texts.append(f"-5. keypoints_1.shape after track_keypoints : {keypoints_1.shape}")
        st = st.reshape(-1)
        #remove keypoints that are not tracked
        landmarks_1 = landmarks_0[:, st == 1]
        descriptors_1 = descriptors_0[:, st == 1]
        history.texts.append(f"-4. landmarks_1.shape after track_keypoints : {landmarks_1.shape}")

        ###estimate motion using PnP###
        R_1,t_1, inliers = self.estimate_motion(keypoints_1, landmarks_1)
        # Use inliers to filter out outliers from keypoints and landmarks
        inliers = inliers.flatten()
        keypoints_1 = keypoints_1[:, inliers]
        landmarks_1 = landmarks_1[:, inliers]
        history.texts.append(f"-3. landmarks_1.shape after inliers filtering : {landmarks_1.shape}")
        descriptors_1 = descriptors_1[:, inliers]
        history.camera_position.append(-R_1.T @ t_1)

        ###Triangulate new Landmarks###

        # Adapt Parameter for Landmark Detection dynamically #
        self.adapt_parameters(Hidden_state, keypoints_1, landmarks_1, descriptors_1, R_1, t_1)
        history.texts.append(f"-2. self.num_keypoints: {self.num_keypoints}")
        history.texts.append(f"-1. self.threshold_angle: {self.threshold_angle}")

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
