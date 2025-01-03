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

    def match_features(self, descriptors_1, descriptors_0, keypoints_1, keypoints_0, history):
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
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # Number of checks to speed up search

            # Create FLANN-based matcher
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # Perform KNN matching
            knn_matches = flann.knnMatch(descriptors_0, descriptors_1, k=2)


            good = []
            for i, pair in enumerate(knn_matches):
                # m, n, _, _, _ = pair
                # good.append(m)
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
        retval, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            landmarks_1.astype(np.float32).T, 
            keypoints_1.astype(np.float32).T, 
            self.K, 
            distCoeffs=None,
            iterationsCount=2000,
            reprojectionError=10.0,
            confidence=0.995,
        )
        
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
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # Select good points
        st_here = st.reshape(-1)
        #get new keypoints
        
        keypoints_1 = keypoints_1.T[:,st_here == 1]
        
        return keypoints_1, st

    def track_and_update_hidden_state(self, Hidden_state, R_1, t_1):

        
        new_Hidden_state = []
        #check if Hidden_state is not just an emtpy list od lists
        if Hidden_state:
            for candidate in Hidden_state[:-1]:
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

    def remove_duplicate_keypoints(self, Hidden_state, history):

      
        newest_Hidden_state = Hidden_state[-1]
        num_keypoints = newest_Hidden_state[0].shape[1]
        
        indices_to_keep = np.ones(num_keypoints, dtype=bool)


        all_kp1 = np.empty((2, 0))
        all_kp2 = np.empty((2, 0))


        for candidate in Hidden_state[:-1]:
            if len(candidate) == 0:
                continue
            # Match features between the newest state and the candidate
            kp1, kp2, matches = self.match_features(
                newest_Hidden_state[6], candidate[6],
                newest_Hidden_state[3], candidate[3], history
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

            if kp1.shape != (0,) and kp2.shape != (0,):
                all_kp1 = np.hstack((all_kp1, kp1))
                all_kp2 = np.hstack((all_kp2, kp2))

        history.matches.append([all_kp1, all_kp2, []])


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
            for candidate_i, candidate in enumerate(Hidden_state[:-1]):
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
        """
        A one-stop 'statistical_filtering' that combines:
        1. Distance-based and outlier-based filtering (non-recursive).
        2. Angular-based binning to ensure a more even horizontal distribution.
        """

        # -------------------------
        # (1) Normal filtering by distance, etc.
        # -------------------------
        if landmarks.size == 0:
            return landmarks, keypoints, descriptors

        # Compute the camera position
        camera_position = -R_1.T @ t_1
        camera_position = camera_position.flatten()

        # 1a) Filter out extremely large outliers in the *landmarks distribution* itself
        mean_landmark = np.mean(landmarks, axis=1)
        distances = np.linalg.norm(landmarks - mean_landmark.reshape(3,1), axis=0)
        mean_distance = np.mean(distances)
        # Keep only points within 5x average spread
        idx_keep = (distances < 5 * mean_distance)

        # 1b) Filter by distance to camera
        distances_to_camera = np.linalg.norm(landmarks - camera_position.reshape(3,1), axis=0)
        mean_camdist = np.mean(distances_to_camera)
        std_camdist = np.std(distances_to_camera)

        # Keep only those not too close or too far from average
        # For instance: ±2σ from mean, also clamp absolute [0.5 ... 150] as in your code
        lower_bound = np.maximum(0.5, mean_camdist - 2*std_camdist)
        upper_bound = np.minimum(150.0, mean_camdist + 2*std_camdist)
        in_range = (distances_to_camera > lower_bound) & (distances_to_camera < upper_bound)

        idx_keep = idx_keep & in_range

        # Apply this first pass keep-mask
        landmarks = landmarks[:, idx_keep]
        keypoints = keypoints[:, idx_keep]
        descriptors = descriptors[:, idx_keep]

        if landmarks.size == 0:
            return landmarks, keypoints, descriptors

        # -------------------------
        # (2) Angular Binning to Evenly Distribute Points
        # -------------------------
        # 2a) Transform the (already filtered) landmarks into camera coords
        #     shape -> (3, N)
        land_cam = R_1 @ (landmarks - camera_position.reshape(3,1))

        # 2b) Compute horizontal angles alpha_i = atan2(X, Z)
        angles = np.arctan2(land_cam[0,:], land_cam[2,:])  # shape (N,)

        # 2c) Bin them
        num_bins = 36   # e.g., 10° increments, 36 bins in [-π, π]
        bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
        bin_indices = np.digitize(angles, bin_edges)  # yields bin index in [1..num_bins]

        # 2d) Sub-sample each bin
        max_per_bin = 10  # keep at most 10 points per bin
        keep_mask = np.zeros(angles.shape[0], dtype=bool)  # track which points to keep

        for b in range(1, num_bins + 1):
            idx_in_bin = np.where(bin_indices == b)[0]
            count_bin = len(idx_in_bin)
            if count_bin == 0:
                continue

            if count_bin <= max_per_bin:
                keep_mask[idx_in_bin] = True
            else:
                # pick random or top scoring (example: random)
                chosen = np.random.choice(idx_in_bin, size=max_per_bin, replace=False)
                keep_mask[chosen] = True

        # 2e) Final keep
        landmarks = landmarks[:, keep_mask]
        keypoints = keypoints[:, keep_mask]
        descriptors = descriptors[:, keep_mask]

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

        # Remove all the newly detected keypoints based on the keypoints
        # that are already in the Hidden state and in the current frame (or at least the ones that are in the current frame)
        removal_index = self.NMS_on_keypoints(new_keypoints, keypoints_1, radius=self.nonmaximum_suppression_radius)


        #remove the newly detected keypoints that are too close to the already tracked keypoints
        new_keypoints = np.delete(new_keypoints, removal_index, axis=1)
        new_descriptors = np.delete(new_descriptors, removal_index, axis=1)

        history.texts.append(f"Number of new keypoints after NMS added to latest Hidden State: {new_keypoints.shape[1]}")


        # Add new keypoints & descriptors to the Hidden_state
        # TODO: why are new keypoints, rotation and translation doubled?
        # TODO: Why do appear rotation and translation at all? also counts as landmakrs because len>0
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
        Hidden_state = self.remove_duplicate_keypoints(Hidden_state, history)

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
        ### Spatial Non Maximum Suppression between within new and old Landmarks ###
        triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.spatial_non_maximum_suppression(triangulated_keypoints, triangulated_landmarks, triangulated_descriptors, keypoints_1, landmarks_1, descriptors_1)
        if len(triangulated_landmarks) > 0:
            history.texts.append(f"Number of the triangulated_landmarks after NMS: {triangulated_landmarks.shape[1]}")
        else:
            history.texts.append(f"Number of the triangulated_landmarks after NMS: is zero")

        ### Statistical Filtering of new Landmarks ###
        triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.statistical_filtering(triangulated_keypoints, triangulated_landmarks, triangulated_descriptors, R_1, t_1)

        if len(triangulated_landmarks) > 0:
            history.texts.append(f"Number of the triangulated_landmarks after statistical_filtering: {triangulated_landmarks.shape[1]}")
        else:
            history.texts.append(f"Number of the triangulated_landmarks after statistical_filtering: is zero")
        
        if triangulated_landmarks.size == 0:
            return keypoints_1, landmarks_1, descriptors_1, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors
        

        # # # Reduce number of new points if they are too many (more than 10% of the currently tracked points)
        # num_points_to_keep = 50
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
            self.num_keypoints = 100#max(10,int(-sum_hidden_state_landmarks + min(500,self.current_image_counter*200)))
            # print(f"-6. self.num_keypoints: {self.num_keypoints}")


        #self.num_keypoints = max(1,-landmarks_1.shape[1] + 500)

        
        #Adapt the threshold angle dynamically based on the number of keypoints being tracked right now

        #this adapts the threshold angle to be higher if more keypoints are tracked (make it harder for new ones to be added)
        #and lower if less keypoints are tracked (make it easier for new ones to be added)
        if not self.use_sift:
            self.threshold_angle = round(max(0.02, landmarks_1.shape[1] / 3000), 2)
        if self.use_sift:
            self.threshold_angle = 0.00 #round(max(0.001, landmarks_1.shape[1] / 18000), 2)
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

        # # Filter negative landmarks (opposite of positive_indices)
        # negative_indices = np.where(dot_product <= 0)[0]

        # landmarks_negative = landmarks[:, negative_indices]
        # keypoints_negative = keypoints[:, negative_indices]
        # descriptors_negative = descriptors[:, negative_indices]

        # #flip location of negative landmarks around the camera location point
        # landmarks_negative = 2 * camera_position - landmarks_negative


        # landmarks_positive = np.hstack((landmarks_positive, landmarks_negative))
        # keypoints_positive = np.hstack((keypoints_positive, keypoints_negative))
        # descriptors_positive = np.hstack((descriptors_positive, descriptors_negative))


        return landmarks_positive, keypoints_positive, descriptors_positive

    def process_image(self, prev_image, image, keypoints_0, landmarks_0, descriptors_0, R_0, t_0, Hidden_state, history):
        


        #TODO: find memory leak in history
        #TODO: find possible memory leak in Hidden_state
        




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




        def plot_matches(img1, img2, kp1, kp2, matches):
            """
            Plots matches between img1 and img2 using the provided kp1, kp2 keypoints.
            
            Args:
                img1 (numpy.ndarray): The first image (BGR format).
                img2 (numpy.ndarray): The second image (BGR format).
                kp1 (numpy.ndarray): Array of shape (N, 2) with x,y keypoint coordinates for the first image.
                kp2 (numpy.ndarray): Array of shape (N, 2) with x,y keypoint coordinates for the second image.
                matches (numpy.ndarray): 1D array where matches[i] is either -1 (no match)
                                         or an index referencing a corresponding keypoint.
                                         In this snippet, you're using it so that if matches[i] != -1,
                                         then keypoint i is matched with keypoint i (i.e., a 1-to-1 match).
            """

            # Convert kp1, kp2 to lists of cv2.KeyPoint.
            # Here we assume kp1[i] = [x_i, y_i], shape is (N,2).
            # cv2.KeyPoint expects (x, y, size).
            # Convert kp1, kp2 to lists of cv2.KeyPoint.
            if kp1.shape != (0,) and kp2.shape != (0,):
                keypoints1 = [cv2.KeyPoint(x=float(kp1[0, i]), y=float(kp1[1, i]), size=50) for i in range(kp1.shape[1])]
                keypoints2 = [cv2.KeyPoint(x=float(kp2[0, i]), y=float(kp1[1, i]), size=50) for i in range(kp2.shape[1])]

                # Validate and build a list of DMatch objects for matched keypoints.
                # print(f"kp1: {kp1}")
                # print(f"matches.shape: {matches.shape}")
                # print(matches)
                dmatches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0) for i in range(len(keypoints1))]

                # print(dmatches)

                # Draw the matches on a single image
                img_matches = cv2.drawMatches(
                    img1, 
                    keypoints1,
                    img2, 
                    keypoints2, 
                    dmatches, 
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    matchesThickness=5,
                )



                # Plot using matplotlib
                plt.figure(figsize=(25, 18))
                plt.imshow(img_matches)
                plt.axis('off')
                plt.savefig(f"output/debug_matches_{len(history.camera_position):06}.png")
                plt.close()

        # if history.matches:
        #     plot_matches(prev_image, image, history.matches[-1][0], history.matches[-1][1], history.matches[-1][2])



        return keypoints_3, landmarks_3, descriptors_3, R_2, t_2, Hidden_state, history