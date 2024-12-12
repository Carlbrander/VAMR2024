import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

#solution scripts from exercise 3 for feature detection and matching using shi-tomasi
from bootstrapping_utils.exercise_3.harris import harris
from bootstrapping_utils.exercise_3.select_keypoints import selectKeypoints
from bootstrapping_utils.exercise_3.describe_keypoints import describeKeypoints
from bootstrapping_utils.exercise_3.match_descriptors import matchDescriptors

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

        harris_scores = harris(gray, self.corner_patch_size, self.harris_kappa)
        keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_suppression_radius)
        descriptors = describeKeypoints(gray, keypoints, self.descriptor_radius)


        return keypoints, descriptors

    def match_features(self, descriptors_0, descriptors_1, keypoints_0, keypoints_1):
        """
        Match features between two frames

        Args:
            descriptors_0 (numpy.ndarray): Descriptors from last image
            descriptors_1 (numpy.ndarray): Descriptors from this image

        Returns:
            list: Good matches between the two frames
        """
        
        matches = matchDescriptors(descriptors_1, descriptors_0, self.match_lambda)

        #get matched keypoints from matches
        matched_keypoints_1 = keypoints_1.T[matches != -1]
        matched_keypoints_0 = keypoints_0.T[matches[matches != -1]]

        #transform both keypoint arrays from harris representation to cv2 representation 
        # Which means Swap coordinates from (row, col) to (x, y)
        matched_keypoints_1 = matched_keypoints_1[:, ::-1]
        matched_keypoints_0 = matched_keypoints_0[:, ::-1]


        

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
                    confidence=0.999,
                    flags=cv2.SOLVEPNP_P3P)
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        #R_1 = np.eye(4)
        #R_1[:3, :3] = rotation_matrix
        #
        #t_1 = np.zeros((4, 1))
        #t_1[:3, 0] = translation_vector.flatten()

        #non homogenoius transformation
        R_1 = rotation_matrix
        t_1 = translation_vector


        return R_1, t_1

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

          
            # Indices where there is a match
            matched_indices = np.where(matches != -1)[0]

            # Ensure matched_indices are within bounds
            matched_indices = matched_indices[matched_indices < indices_to_keep.size]

            # Update the mask to False for matched indices
            indices_to_keep[matched_indices] = False


            # Do NMS on the new matched keypoints compared to the tracked existing ones to not introduce duplicates close by
            for i in candidate[3].T:
                for j in newest_Hidden_state[3].T:
                    if np.linalg.norm(i - j) < self.nonmaximum_suppression_radius:
                        indices_to_keep[np.where(np.all(newest_Hidden_state[3].T == j, axis=1))] = False

            
        
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
            for candidate in Hidden_state[:-1]:
                
                
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
    
    def spatial_non_maximum_suppression(self, keypoints, landmarks, descriptors, keypoints_1, landmarks_1, descriptors_1):
        


        if landmarks.size == 0:
            return landmarks, keypoints, descriptors
        


        #define a threshold for the distance between keypoints
        threshold = 0.5 #in meters

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
        indices_to_keep = np.logical_and(indices_to_keep, distances_to_camera < 100)

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

        ### Detect new Keypoints ###
        new_keypoints, new_descriptors = self.detect_keypoints(self.image, self.num_keypoints, self.nonmaximum_suppression_radius)
        # switch rows and columns to get the correct format
        new_keypoints = new_keypoints[[1, 0], :]

        
        
        # Add new keypoints & descriptors to the Hidden_state
        Hidden_state.append([new_keypoints, R_1, t_1.reshape(3,1), new_keypoints, R_1, t_1.reshape(3,1), new_descriptors, self.current_image_counter]) 
        
        
        ### Track and Update all Hidden States ###
        Hidden_state = self.track_and_update_hidden_state(Hidden_state, R_1, t_1)

        
        ### Remove Duplicate Keypoints in newest Hidden state ###
        Hidden_state = self.remove_duplicate_keypoints(Hidden_state)
        
        #Remove hidden state that are empty lists
        Hidden_state = [candidate for candidate in Hidden_state if len(candidate) > 0]
        # Safely remove Hidden States that have less than 4 keypoints using list comprehension
        Hidden_state = [candidate for candidate in Hidden_state if candidate[3].shape[1] >= 4]
                
        
        # Check if current Hidden State has less than 4 keypoints
        if Hidden_state and Hidden_state[-1][0].shape[1] < 4:
            print("Not enough keypoints to triangulate, removing newest Hidden State")
            Hidden_state.pop()
            return keypoints_1, landmarks_1, descriptors_1, Hidden_state, None, None, None
        
      

     

        ### Triangulate new Landmarks ###
        
        triangulated_keypoints, triangulated_landmarks, triangulated_descriptors = self.triangulate_new_landmarks(Hidden_state)
        
        ### Remove Negative Points from Landmarks we want to add next###
        triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.remove_negative_points(triangulated_landmarks, triangulated_keypoints, triangulated_descriptors, R_1, t_1)

        ### Spatial Non Maximum Suppression between within new and old Landmarks ###
        triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.spatial_non_maximum_suppression(triangulated_keypoints, triangulated_landmarks, triangulated_descriptors, keypoints_1, landmarks_1, descriptors_1)

        ### Statistical Filtering of new Landmarks ###
        triangulated_landmarks, triangulated_keypoints, triangulated_descriptors = self.statistical_filtering(triangulated_keypoints, triangulated_landmarks, triangulated_descriptors, R_1, t_1)
        
        if triangulated_landmarks.size == 0:
            return keypoints_1, landmarks_1, descriptors_1, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors
        

        # Reduce number of new points if they are too many (more than 10% of the currently tracked points)
        if triangulated_landmarks.shape[1] > 0.1 * landmarks_1.shape[1]:
            print("Too many new landmarks, reducing number")
            num_points_to_keep = int(0.1 * landmarks_1.shape[1])
            indices_to_keep = np.random.choice(triangulated_landmarks.shape[1], num_points_to_keep, replace=False)
            triangulated_landmarks = triangulated_landmarks[:, indices_to_keep]
            triangulated_keypoints = triangulated_keypoints[:, indices_to_keep]
            triangulated_descriptors = triangulated_descriptors[:, indices_to_keep]
            #update the Hidden state with the reduced number of new landmarks
            Hidden_state[-1][0] = triangulated_keypoints
            Hidden_state[-1][3] = triangulated_keypoints
            Hidden_state[-1][6] = triangulated_descriptors



        landmarks_2 = np.hstack((landmarks_1, triangulated_landmarks))
        keypoints_2 = np.hstack((keypoints_1, triangulated_keypoints))
        descriptors_2 = np.hstack((descriptors_1, triangulated_descriptors))
        
        return keypoints_2, landmarks_2, descriptors_2, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors
    
    def adapt_parameters(self, Hidden_state, keypoints_1, landmarks_1, descriptors_1, R_1, t_1):

        #Adapt the number of keypoints to detect dynamically based on the number of keypoints in the hidden state
        
        #linear function to adapt the number of keypoints to detect with no more at 300 keypoints
        

        landmarks_count = []
        for candidate in Hidden_state:
            #if candidate is an empty list:
            if len(candidate) == 0:
                landmarks_count.append(0)
                continue
            landmarks_count.append(candidate[0].shape[1])  

        #get the number of keypoints in the hidden state
        sum_hidden_state_landmarks = sum(landmarks_count)

        self.num_keypoints = max(1,int(-sum_hidden_state_landmarks + min(400,self.current_image_counter*200)))



        #self.num_keypoints = max(1,-landmarks_1.shape[1] + 500)

        
        #Adapt the threshold angle dynamically based on the number of keypoints being tracked right now

        #this adapts the threshold angle to be higher if more keypoints are tracked (make it harder for new ones to be added)
        #and lower if less keypoints are tracked (make it easier for new ones to be added)
        self.threshold_angle = round(max(0.02, landmarks_1.shape[1] / 3000), 2)

    def remove_negative_points(self, landmarks, keypoints, descriptors, R_1, t_1):

        if landmarks.size == 0:
            return landmarks, keypoints, descriptors
        
        # Transform landmarks to camera coordinates
        X_cam = R_1 @ landmarks + t_1.reshape(3, 1)

        # Get z-coordinates in camera frame
        z_coords = X_cam[2, :]

        # Find indices where landmarks are in front of the camera
        positive_indices = np.where(z_coords > 0)[0]

        # Filter landmarks, keypoints, and descriptors
        landmarks_positive = landmarks[:, positive_indices]
        keypoints_positive = keypoints[:, positive_indices]
        descriptors_positive = descriptors[:, positive_indices]

        return landmarks_positive, keypoints_positive, descriptors_positive

    def visualize_dashboard(self, triangulated_keypoints, triangulated_landmarks, history):
        
        fig = plt.figure(figsize=(15, 10))
    
        #plot Dashboard
        self.plot_3d(history.landmarks, history.R, history.t, triangulated_landmarks, fig)
        self.plot_top_view(history.landmarks, history.R, history.t, triangulated_landmarks, fig)
        self.plot_2d(history.keypoints, triangulated_keypoints, fig)
        self.plot_line_graph(history.landmarks, history.Hidden_states, history.triangulated_landmarks, fig)

        #Add text on a free space between subplots for tracking parameters
        fig.text(0.27, 0.55, f'Threshold Angle: {self.threshold_angle}', ha='center', va='center', fontsize=12)
        #text right below
        fig.text(0.27, 0.53, f'New Keypoints Detection: {self.num_keypoints}', ha='center', va='center', fontsize=12)



        plt.tight_layout()

        plt.show()
        cv2.waitKey(0)

    def plot_3d(self,history_landmarks, history_R, history_t, triangulated_landmarks, ax):

        ax_3d = ax.add_subplot(221, projection='3d')
       
       
        ax_3d.view_init(elev=0, azim=-90)
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')

        #set axis lim
        ax_3d.set_xlim(-25, 25)
        ax_3d.set_ylim(-25, 25)
        ax_3d.set_zlim(0, 50)

        ax_3d.set_title('3D Plot')
        

        #get a set of history landmarks without the latest landmarks (as they are also in the history landmarks as duplicated likely)
        historic_landmarks = []
        for landmarks in history_landmarks[max(-20,-len(history_landmarks)):-1]:
            for landmark in landmarks.T:
                #check if it is in the latest landmarks
                if np.all(np.abs(landmark - history_landmarks[-1].T) < 1e-6):
                    continue
                historic_landmarks.append(landmark)

        historic_landmarks = np.array(historic_landmarks).T
        ax_3d.scatter(historic_landmarks[0, :], historic_landmarks[1, :], historic_landmarks[2, :], c='y', marker='o')





        for landmarks in history_landmarks[-len(history_landmarks):-1]:
            ax_3d.scatter(landmarks[0, :], landmarks[1, :], landmarks[2, :], c='y', marker='o')


        
        #plot landmarks from current frame in blue which have not been plotted before

        #get a set of latest landmarks without the triangulated_landmarks
        latest_landmarks = history_landmarks[-1][:, :]
        if triangulated_landmarks.size != 0:
            latest_landmarks = latest_landmarks[:, :-triangulated_landmarks.shape[1]]
        ax_3d.scatter(latest_landmarks[0, :], latest_landmarks[1, :], latest_landmarks[2, :], c='b', marker='o')
        #ax.scatter(history_landmarks[-1][0, :], history_landmarks[-1][1, :], history_landmarks[-1][2, :], c='b', marker='o')

        #plot triangulated landmarks in red
        if triangulated_landmarks.size != 0:
            ax_3d.scatter(triangulated_landmarks[0, :], triangulated_landmarks[1, :], triangulated_landmarks[2, :], c='r', marker='o')

        

        #plot camera positions in green
        for i in range(len(history_R[:-1])):
            R = history_R[i]
            t = history_t[i]
            camera_position = -R.T @ t

            camera_x = camera_position[0]
            camera_y = camera_position[1]
            camera_z = camera_position[2]

            ax_3d.scatter(camera_x, camera_y, camera_z, c='g', marker='x', s=100)


        # Plot the latest pose in red
        R = history_R[-1]
        t = history_t[-1]
        camera_position = -R.T @ t

        camera_x = camera_position[0]
        camera_y = camera_position[1]
        camera_z = camera_position[2]

        ax_3d.scatter(camera_x, camera_y, camera_z, c='r', marker='x', s=100)


        ############################################################
       
    def plot_top_view(self, history_landmarks, history_R, history_t, triangulated_landmarks, ax):
        #on second subplot show a 2D plot as top view (X-Z plane) with all landmarks and cameras
        ax_3d_1 = ax.add_subplot(222)
        ax_3d_1.set_xlabel('X')
        ax_3d_1.set_ylabel('Z')
        ax_3d_1.set_aspect('equal', adjustable='datalim')

        #plot old landmarks from the history in yellow until previous frame
        for landmarks in history_landmarks[max(-20,-len(history_landmarks)):-1]:
            ax_3d_1.scatter(landmarks[0, :], landmarks[2, :], c='y', marker='o')

        #plot landmarks from current frame in blue which have not been plotted before
        ax_3d_1.scatter(history_landmarks[-1][0, :], history_landmarks[-1][2, :], c='b', marker='o')

        #plot triangulated landmarks in red
        if triangulated_landmarks.size != 0:
            ax_3d_1.scatter(triangulated_landmarks[0, :], triangulated_landmarks[2, :], c='r', marker='o')

        #plot camera positions in green
        for i in range(len(history_R[:-1])):
            R = history_R[i]
            t = history_t[i]
            camera_position = -R.T @ t

            camera_x = camera_position[0]
            camera_z = camera_position[2]

            ax_3d_1.scatter(camera_x, camera_z, c='g', marker='x')

        # Plot the latest pose in red
        R = history_R[-1]
        t = history_t[-1]
        camera_position = -R.T @ t

        camera_x = camera_position[0]
        camera_z = camera_position[2]

        #set the limits of the plot to 4* the standard deviation of the landmarks in x and z direction
        #this is to make sure that the plot is not too zoomed in and doesnt explode if there is one mismatch

        x_std = np.std(np.abs(history_landmarks[-1][0, :]))
        z_std = np.std(np.abs(history_landmarks[-1][2, :]))

        x_mean = np.mean(history_landmarks[-1][0, :])
        z_mean = np.mean(history_landmarks[-1][2, :])

        ax_3d_1.set_xlim((-4 * x_std )+ x_mean, (4 * x_std) + x_mean)
        ax_3d_1.set_ylim((-4 * z_std) + z_mean, (4 * z_std) + z_mean)

        
        

        #ax_3d_1.set_xlim((-4 * x_std )+ camera_x, (4 * x_std) + camera_x)
        #ax_3d_1.set_ylim((-4 * z_std) + camera_z, (4 * z_std) + camera_z)

        




        ax_3d_1.scatter(camera_x, camera_z, c='r', marker='x')
        ax_3d_1.set_title('Top View')  
        
    def plot_2d(self, keypoints_history, triangulated_keypoints, ax):
       

        #add image to bottom subplot
        ax_2d = ax.add_subplot(223)
        #make sure the image is in color
        image_plotting = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    
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

        ax_2d.imshow(image_rgb)
        ax_2d.set_title('2D Plot')

    def plot_line_graph(self, history_landmarks, history_hidden_states, history_triangulated_landmarks, ax):

        #plots line graphs in a 4th subplot with 
        # 1) number of  tracked landmarks in each step
        # 2) number of newly triangulated landmarks in each step
        # 3) sum of landmarks in the hidden state

        ax_4 = ax.add_subplot(224)

        #plot number of tracked landmarks in each step
        tracked_landmarks = [landmarks.shape[1] for landmarks in history_landmarks]
        ax_4.plot(tracked_landmarks, label='Tracked Landmarks', color='b')

        #plot number of newly triangulated landmarks in each step
        triangulated_landmarks = []
        for landmarks in history_triangulated_landmarks:
            if landmarks.size == 0:
                triangulated_landmarks.append(0)
            else:
                triangulated_landmarks.append(landmarks.shape[1])
        ax_4.plot(triangulated_landmarks, label='Triangulated Landmarks', color='r')

        #plot sum of landmarks in the hidden state
        landmarks_count = []

        
        for candidate in history_hidden_states:

            #if candidate is an empty list:
            if len(candidate) == 0:
                landmarks_count.append(0)
            else:
                
                landmarks_count.append(candidate[0].shape[1])  

        #create a list with the sum of all values left of the current one for each element
        landmarks_sums = []
        for i in range(len(landmarks_count)):
            landmarks_sums.append(np.sum(landmarks_count[:i+1]))
        landmarks_sums = [0] + [0] + landmarks_sums
        ax_4.plot(landmarks_sums, label='Sum of Landmarks in Hidden State', color='g')



        ax_4.set_title('Line Graph')
        ax_4.legend(loc='lower left')

    def process_image(self, prev_image, image, keypoints_0, landmarks_0, descriptors_0, R_0, t_0, Hidden_state, history):
        


        #TODO: find memory leak in history
        #TODO: find possible memory leak in Hidden_state
        




        self.image = image
        self.prev_image = prev_image
        
        ###Track keypoints from last frame to this frame using KLT###
        
        keypoints_1, st = self.track_keypoints(prev_image, image, keypoints_0)
        st = st.reshape(-1)
        #remove keypoints that are not tracked
        landmarks_1 = landmarks_0[:, st == 1]
        descriptors_1 = descriptors_0[:, st == 1]

        ###estimate motion using PnP###
        R_1,t_1 = self.estimate_motion(keypoints_1, landmarks_1)

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


        ### Visualize the Results ###
        self.visualize_dashboard(triangulated_keypoints, triangulated_landmarks, history)


        self.current_image_counter += 1
        return keypoints_3, landmarks_3, descriptors_3, R_2, t_2, Hidden_state, history