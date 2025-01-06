import cv2
import numpy as np
import matplotlib.pyplot as plt

#solution scripts from exercise 3 for feature detection and matching using shi-tomasi
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
        self.num_keypoints = args.num_keypoints
        self.ds = args.ds

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
            # Detect keypoints using Shi-Tomasi
            keypoints = cv2.goodFeaturesToTrack(gray, maxCorners = 1000,
                                    qualityLevel = 0.01,
                                    minDistance = 7,
                                    blockSize = 2)
            
            keypoints = keypoints.reshape(-1, 2).T
            #switch x and y coordinates of the keypoints
            keypoints = keypoints[[1, 0], :]
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
    
          
        retval, rotation_vector, translation_vector, inliers = \
            cv2.solvePnPRansac(
                    landmarks_1.astype(np.float32).T, 
                    keypoints_1.astype(np.float32).T, 
                    self.K, 
                    distCoeffs=None,
                    iterationsCount=200000,
                    reprojectionError=2.0,
                    confidence=0.9999)
        
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

        P1 = P1.astype(np.float32)
        P2 = P2.astype(np.float32)


        keypoints_1 = keypoints_1.astype(np.float32)
        keypoints_2 = keypoints_2.astype(np.float32)


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

      
        keypoints_1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_image,
        image,
        keypoints_0.T.astype(np.float32),
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 0.03))



        
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

            # Vectorized distance calculation between keypoints
            dist_matrix = np.linalg.norm(
                newest_Hidden_state[3][:, :, np.newaxis] - candidate[3][:, np.newaxis, :], axis=0
            )

            # Find indices where distance is less than the suppression radius
            close_indices = np.any(dist_matrix < self.nonmaximum_suppression_radius, axis=1)

            # Update indices to keep
            indices_to_keep[close_indices] = False
           
        # Apply the mask once after the loop
        newest_Hidden_state[0] = newest_Hidden_state[0][:, indices_to_keep]
        newest_Hidden_state[3] = newest_Hidden_state[3][:, indices_to_keep]
        newest_Hidden_state[6] = newest_Hidden_state[6][:, indices_to_keep]

        Hidden_state[-1] = newest_Hidden_state

        return Hidden_state

    def triangulate_new_landmarks(self, Hidden_state,history):
        new_keypoints = []
        new_descriptors = []
        new_landmarks = []

        all_angles = []
        all_angles_after = []

        angles_and_keypoints = []
        angles_and_landmarks_r_t = []

       
        if Hidden_state:
            for candidate_i, candidate in enumerate(Hidden_state[:-1]):
                angles = []  # Reset angles for each candidate
                to_delete_index = []

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

                for i,landmark in enumerate(landmarks.T):
                    # Calculate bearing angle between the landmark and both camera views
                    angle = self.calculate_angle(landmark,candidate[1],candidate[4], candidate[2], candidate[5])
                    angles.append(angle)   
                    all_angles.append(angle)
                    all_angles_after.append(angle)
                    angles_and_keypoints.append([angle, candidate[3][:, i]])
                    angles_and_landmarks_r_t.append([angle, landmark, candidate[1], candidate[2], candidate[4], candidate[5]])
                    #sort by biggest angle first
                angles = np.array(angles)
                angles = np.sort(angles)[::-1]

                # If angle > threshold, add to lists
                for idx, angle in enumerate(angles):
                    if angle >= self.threshold_angle and len(new_keypoints) < self.landmarks_allowed:
                        new_keypoints.append(candidate[3][:, idx])
                        new_descriptors.append(candidate[6][:, idx])
                        new_landmarks.append(landmarks[:, idx])

                        #remove the keypoints in their hidden state if the are triangulated with large enough angle
                        to_delete_index.append(idx)

                        all_angles_after.remove(angle)
                        
                # Remove keypoints that were triangulated with large enough angle
                candidate[0] = np.delete(candidate[0], to_delete_index, axis=1)
                candidate[3] = np.delete(candidate[3], to_delete_index, axis=1)
                candidate[6] = np.delete(candidate[6], to_delete_index, axis=1)


                history.texts.append(f"Candidate Nr. {candidate[7]}: mean angle: {round(np.mean(angles),4)} std:  {round(np.std(angles),4)}, max: {round(np.max(angles),4)}, added: {len(to_delete_index)}, Nr. of elements after: {len(angles)}, baseline: {baseline}")

        # Convert lists to numpy arrays
        new_keypoints = np.array(new_keypoints).T
        new_landmarks = np.array(new_landmarks).T
        new_descriptors = np.array(new_descriptors).T

        #create histogram of all angles before removing the ones with large enough angles
        history.angles_before.append(all_angles)
        history.angles_after.append(all_angles_after)
        history.angles_and_keypoints.append(angles_and_keypoints)
        history.angles_and_landmarks_r_t.append(angles_and_landmarks_r_t)


        return new_keypoints, new_landmarks, new_descriptors, history

    def calculate_angle(self, landmark, R_1, R_2,t_1, t_2):
        #get the direction vector from the first observation camera pose to the landmark
        #direction_1 = landmark - t_1.flatten()
        #get the direction vector from the current camera pose to the landmark
        #direction_2 = landmark - t_2.flatten()

        #get the direction vector from the first observation camera pose to the landmark
        direction_1 = landmark + (R_1.T @ t_1).flatten()
        #get the direction vector from the current camera pose to the landmark
        direction_2 = landmark + (R_2.T @ t_2).flatten()

        #normalize the vectors
        direction_1 = direction_1 / np.linalg.norm(direction_1)
        direction_2 = direction_2 / np.linalg.norm(direction_2)

        #get the angle between the two vectors
        angle = np.arccos(np.clip(np.dot(direction_1, direction_2), -1.0, 1.0))

        return angle


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
        history.texts.append(f"Number of freshly Detected Keypoints: {new_keypoints.shape[1]}")



        history.texts.append(f"Number of new keypoints after NMS added to latest Hidden State: {new_keypoints.shape[1]}")


        # Add new keypoints & descriptors to the Hidden_state
        Hidden_state.append([new_keypoints, R_1, t_1.reshape(3,1), new_keypoints, R_1, t_1.reshape(3,1), new_descriptors, self.current_image_counter]) 
        
        

        #Remove hidden state that are empty lists
        Hidden_state = [candidate for candidate in Hidden_state if len(candidate) > 0]

        #Remove hidden state candidates that have less than 4 keypoints
        Hidden_state = [candidate for candidate in Hidden_state if candidate[0].shape[1] >= 4]

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
        

        #check each hidden state if there are any candidates 
        #Remove hidden state that are empty lists
        Hidden_state = [candidate for candidate in Hidden_state if len(candidate) > 0]

        #Remove hidden state candidates that have less than 4 keypoints
        Hidden_state = [candidate for candidate in Hidden_state if candidate[0].shape[1] >= 4]



        #### If there are enough keypoints tracked already we don't continue with the adding new landmarks and leave
        #### Them in the hidden state for later.


        #if landmarks_1.shape[1] > 500:
        #    return keypoints_1, landmarks_1, descriptors_1, Hidden_state, np.array([]), np.array([]), np.array([])
        
        #elif landmarks_1.shape[1] <= 500:
        self.landmarks_allowed = 10000#500 - landmarks_1.shape[1]








    
        #print cummulative number of keypoints in hidden state
        sum_hidden_state_landmarks = 0
        for candidate in Hidden_state:
            sum_hidden_state_landmarks += candidate[0].shape[1]
        history.texts.append(f"Number of Keypoints in Hidden State bafter removing too small candidates: {sum_hidden_state_landmarks}")
                
 
      
        ### Triangulate new Landmarks ###
        triangulated_keypoints, triangulated_landmarks, triangulated_descriptors, history = self.triangulate_new_landmarks(Hidden_state, history)
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
        
        if len(triangulated_landmarks) > 0:
            history.texts.append(f"Number of the triangulated_landmarks after NMS: {triangulated_landmarks.shape[1]}")
        else:
            history.texts.append(f"Number of the triangulated_landmarks after NMS: is zero")

     
        if len(triangulated_landmarks) > 0:
            history.texts.append(f"Number of the triangulated_landmarks after statistical_filtering: {triangulated_landmarks.shape[1]}")
        else:
            history.texts.append(f"Number of the triangulated_landmarks after statistical_filtering: is zero")
        
        if triangulated_landmarks.size == 0:
            return keypoints_1, landmarks_1, descriptors_1, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors
        

      
        history.texts.append(f"Number of the triangulated_landmarks after reducing number: {triangulated_landmarks.shape[1]}")

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

    def process_image(self, prev_image, image, keypoints_0, landmarks_0, descriptors_0, R_0, t_0, Hidden_state, history):

        self.image = image
        self.prev_image = prev_image
        
        ###Track keypoints from last frame to this frame using KLT###
        history.texts.append(f"-6. keypoints_0.shape at the beginning of process_image: {keypoints_0.shape}")
        keypoints_1, st = self.track_keypoints(prev_image, image, keypoints_0)
        history.texts.append(f"-5. keypoints_1.shape after track_keypoints : {keypoints_1.shape}")
        st = st.reshape(-1)

        #print percentage of correctly tracked points
        history.texts.append(f"Percentage of correctly tracked points: {np.sum(st) / len(st) * 100}%")
        #remove keypoints that are not tracked
        landmarks_1 = landmarks_0[:, st == 1]
        descriptors_1 = descriptors_0[:, st == 1]
        history.texts.append(f"-4. landmarks_1.shape after track_keypoints : {landmarks_1.shape}")

        ###estimate motion using PnP###
        R_1,t_1, inliers = self.estimate_motion(keypoints_1, landmarks_1)
        # Use inliers to filter out outliers from keypoints and landmarks

        if inliers is None:
            print("No inliers found for motion estimation, pipeline failed")
            exit()

        inliers = inliers.flatten()
        keypoints_1 = keypoints_1[:, inliers]
        landmarks_1 = landmarks_1[:, inliers]
        history.texts.append(f"-3. landmarks_1.shape after inliers filtering : {landmarks_1.shape}")
        descriptors_1 = descriptors_1[:, inliers]
        history.camera_position.append(-R_1.T @ t_1)

        ###Adapt Angle Threshold and Number of Keypoints dynamically###
        #tries to keep it between 200 an 500 keypoints

        
        if self.ds == 1 or self.ds == 0:
            self.threshold_angle = np.maximum(0.1,np.minimum(0.3, landmarks_1.shape[1] / 2800))
        else:
            self.threshold_angle = np.maximum(0.1,np.minimum(0.3, landmarks_1.shape[1] / 1400))




        # Adapt Parameter for Landmark Detection dynamically #
        history.texts.append(f"-2. self.num_keypoints: {self.num_keypoints}")
        history.texts.append(f"-1. self.threshold_angle: {self.threshold_angle}")

        keypoints_2, landmarks_2, descriptors_2, Hidden_state, triangulated_keypoints, triangulated_landmarks, triangulated_descriptors = \
            self.add_new_landmarks(keypoints_1, landmarks_1, descriptors_1, R_1, t_1, Hidden_state, history)


        ###Update History###
        history.keypoints.append(keypoints_2)
        history.landmarks.append(landmarks_2)
        history.R.append(R_1)
        history.t.append(t_1)
        
        history.triangulated_landmarks.append(triangulated_landmarks)
        history.triangulated_keypoints.append(triangulated_keypoints)
        history.threshold_angles.append(self.threshold_angle)
        history.num_keypoints.append(self.num_keypoints)
       
        #This is only for the line plot plotting:
        history.Hidden_states = Hidden_state

        history.current_Hidden_state = Hidden_state

       
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


        return keypoints_2, landmarks_2, descriptors_2, R_1, t_1, Hidden_state, history
