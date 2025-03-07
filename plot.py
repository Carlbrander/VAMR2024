from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import cv2
import numpy as np
import textwrap

class Plotter:
    def __init__(self, camera_positions, camera_position_bm, bootstrap_frames,args):
        self.fig = plt.figure(figsize=(18, 10))
        # self.mng = plt.get_current_fig_manager()
        # if platform.system() != 'Linux':
        #     self.mng.window.state('normal')
        self.gt_camera_position = camera_positions[2:-1]
        self.camera_position_bm = camera_position_bm
        self.bootstrap_frames = bootstrap_frames
        self.i = 0
        self.visualize_dashboard_1 = args.visualize_dashboard
        self.visualize_every_nth_frame = args.visualize_every_nth_frame
        self.threshold_angle = args.threshold_angle

        # Pause logic
        self.paused = [False]
        self.ds = args.ds

    def visualize_dashboard(self, history, img, RMS, is_benchmark, current_iteration, initial_landmarks):
        if history.triangulated_landmarks[-1].size == 0:
            triangulated_landmarks = initial_landmarks
        else:
            triangulated_landmarks = history.triangulated_landmarks[-1]
        # Clear the figure to update it
        self.fig.clf()
        self.i = current_iteration

        # Plot Dashboard
        self.plot_full_traj(history)
        self.plot_top_view(history, history.landmarks, history.R, history.t, triangulated_landmarks, self.fig)
        self.plot_2d(img, history)
        self.plot_line_graph(history.landmarks, history.Hidden_states, history.triangulated_landmarks, self.fig)
       # self.plot_text(img, history, current_iteration)

        # Add text on a free space between subplots for tracking parameters
       # self.fig.text(0.27, 0.5, f'Threshold Angle: {history.threshold_angles[-1]}', ha='center', va='center', fontsize=12)
       # self.fig.text(0.27, 0.47, f'New Keypoints Detection: {history.num_keypoints[-1]}', ha='center', va='center', fontsize=12)
        color = 'green' if is_benchmark else 'black'
       # self.fig.text(0.27, 0.44, f'RMS Trajectory: {RMS}', ha='center', va='center', fontsize=12, color=color)
        
        # Draw the pause button
        pause_button_ax = plt.axes([0.45, 0.01, 0.1, 0.05])
        self.pause_button = Button(pause_button_ax, 'Pause/Resume')
        self.pause_button.on_clicked(self.toggle_pause)

        self.fig.canvas.draw()
        if self.visualize_dashboard_1:
            plt.show(block=False)
            plt.pause(0.00000000001)
            plt.savefig("output/output_{0:06}.png".format(len(history.camera_position)))
        else:
            
            plt.savefig("output/output_{0:06}.png".format(len(history.camera_position)))

        # Pause loop
        while self.paused[0]:
            plt.pause(0.1)

    def plot_full_traj(self, history):
        ax_3d = self.fig.add_subplot(221)
        
        # Plot estimated trajectory
        est_trans = np.array(history.camera_position)
        ax_3d.plot(est_trans[:, 0], est_trans[:, 2], 'g', marker='*', markersize=3, label='Estimated Trajectory')
        
        # Plot ground truth trajectory if available
        if len(self.gt_camera_position) > 0:
            gt_trans = np.array(self.gt_camera_position[self.bootstrap_frames[0]+1:self.i-1])
            bm_trans = np.array(self.camera_position_bm[:len(history.camera_position)])

            #only use first and third element of est_trans and remove the last 1 dimension
            est_trans = est_trans.reshape(-1,3)[:, [0, 2]]

            #get last 40 points from est_trans
            #est_trans_last_40 = est_trans[-200:]
            #get last 40 points from gt trans
            #gt_trans_last_40 = gt_trans[-200:]

            #transform the gt_trans_last_40 to match with the est_trans_last_40 and remember the transformation
            
            #M, _ = cv2.estimateAffinePartial2D(gt_trans_last_40, est_trans_last_40)
#
            #a, b = M[0,0], M[0,1]
            #scale = np.sqrt(a*a + b*b)
            #rotation = np.arctan2(b, a)
            #tx, ty = M[0,2], M[1,2]
#
            #ones = np.ones((gt_trans.shape[0], 1))
            #gt_hom = np.hstack([gt_trans, ones])
            ## Build full 3x3 matrix
            #M_full = np.vstack([M, [0,0,1]])
            #aligned_gt = (M_full @ gt_hom.T).T[:, :2]


            ax_3d.plot(gt_trans[:, 0], gt_trans[:, 1], 'black', marker='*', markersize=3, label='Scaled Ground Truth')
        
            #ax_3d.plot(aligned_gt[:, 0], aligned_gt[:, 1], 'black', marker='*', markersize=3, label='Scaled Ground Truth')
            # ax_3d.plot(bm_trans[:, 0], bm_trans[:, 1], 'y', marker='*', markersize=3, label='Benchmark')
        
        ax_3d.set_title('Full Trajectory')
        ax_3d.axis('equal')
        ax_3d.legend(fontsize=8, loc='upper right')
        

        # #get a set of history landmarks without the latest landmarks (as they are also in the history landmarks as duplicated likely)
        # historic_landmarks = []
        # for landmarks in history_landmarks[max(-5,-len(history_landmarks)):-1]:
        #     for landmark in landmarks.T:
        #         #check if it is in the latest landmarks
        #         if np.all(np.abs(landmark - history_landmarks[-1].T) < 0.1):
        #             continue
        #         historic_landmarks.append(landmark)

        # historic_landmarks = np.array(historic_landmarks).T
        # ax_3d.scatter(historic_landmarks[0, :], historic_landmarks[1, :], historic_landmarks[2, :], c='y', marker='o')





        # #for landmarks in history_landmarks[-len(history_landmarks):-1]:
        # #    ax_3d.scatter(landmarks[0, :], landmarks[1, :], landmarks[2, :], c='y', marker='o')


        
        # #plot landmarks from current frame in blue which have not been plotted before

        # #get a set of latest landmarks without the triangulated_landmarks
        # latest_landmarks = history_landmarks[-1][:, :]
        # if triangulated_landmarks.size != 0:
        #     latest_landmarks = latest_landmarks[:, :-triangulated_landmarks.shape[1]]
        # ax_3d.scatter(latest_landmarks[0, :], latest_landmarks[1, :], latest_landmarks[2, :], c='b', marker='o')
        # #ax.scatter(history_landmarks[-1][0, :], history_landmarks[-1][1, :], history_landmarks[-1][2, :], c='b', marker='o')

        # #plot triangulated landmarks in red
        # if triangulated_landmarks.size != 0:
        #     ax_3d.scatter(triangulated_landmarks[0, :], triangulated_landmarks[1, :], triangulated_landmarks[2, :], c='r', marker='o')

        # camera_x = [point[0] for point in history.camera_position]
        # camera_y = [point[1] for point in history.camera_position]
        # camera_z = [point[2] for point in history.camera_position]

        # ax_3d.scatter(camera_x, camera_y, camera_z, c='g', marker='x', s=100)


        # # Plot the latest pose in red
        # ax_3d.scatter(history.camera_position[-1][0], history.camera_position[-1][1], history.camera_position[-1][2], c='r', marker='x', s=100)


        ############################################################
       
    def plot_top_view(self, history, history_landmarks, history_R, history_t, triangulated_landmarks, ax):
        #on second subplot show a 2D plot as top view (X-Z plane) with all landmarks and cameras
        ax_3d_1 = self.fig.add_subplot(222)
        ax_3d_1.set_xlabel('X')
        ax_3d_1.set_ylabel('Z')
        ax_3d_1.set_aspect('equal', adjustable='datalim')

        #plot old landmarks from the history in yellow until previous frame
        for landmarks in history_landmarks[max(-2, -len(history_landmarks)):-1]:
            if landmarks.size != 0:

                x_vals_1 = landmarks[0, :]
                z_vals_1 = landmarks[2, :]
                x_std = np.std(np.abs(x_vals_1))
                z_std = np.std(np.abs(z_vals_1))

                x_mean = np.mean(landmarks[0, :])
                z_mean = np.mean(landmarks[2, :])

                distances = np.sqrt((x_vals_1 - x_mean)**2 + (z_vals_1 - z_mean)**2)

                dist_std = np.std(distances)
                threshold = dist_std



                # Filter landmarks within one standard deviation
                #mask = (np.abs(landmarks[0, :] - x_mean) <= x_std) & (np.abs(landmarks[2, :] - z_mean) <= z_std)
                mask = distances <= threshold
                filtered_landmarks = landmarks[:, mask]
            else:
                filtered_landmarks = np.array([[], [], []])

            # Plot filtered landmarks with a single label
            if filtered_landmarks.size != 0:
                ax_3d_1.scatter(filtered_landmarks[0, :], filtered_landmarks[2, :], color=(128/255, 0, 128/255), marker='o', s=2)
        ax_3d_1.scatter([], [], color=(128/255, 0, 128/255), marker='o', s=2, label='Tracked Landmarks')

        # #plot landmarks from current frame in blue which have not been plotted before
        # ax_3d_1.scatter(history_landmarks[-1][0, :], history_landmarks[-1][2, :], c='b', marker='o', s = 2)

        #plot triangulated landmarks in red
        # if isinstance(triangulated_landmarks, np.ndarray) and triangulated_landmarks.size != 0:
        #     ax_3d_1.scatter(triangulated_landmarks[0, :], triangulated_landmarks[2, :], c='r', marker='o', s = 4)


        camera_x = [point[0] for point in history.camera_position]
        camera_z = [point[2] for point in history.camera_position]
        camera_x_gt = [point[0] for point in self.gt_camera_position[:len(history.camera_position)]]
        camera_z_gt = [point[1] for point in self.gt_camera_position[:len(history.camera_position)]]

        camera_x_gt_last_40 = camera_x_gt[-40:]
        camera_z_gt_last_40 = camera_z_gt[-40:]

        assert len(camera_x_gt_last_40) == len(camera_z_gt_last_40)

        #get last 40 points from camera_x and camera_z
        camera_x_last_40 = camera_x[-40:]
        camera_z_last_40 = camera_z[-40:]
        if (len(camera_x_last_40) != 0 and len(camera_z_last_40) != 0) and self.ds != 1:
         
            #transform the camera_x_gt_last_40 and camera_z_gt_last_40 to match with the camera_x_last_40 and camera_z_last_40 and remember the transformation
            M, _ = cv2.estimateAffinePartial2D(np.array([camera_x_gt_last_40, camera_z_gt_last_40]).T, np.array([camera_x_last_40, camera_z_last_40]).T)
            if M is not None:
                a, b = M[0,0], M[0,1]
                scale = np.sqrt(a*a + b*b)
                rotation = np.arctan2(b, a)
                tx, ty = M[0,2], M[1,2]

                ones = np.ones((len(camera_x_gt_last_40), 1))
                gt_hom = np.hstack([np.array([camera_x_gt_last_40, camera_z_gt_last_40]).T, ones])
                # Build full 3x3 matrix
                M_full = np.vstack([M, [0,0,1]])
                aligned_gt = (M_full @ gt_hom.T).T[:, :2]

                ax_3d_1.plot(aligned_gt[:, 0], aligned_gt[:, 1], 'black', marker='*', markersize=3, label='Scaled Ground Truth')

        else: 
            ax_3d_1.plot(camera_x_gt_last_40, camera_z_gt_last_40, 'black', marker='*', markersize=3, label='Ground Truth Trajectory')


        ax_3d_1.scatter(camera_x[-40:], camera_z[-40:], c='g', marker='x', label='Estimated Trajectory')
        #ax_3d_1.plot(camera_x_gt[-20:], camera_z_gt[-20:], 'k-', label='Ground Truth Trajectory')


        # Plot the latest pose in red
        ax_3d_1.scatter(history.camera_position[-1][0], history.camera_position[-1][2], c='r', marker='x', label='Estimated Current Position')

        #set the limits of the plot to 4* the standard deviation of the landmarks in x and z direction
        #this is to make sure that the plot is not too zoomed in and doesnt explode if there is one mismatch

        if triangulated_landmarks.size != 0:

            x_vals = triangulated_landmarks[0, :]
            z_vals = triangulated_landmarks[2, :]
    
            x_std = np.std(np.abs(x_vals))
            z_std = np.std(np.abs(z_vals))

            x_mean = np.mean(x_vals)
            z_mean = np.mean(z_vals)

            distances = np.sqrt((x_vals - x_mean)**2 + (z_vals - z_mean)**2)

            dist_std = np.std(distances)
            threshold =  dist_std
    
            
            # Filter landmarks within one standard deviation
            #mask = (np.abs(triangulated_landmarks[0, :] - x_mean) <= x_std) & (np.abs(triangulated_landmarks[2, :] - z_mean) <= z_std)
            mask = distances <= threshold
            filtered_landmarks = triangulated_landmarks[:, mask]
        else:
            filtered_landmarks = np.array([[], [], []])

        # Plot filtered landmarks
        if filtered_landmarks.size != 0:
            ax_3d_1.scatter(filtered_landmarks[0, :], filtered_landmarks[2, :], color='lightgreen', marker='o', s=4, label='Added Landmarks')

        ax_3d_1.legend()
        # x_lim = (-1 * x_std) + x_mean, (1 * x_std) + x_mean
        # z_lim = (-1 * z_std) + z_mean, (1 * z_std) + z_mean
        # ax_3d_1.set_xlim(x_lim)
        # ax_3d_1.set_ylim(z_lim)

        #ax_3d_1.set_xlim((-4 * x_std )+ camera_x, (4 * x_std) + camera_x)
        #ax_3d_1.set_ylim((-4 * z_std) + camera_z, (4 * z_std) + camera_z)

        # Compute the camera's forward direction in world coordinates
        forward_vector = history_R[-1].T @ np.array([0, 0, 1])
        
        dx = forward_vector[0]
        dz = forward_vector[2]

        # Normalize the direction vector
        norm = np.sqrt(dx**2 + dz**2)
        dx /= norm
        dz /= norm

        #add arrow in the direction the camera is looking:
        ax_3d_1.quiver(camera_x[-1], camera_z[-1], dx, dz, color='r', pivot='tail')
        ax_3d_1.set_title('Trajectory of last 40 frames and landmarks') 
        
    def plot_2d(self, img, history):
        
        triangulated_keypoints = history.triangulated_keypoints[-1]
        keypoints_history = history.keypoints

        #add image to bottom subplot
        ax_2d = self.fig.add_subplot(223)
        #make sure the image is in color
        image_plotting = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    
        # #plot previous keypoints in yellow
        # for keypoints_from_history in keypoints_history[max(-10, -len(keypoints_history)):-1]:
        #     for kp in keypoints_from_history.T:
        #         center = tuple(kp.astype(int))
        #         cv2.circle(image_plotting, center, 3, (0, 255, 255), -1)
     
        #plot current keypoints blue
        for keypoints_from_history in keypoints_history[-1].T:
            center = tuple(keypoints_from_history.astype(int))
            cv2.circle(image_plotting, center, 3, (128, 0, 128), -1)

        #plot new keypoints in red
        for kp in triangulated_keypoints.T:
            center = tuple(kp.astype(int))
            cv2.circle(image_plotting, center, 2, (0, 255, 0), -1)

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
        ax_4.plot(tracked_landmarks[-20:], label='Tracked Landmarks', color=(128/255, 0, 128/255))

        #plot number of newly triangulated landmarks in each step
        triangulated_landmarks = []
        for landmarks in history_triangulated_landmarks[:-1]:
            if landmarks.size == 0:
                triangulated_landmarks.append(0)
            else:
                triangulated_landmarks.append(landmarks.shape[1])
        ax_4.plot(triangulated_landmarks[-20:], label='Added Landmarks', color='lightgreen')

        #plot sum of landmarks in the hidden state
        landmarks_count = []

        
        for candidate in history_hidden_states[:-1]:

            #if candidate is an empty list:
            if len(candidate) == 0:
                landmarks_count.append(0)
            else:
                
                landmarks_count.append(candidate[0].shape[1])  

        ax_4.set_xticks(range(20))  # Positionen der x-Achsen-Ticks von 0 bis 19


        ax_4.set_title('Line Graph')
        ax_4.legend(loc='lower left')


    def plot_text(self, img, history, current_iteration):
        ax = self.fig.add_subplot(233)
        # ax = self.fig.add_subplot(236)

        # Adding multi-line text at a specified location
        wrapped_texts = [wrap_text(text, width=90) for text in history.texts]
        text_str = '\n'.join(wrapped_texts)

        ax.text(-2, 9.5, text_str, fontsize=10, color='black', ha='left', va='top')
        # ax.text(0.2, -9.5, text_str, fontsize=12, color='black', ha='left', va='top')

        # Optional: Add labels and title
        ax.set_xlim([0, 15])
        ax.set_ylim([0, 10])
        ax.axis('off')
        ax.set_title('Plot with Multi-line Text')


        # ax.imshow(image_rgb)
        ax.set_title(f'Logs. Iteration={current_iteration}')

    def plot_top_view__constant_zoom(self, history, history_landmarks, history_R, history_t, triangulated_landmarks, ax):
        #on second subplot show a 2D plot as top view (X-Z plane) with all landmarks and cameras
        ax_3d_1 = ax.add_subplot(236)
        ax_3d_1.set_xlabel('X')
        ax_3d_1.set_ylabel('Z')
        ax_3d_1.set_aspect('equal', adjustable='datalim')

        #plot old landmarks from the history in yellow until previous frame
        for landmarks in history_landmarks[max(-20,-len(history_landmarks)):-1]:
            ax_3d_1.scatter(landmarks[0, :], landmarks[2, :], c='y', marker='o', s = 2)

        #plot landmarks from current frame in blue which have not been plotted before
        ax_3d_1.scatter(history_landmarks[-1][0, :], history_landmarks[-1][2, :], c='b', marker='o', s = 2)

        #plot triangulated landmarks in red
        if triangulated_landmarks.size != 0:
            ax_3d_1.scatter(triangulated_landmarks[0, :], triangulated_landmarks[2, :], c='r', marker='o', s = 4)


        camera_x = [point[0] for point in history.camera_position]
        camera_z = [point[2] for point in history.camera_position]
        camera_x_gt = [point[0] for point in self.gt_camera_position[:len(history.camera_position)]]
        camera_z_gt = [point[1] for point in self.gt_camera_position[:len(history.camera_position)]]
        ax_3d_1.scatter(camera_x, camera_z, c='g', marker='x')
        ax_3d_1.plot(camera_x_gt, camera_z_gt, 'k-', label='Ground Truth Trajectory')
        ax_3d_1.legend()


        # Plot the latest pose in red
        ax_3d_1.scatter(history.camera_position[-1][0], history.camera_position[-1][2], c='r', marker='x')

        #set the limits of the plot to 4* the standard deviation of the landmarks in x and z direction
        #this is to make sure that the plot is not too zoomed in and doesnt explode if there is one mismatch

        x_std = np.std(np.abs(history_landmarks[-1][0, :]))
        z_std = np.std(np.abs(history_landmarks[-1][2, :]))

        x_mean = np.mean(history_landmarks[-1][0, :])
        z_mean = np.mean(history_landmarks[-1][2, :])

        # ax_3d_1.set_xlim((-4 * x_std )+ x_mean, (4 * x_std) + x_mean)
        # ax_3d_1.set_ylim((-4 * z_std) + z_mean, (4 * z_std) + z_mean)

        #ax_3d_1.set_xlim((-4 * x_std )+ camera_x, (4 * x_std) + camera_x)
        #ax_3d_1.set_ylim((-4 * z_std) + camera_z, (4 * z_std) + camera_z)

        ax_3d_1.set_xlim((-4 * 4) + camera_x[-1], (4 * 4) + camera_x[-1])
        ax_3d_1.set_ylim((-2 * 4) + camera_z[-1], (6 * 4) + camera_z[-1])

        # Compute the camera's forward direction in world coordinates
        forward_vector = history_R[-1].T @ np.array([0, 0, 1])

        dx = forward_vector[0]
        dz = forward_vector[2]

        # Normalize the direction vector
        norm = np.sqrt(dx**2 + dz**2)
        dx /= norm
        dz /= norm

        #add arrow in the direction the camera is looking:
        ax_3d_1.quiver(camera_x[-1], camera_z[-1], dx, dz, color='r', pivot='tail')
        ax_3d_1.set_title('Top View: Constant Zoom')

    def toggle_pause(self, event):
        self.paused[0] = not self.paused[0]



def wrap_text(text, width, subsequent_indent='    '):
    """Wrap text to fit within a specified width with indentation for subsequent lines."""
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent=subsequent_indent)
    return '\n'.join(wrapper.wrap(text))
