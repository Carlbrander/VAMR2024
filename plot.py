from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import cv2
import numpy as np
import platform
import textwrap

class Plotter:
    def __init__(self, camera_positions, camera_position_bm, bootstrap_frames,args):
        self.fig = plt.figure(figsize=(18, 10))
        self.mng = plt.get_current_fig_manager()
        if platform.system() != 'Linux':
            self.mng.window.state('normal')
        self.gt_camera_position = camera_positions[2:-1]
        self.camera_position_bm = camera_position_bm
        self.bootstrap_frames = bootstrap_frames
        self.i = 0
        self.visualize_dashboard_1 = args.visualize_dashboard
        self.visualize_every_nth_frame = args.visualize_every_nth_frame
        self.threshold_angle = args.threshold_angle

        # Pause logic
        self.paused = [False]

    def visualize_dashboard(self, history, img, RMS, is_benchmark, current_iteration):
        
        # Clear the figure to update it
        self.fig.clf()
        self.i = current_iteration

        original_image = img.copy()

        # Plot Dashboard
        self.plot_3d(history.landmarks, history, history.triangulated_landmarks[-1])
        #self.plot_top_view(history, history.landmarks, history.R, history.t, history.triangulated_landmarks[-1], self.fig)
        self.plot_2d(img, history)
        self.plot_line_graph(history.landmarks, history.Hidden_states, history.triangulated_landmarks, self.fig)
        #self.plot_text(img, history, current_iteration)
        #self.plot_top_view__constant_zoom(history, history.landmarks, history.R, history.t, history.triangulated_landmarks[-1], self.fig)


        bins = np.linspace(0, 2*self.threshold_angle, 51)  # 50 bins between 0 and 20 degrees
        #remove all angles under the threshold from angles_before
        history.angles_before[-1] = [angle for angle in history.angles_before[-1] if angle > history.threshold_angles[-1]]

        #plot histogram of annlge above
        ax_3d_2 = self.fig.add_subplot(236)
        ax_3d_2.hist(history.angles_after[-1], bins=bins, color='r', alpha=0.7)
        ax_3d_2.hist(history.angles_before[-1], bins=bins, color='b', alpha=0.7)
        ax_3d_2.set_title('Histogram of Angles In all Current Hidden States summed up')
        ax_3d_2.set_xlabel('Angle in radian')
        ax_3d_2.set_ylabel('Frequency')
        ax_3d_2.set_xlim(0, 2*self.threshold_angle)
        ax_3d_2.set_xticks(np.arange(0, 2*self.threshold_angle, 2*self.threshold_angle/10))
     
        #make x text tick vertical
        plt.xticks(rotation=90)






        ##Plot another RGB image this time with all hidden states colord by age

        #add image to bottom subplot
        ax_2d = self.fig.add_subplot(233)
        #make sure the image is in color
        image_plotting = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        #get newest frame number
        newest_frame = history.Hidden_states[-1][7]

        #plot all hidden states in different colors
        #older ones in red, newer ones in green
        #create color map

        oldest_frame = np.min([candidate[7] for candidate in history.current_Hidden_state if len(candidate) > 0])
        
        ### COLOR MAP BY HOW NEW THE HIDDEN STATE IS ###
        
        #color_map = np.linspace(0, 255, newest_frame+1-oldest_frame)
#
        #for candiate in history.current_Hidden_state:
        #    if len(candiate) == 0:
        #        continue
        #    if len(candiate[3]) == 0:
        #        continue
        #    for keypoint in candiate[3].T:
        #        center = tuple(keypoint.astype(int))
#
        #        cv2.circle(image_plotting, center, 3, (255-int(color_map[candiate[7]-oldest_frame]), int(color_map[candiate[7]-oldest_frame]), 0), -1)
      #
        #ax_2d.imshow(image_plotting)
        #ax_2d.set_title('2D Plot All Hidden state Keypoints (Red = Old, Green = New)')

        ### COLOR MAP BY ANGLE OF POINT ###

        angles_and_keypoints = history.angles_and_keypoints[-1]
        #angles and keypoints is a list of arrays with the angle and the keypoint

       
        #colors red to green
        #angle of threshold angle = 255 (red)
        #angle of 0 = 0 (green)
        
        for angle, keypoint in angles_and_keypoints:
            
            center = tuple(keypoint.astype(int))
            green_color = 255-int(angle*255/self.threshold_angle)
            red_color = int(angle*255/self.threshold_angle)
            cv2.circle(image_plotting, center, 3, (red_color, green_color, 0), -1)
        ax_2d.imshow(image_plotting)
        ax_2d.set_title(f'2D Plot All Hidden states (Red = Angle {self.threshold_angle}, Green = Angle 0)')




        ###COLOR = ANGLE OF POINTS BUT FROM TOP VIEW ###

        ax_2d_2 = self.fig.add_subplot(232)


        ax_2d_2.set_xlabel('X')
        ax_2d_2.set_ylabel('Z')
        ax_2d_2.set_aspect('equal', adjustable='datalim')

        #angles_and_landmarks_r_t.append([angle, landmark, candidate[1], candidate[2], candidate[4], candidate[5]])

        landmarks_plot = np.array([landmark for angle, landmark, r_0, t_0, r_1, t_1 in history.angles_and_landmarks_r_t[-1]]).T
        #plot landmarks from current frame in blue which have not been plotted before
        
        for angle, landmark, r_0, t_0, r_1, t_1 in history.angles_and_landmarks_r_t[-1]:
            
            center = tuple(landmark.astype(int))
            green_color = np.clip(255-int(angle*255/self.threshold_angle), 0, 255)
            red_color = np.clip(int(angle*255/self.threshold_angle), 0, 255)

            ax_2d_2.scatter(landmark[0], landmark[2], c=[(red_color/255, green_color/255, 0)], marker='o', s = 2)

        camera_x = [point[0] for point in history.camera_position]
        camera_z = [point[2] for point in history.camera_position]
        camera_x_gt = [point[0] for point in self.gt_camera_position[:len(history.camera_position)]]
        camera_z_gt = [point[1] for point in self.gt_camera_position[:len(history.camera_position)]]
        ax_2d_2.scatter(camera_x, camera_z, c='g', marker='x')
        ax_2d_2.plot(camera_x_gt, camera_z_gt, 'k-', label='Ground Truth Trajectory')
        ax_2d_2.legend()


        # Plot the latest pose in red
        ax_2d_2.scatter(history.camera_position[-1][0], history.camera_position[-1][2], c='r', marker='x')



      
        #set the limits of the plot to 2* the standard deviation of the landmarks in x and z direction

        if len(landmarks_plot) == 0:
            ax_2d_2.set_xlim(-1, 1)
            ax_2d_2.set_ylim(-1, 1)
        else:

            #remove any landmarks that are too far away from the mean
            landmarks_plot = landmarks_plot[:, np.all(np.abs(landmarks_plot - np.mean(landmarks_plot, axis=1)[:, None]) < 2 * np.std(landmarks_plot, axis=1)[:, None], axis=0)]

            x_std = np.std(np.abs(landmarks_plot[0, :]))
            z_std = np.std(np.abs(landmarks_plot[2, :]))

            x_mean = np.mean(landmarks_plot[0, :])
            z_mean = np.mean(landmarks_plot[2, :])

            ax_2d_2.set_xlim((-2 * x_std )+ x_mean, (2 * x_std) + x_mean)
            ax_2d_2.set_ylim((-2 * z_std) + z_mean, (2 * z_std) + z_mean)

        # Compute the camera's forward direction in world coordinates
        forward_vector = history.R[-1].T @ np.array([0, 0, 1])
        
        dx = forward_vector[0]
        dz = forward_vector[2]

        # Normalize the direction vector
        norm = np.sqrt(dx**2 + dz**2)
        dx /= norm
        dz /= norm

        #add arrow in the direction the camera is looking:
        ax_2d_2.quiver(camera_x[-1], camera_z[-1], dx, dz, color='r', pivot='tail')
        ax_2d_2.set_title(f'Top View of HIDDEN STATES colored by angle (red={self.threshold_angle}, green=0)')  


















        

        




        # Add text on a free space between subplots for tracking parameters
        self.fig.text(0.27, 0.5, f'Threshold Angle: {history.threshold_angles[-1]}', ha='center', va='center', fontsize=12)
        self.fig.text(0.27, 0.47, f'New Keypoints Detection: {history.num_keypoints[-1]}', ha='center', va='center', fontsize=12)
        color = 'green' if is_benchmark else 'black'
        self.fig.text(0.27, 0.44, f'RMS Trajectory: {RMS}', ha='center', va='center', fontsize=12, color=color)
        
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

    def plot_3d(self, history_landmarks, history, triangulated_landmarks):
        ax_3d = self.fig.add_subplot(231)
        
        # Plot estimated trajectory
        est_trans = np.array(history.camera_position)
        ax_3d.plot(est_trans[:, 0], est_trans[:, 2], 'r', marker='*', markersize=3, label='Estimated pose')
        
        # Plot ground truth trajectory if available
        if len(self.gt_camera_position) > 0:
            gt_trans = np.array(self.gt_camera_position[self.bootstrap_frames[0]+1:self.i-1])
            bm_trans = np.array(self.camera_position_bm[:len(history.camera_position)])
            ax_3d.plot(gt_trans[:, 0], gt_trans[:, 1], 'b', marker='*', markersize=3, label='Scaled GT')
            ax_3d.plot(bm_trans[:, 0], bm_trans[:, 1], 'y', marker='*', markersize=3, label='Benchmark')
        
        ax_3d.set_title('Estimated trajectory')
        ax_3d.axis('equal')
        ax_3d.legend(fontsize=8, loc='best')
        

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
        ax_3d_1 = ax.add_subplot(232)
        ax_3d_1.set_xlabel('X')
        ax_3d_1.set_ylabel('Z')
        ax_3d_1.set_aspect('equal', adjustable='datalim')

        #plot old landmarks from the history in yellow until previous frame
        for landmarks in history_landmarks[max(-20,-len(history_landmarks)):-1]:
            ax_3d_1.scatter(landmarks[0, :], landmarks[2, :], c='y', marker='o', s = 2)

        #plot landmarks from current frame in blue which have not been plotted before
        ax_3d_1.scatter(history_landmarks[-1][0, :], history_landmarks[-1][2, :], c='b', marker='o', s = 2)

        #plot triangulated landmarks in red
        if isinstance(triangulated_landmarks, np.ndarray) and triangulated_landmarks.size != 0:
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

        ax_3d_1.set_xlim((-4 * x_std )+ x_mean, (4 * x_std) + x_mean)
        ax_3d_1.set_ylim((-4 * z_std) + z_mean, (4 * z_std) + z_mean)

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
        ax_3d_1.set_title('Top View')  
        
    def plot_2d(self, img, history):
        
        triangulated_keypoints = history.triangulated_keypoints[-1]
        keypoints_history = history.keypoints

        #add image to bottom subplot
        ax_2d = self.fig.add_subplot(234)
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
            cv2.circle(image_plotting, center, 3, (255, 0, 0), -1)

        #plot new keypoints in red
        for kp in triangulated_keypoints.T:
            center = tuple(kp.astype(int))
            cv2.circle(image_plotting, center, 2, (0, 0, 255), -1)

        image_rgb = cv2.cvtColor(image_plotting, cv2.COLOR_BGR2RGB)

        ax_2d.imshow(image_rgb)
        ax_2d.set_title('2D Plot')

    def plot_line_graph(self, history_landmarks, history_hidden_states, history_triangulated_landmarks, ax):

        #plots line graphs in a 4th subplot with 
        # 1) number of  tracked landmarks in each step
        # 2) number of newly triangulated landmarks in each step
        # 3) sum of landmarks in the hidden state

        ax_4 = ax.add_subplot(235)

        #plot number of tracked landmarks in each step
        tracked_landmarks = [landmarks.shape[1] for landmarks in history_landmarks]
        ax_4.plot(tracked_landmarks, label='Tracked Landmarks', color='b')

        #plot number of newly triangulated landmarks in each step
        triangulated_landmarks = []
        for landmarks in history_triangulated_landmarks[:-1]:
            if landmarks.size == 0:
                triangulated_landmarks.append(0)
            else:
                triangulated_landmarks.append(landmarks.shape[1])
        ax_4.plot(triangulated_landmarks, label='Triangulated Landmarks', color='r')

        #plot sum of landmarks in the hidden state
        landmarks_count = []

        
        for candidate in history_hidden_states[:-1]:

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
