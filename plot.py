from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import cv2
import numpy as np
import platform
import textwrap

class Plotter:
    def __init__(self, camera_positions, camera_position_bm, bootstrap_frames):
        self.fig = plt.figure(figsize=(18, 10))
        self.mng = plt.get_current_fig_manager()
        if platform.system() != 'Linux':
            self.mng.window.state('normal')
        self.gt_camera_position = camera_positions[2:-1]
        self.camera_position_bm = camera_position_bm
        self.bootstrap_frames = bootstrap_frames
        self.i = 0

        # Pause logic
        self.paused = [False]

    def visualize_dashboard(self, history, img, RMS, is_benchmark, current_iteration):
        # Clear the figure to update it
        self.fig.clf()
        self.i = current_iteration

        # Plot Dashboard
        self.plot_3d(history.landmarks, history, history.triangulated_landmarks[-1])
        self.plot_top_view(history, history.landmarks, history.R, history.t, history.triangulated_landmarks[-1], self.fig)
        self.plot_2d(img, history)
        self.plot_line_graph(history.landmarks, history.Hidden_states, history.triangulated_landmarks, self.fig)
        self.plot_text(img, history, current_iteration)
        self.plot_top_view__constant_zoom(history, history.landmarks, history.R, history.t, history.triangulated_landmarks[-1], self.fig)

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
        plt.pause(0.01)
        plt.savefig("output/output_{0:06}.png".format(len(history.camera_position)))

        # Pause loop
        while self.paused[0]:
            plt.pause(0.1)

    def plot_3d(self,history_landmarks, history, triangulated_landmarks):

        ax_3d = self.fig.add_subplot(231)
        
        # Plot estimated trajectory
        est_trans_x = [point[0] for point in history.camera_position]
        est_trans_z = [point[2] for point in history.camera_position]
        ax_3d.plot(est_trans_x, est_trans_z, 'r', marker='*', markersize=3, label='Estimated pose')
        
        # Plot ground truth trajectory if available
        if len(self.gt_camera_position) > 0:
            gt_x = [point[0] for point in self.gt_camera_position[self.bootstrap_frames[0]+1:self.i-1]]
            gt_z = [point[1] for point in self.gt_camera_position[self.bootstrap_frames[0]+1:self.i-1]]
            bm_x = [point[0] for point in self.camera_position_bm[:len(history.camera_position)]]
            bm_z = [point[1] for point in self.camera_position_bm[:len(history.camera_position)]]
            ax_3d.plot(gt_x, gt_z, 'b', marker='*', markersize=3, label='Scaled GT')
            ax_3d.plot(bm_x, bm_z, 'y', marker='*', markersize=3, label='Benchmark')
        
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
