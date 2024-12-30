import os
import numpy as np
class Benchmarker:
    def __init__(self, gt_camera_position, ds):
        if ds == 0:
            self.name = "kitti"

        self._save_path = f"data/{self.name}/"
        self.state = np.empty((0, 4))  # Initialisiere als leeres 2D-Array
        self._avg_execution_time = 0
        self._frame = 0
        self.gt_camera_position = gt_camera_position
        try:
            self.state_benchmark = np.loadtxt(os.path.join(self._save_path, "RMS.txt"))
            self.camera_position_bm = self.state_benchmark[:, [1, 3]]
        except OSError:
            self.state_benchmark = np.empty((0,))
            self.camera_position_bm = gt_camera_position

    def process(self, camera_position, i):
        # Calculate RMS
        self._frame = i
        RMS = self.getTrajRMS(
            self.gt_camera_position[:len(camera_position)],
            [pos[:2] for pos in camera_position]
        )
        camera_position_array = np.array(camera_position[i-3]).flatten()  # Umwandlung in NumPy-Array und Flatten
        self.state = np.append(self.state, np.hstack(([RMS], camera_position_array)).reshape(1, -1), axis=0)

        try:
            RMS_benchmark = self.state_benchmark[self._frame-3][0]
        except IndexError:
            RMS_benchmark = 100
        if RMS < 0.999*RMS_benchmark:
            is_benchmark = True
            fmt = ' '.join(['%d'] + ['%.6f'] * (self.state.shape[1] - 1))
            np.savetxt(os.path.join(self._save_path, "RMS.txt"), self.state, fmt=fmt)
        else:
            is_benchmark = False

        return RMS, is_benchmark


    def getTrajRMS(self, gt, est):
        # Compute the Root Mean Square Error of the trajectory
        assert len(gt) == len(est)
        N = len(gt)
        sum = 0
        for i in range(N):
            sum += np.linalg.norm(gt[i] - est[i])
        return sum / N