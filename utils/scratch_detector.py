# we write the previous process as a ScratchDetector class
import cv2
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

class ScratchDetector:
    def __init__(self, video_path, pretrained_weights, save_path, plot_prediction=True, high_conf_class=True, device=None):
        """
        ScratchDetector class to detect, time and count scratching behaviour in mice
        Args:
            video_path (str): path to the video file
            pretrained_weights (str): path to the pretrained weights file
            save_path (str): path to save the results in a CSV file
            plot_prediction (bool): plot the bounding boxes, keypoints, and angles
            high_conf_class (bool): use the class with the highest confidence
            device (torch.device): device to run the model on (GPU or CPU)
            
        Attributes:
            scratchings (dict): dictionary to store the scratching events
            keypoints_scratching (list): list to store the keypoints of scratching events
            behaviours_scratching (list): list to store the behaviours of scratching events
            fps (int): frames per second of the video
            num_frames (int): number of frames to process
            i (int): frame index
        
        Methods:
            device_diagnosis: detect the device (GPU or CPU)
            initialize_csv: initialize the CSV file to write in the scratching results
            initialize_plots: initialize the plots for the video and the angle/distance/speed curves
            smooth_data: smooth data using Gaussian filter
            plot_results: plot the results (bounding boxes, keypoints, etc.)
            frame_index2time: convert frame index to time in minutes, seconds, and frames
            compute_angle: compute the angle between three points (p1, p2, p3)
            count_local_minima: count the number of local minima in the data
            validate_scratching: validate if a set of keypoints and behaviors form a valid scratching event
            evaluate_scratching: evaluate a scratching event and calculate side, times, and intensity
            detect_scratching: detect when scratching starts, continues, or ends
            merge_scratching: merge the current scratching behaviour with the previous one
            process_video: main video processing loop    
        """
        self.video_path = video_path
        self.pretrained_weights = pretrained_weights
        self.save_path = save_path
        self.plot_prediction = plot_prediction
        self.high_conf_class = high_conf_class
        self.device = device or self.device_diagnosis(verbose=False)  
        self.model = YOLO(pretrained_weights)  # Load the YOLO model
        self.scratchings = {
            'start': [], 'end': [], 'duration': [], 'gaps': [],
            'side': [], 'times': [], 'intensity': []
        }
        self.keypoints_scratching = []
        self.behaviours_scratching = []
        self.fps = None
        self.num_frames = None
        self.i = 0
        self.initialize_csv()
        if self.plot_prediction:
            # self.initialize_plots()
            self.initialize_plots_together()

    def device_diagnosis(self, verbose=True):
        """ Detect the device (GPU or CPU) """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f'Using device: {device}')
        return device

    def initialize_csv(self):
        """ Initialize the CSV file to write in the scratching results """
        with open(self.save_path, 'w', newline='') as f:
            f.write('id,start,end,duration,gap,side,times,intensity\n')
    
    def initialize_plots(self):
        """ Initialize the plots for the video and the angle/distance/speed curves """
        plt.ion() # Turn on interactive mode
        self.fig1, self.ax1 = plt.subplots(figsize=(12, 9))
        self.fig2, self.ax2 = plt.subplots(figsize=(12, 6))
        self.line_right, = self.ax2.plot([], [], label='angle_right', color='red')
        self.line_left, = self.ax2.plot([], [], label='angle_left', color='blue')
        self.ax2.legend()
        self.ax2.xaxis.set_visible(False)
        self.ax2.set_ylim(0, 90)
        self.ax2.set_title('Angle')
        self.xdata, self.right_angle_data, self.left_angle_data = [], [], []

    def initialize_plots_together(self):
        """ Initialize the 3 subplots for the video, the angle, and the table showing the results """
        plt.ion()
        self.fig = plt.figure(figsize=(21, 9))
        self.ax1 = plt.subplot2grid((3, 7), (0, 0), rowspan=3, colspan=4)
        self.ax2 = plt.subplot2grid((3, 7), (0, 4), rowspan=1, colspan=3)
        self.ax3 = plt.subplot2grid((3, 7), (1, 4), rowspan=2, colspan=3)
        self.ax1.axis('off')
        self.ax2.xaxis.set_visible(False)
        self.ax2.set_ylim(0, 90)
        self.line_right, = self.ax2.plot([], [], label='angle_right', color='red')
        self.line_left, = self.ax2.plot([], [], label='angle_left', color='blue')
        self.ax2.legend()
        self.xdata, self.right_angle_data, self.left_angle_data = [], [], []
        self.ax3.axis('off')
        self.blank_row = ['', '', '', '', '', '', '', '']
        self.tabledata = [['id', 'start', 'end', 'duration', 'gap', 'side', 'times', 'intensity']]
        #append 18 blank rows to the table
        self.tabledata.extend([self.blank_row for _ in range(24)])
        # table lines set to light grey
        self.ax3.table(cellText=self.tabledata, cellLoc='center', loc='center', fontsize=16)

        
    def smooth_data(self, data):
        """ Smooth data using Gaussian filter """
        return gaussian_filter(data, sigma=0.5)

    def plot_results(self, r, keypoints):
        """ Plot the results (bounding boxes, keypoints, etc.) """
        img = r.plot()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.ax1.clear()
        self.ax1.axis('off')
        self.ax1.imshow(img)
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()

        # Compute angles and smooth the curves
        right_angle = self.compute_angle(keypoints[0].reshape(1, 2), keypoints[5].reshape(1, 2), keypoints[3].reshape(1, 2))
        left_angle = self.compute_angle(keypoints[0].reshape(1, 2), keypoints[5].reshape(1, 2), keypoints[4].reshape(1, 2))
        self.xdata.append(self.i)
        self.right_angle_data.append(right_angle)
        self.left_angle_data.append(left_angle)

        if self.i >= 50:
            self.xdata = self.xdata[-50:]
            self.right_angle_data = self.right_angle_data[-50:]
            self.left_angle_data = self.left_angle_data[-50:]

        right_angle_smoothed = self.smooth_data(self.right_angle_data)
        left_angle_smoothed = self.smooth_data(self.left_angle_data)

        self.line_right.set_data(self.xdata, right_angle_smoothed)
        self.line_left.set_data(self.xdata, left_angle_smoothed)
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()

    def plot_results_together(self, r, keypoints):
        """ Plot the results (bounding boxes, keypoints, and angles) """
        img = r.plot()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.ax1.clear()
        self.ax1.axis('off')
        self.ax1.imshow(img)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Compute angles and smooth the curves
        right_angle = self.compute_angle(keypoints[0].reshape(1, 2), keypoints[5].reshape(1, 2), keypoints[3].reshape(1, 2))
        left_angle = self.compute_angle(keypoints[0].reshape(1, 2), keypoints[5].reshape(1, 2), keypoints[4].reshape(1, 2))
        self.xdata.append(self.i)
        self.right_angle_data.append(right_angle)
        self.left_angle_data.append(left_angle)

        if self.i >= 50:
            self.xdata = self.xdata[-50:]
            self.right_angle_data = self.right_angle_data[-50:]
            self.left_angle_data = self.left_angle_data[-50:]

        right_angle_smoothed = self.smooth_data(self.right_angle_data)
        left_angle_smoothed = self.smooth_data(self.left_angle_data)

        self.line_right.set_data(self.xdata, right_angle_smoothed)
        self.line_left.set_data(self.xdata, left_angle_smoothed)
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if len(self.scratchings['end']) > 0 and len(self.scratchings['start']) == len(self.scratchings['end']):
            self.ax3.clear()
            self.ax3.axis('off')
            self.tabledata = [['id', 'start', 'end', 'duration', 'gap', 'side', 'times', 'intensity']]
            for i in range(len(self.scratchings['start'])):
                start_min, start_sec, start_frame = self.frame_index2time(self.scratchings['start'][i])
                end_min, end_sec, end_frame = self.frame_index2time(self.scratchings['end'][i])
                duration_min, duration_sec, duration_frame = self.frame_index2time(self.scratchings['duration'][i])
                gap_min, gap_sec, gap_frame = self.frame_index2time(self.scratchings['gaps'][i])
                # add one more row to the table
                self.tabledata.append([i + 1, f'{start_min}:{start_sec}:{start_frame}', f'{end_min}:{end_sec}:{end_frame}',
                                       f'{duration_min}:{duration_sec}:{duration_frame}', f'{gap_min}:{gap_sec}:{gap_frame}',
                                       self.scratchings['side'][i], self.scratchings['times'][i], self.scratchings['intensity'][i]])
            self.tabledata.extend([self.blank_row for _ in range(24 - len(self.scratchings['start']))])
            self.ax3.table(cellText=self.tabledata, cellLoc='center', loc='center',fontsize=12)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def frame_index2time(self, frame_index):
        """ Convert frame index to time in minutes, seconds, and frames """
        minutes = int(frame_index / (60 * self.fps))
        seconds = int(frame_index % (60 * self.fps) / self.fps)
        frames = int(frame_index % self.fps)
        return minutes, seconds, frames

    def compute_angle(self, p1, p2, p3):
        """ Compute the angle between three points (p1, p2, p3) """
        assert p1.shape == p2.shape == p3.shape, f'Expected same shape, got {p1.shape}, {p2.shape}, {p3.shape}'
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.arccos(np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)))
        return np.degrees(angle)

    def count_local_minima(self, data, threshold, include_edges=False):
        """ Count the number of local minima under threshold in the data """
        count = 0
        for i in range(1, len(data) - 1):
            if data[i] < data[i - 1] and data[i] < data[i + 1] and data[i] < threshold:
                count += 1
        if include_edges: # also consider the first and last elements
            if data[0] < data[1] - 5 and data[0] < threshold:
                count += 1
            if data[-1] < data[-2] - 5 and data[-1] < threshold:
                count += 1
        return count

    def validate_scratching(self):
        """ Validate if a set of keypoints and behaviors form a valid scratching event """
        if len(self.behaviours_scratching) < 0.1 * self.fps:
            return False
        # !!! add more validation criteria here !!!
        return True

    def evaluate_scratching(self):
        """ Evaluate a scratching event and calculate side, times, and intensity """
        keypoints_array = np.array(self.keypoints_scratching)
        licking_index = np.where(np.array(self.behaviours_scratching) == 'paw_licking')[0]
        keypoints_scratching = np.array_split(keypoints_array, licking_index)
        keypoints_scratching = [keypoints for keypoints in keypoints_scratching if keypoints.shape[0] > 1]

        # Calculate average speed for right and left limbs
        speed_right = [np.mean(np.linalg.norm(np.diff(scratch[:, 1], axis=0), axis=1)) for scratch in keypoints_scratching]
        speed_left = [np.mean(np.linalg.norm(np.diff(scratch[:, 2], axis=0), axis=1)) for scratch in keypoints_scratching]
        side = ['right' if speed_right[i] > speed_left[i] else 'left' for i in range(len(keypoints_scratching))]

        # Calculate the angles for the side of scratching
        angles = [self.compute_angle(scratch[:, 0], scratch[:, 3], scratch[:, 1 if side[i] == 'right' else 2]) for i, scratch in enumerate(keypoints_scratching)]
        times = np.sum([self.count_local_minima(angle, threshold=45) for angle in angles])
        intensity = times  #!!! modified to how to calculate the intensity here !!!

        return side, times, intensity
    
    def detect_scratching(self, class_highest_conf, keypoints):
        """ Detect when scratching starts, continues, or ends """
        # start scratching when the class with the highest confidence is scratching
        if class_highest_conf == 1 and len(self.scratchings['start']) == len(self.scratchings['end']):
            gap = self.i - self.scratchings['end'][-1] if len(self.scratchings['end']) > 0 else self.i
            # if the gap is less than 0.1 seconds, merge the scratching events with the previous one
            if gap <= self.fps * 0.1:
                self.is_continue = True
            else:
                self.is_continue = False
            self.scratchings['start'].append(self.i)
            self.scratchings['duration'].append(0)
            self.scratchings['gaps'].append(gap)
            self.keypoints_scratching.append(keypoints[[0, 3, 4, 5]])
            self.behaviours_scratching.append('scratching')
        # continue scratching when the class with the highest confidence is scratching or paw licking
        elif (class_highest_conf == 1 or class_highest_conf == 2) and len(self.scratchings['start']) > len(self.scratchings['end']):
            self.scratchings['duration'][-1] += 1
            self.keypoints_scratching.append(keypoints[[0, 3, 4, 5]])
            self.behaviours_scratching.append('scratching' if class_highest_conf == 1 else 'paw_licking')
        # end scratching when the class with the highest confidence is not scratching
        elif class_highest_conf == 0 and len(self.scratchings['start']) > len(self.scratchings['end']):
            self.scratchings['end'].append(self.i)
            self.scratchings['duration'][-1] += 1
            # validate the process is a scratching behaviour
            if self.validate_scratching():
                side, times, intensity = self.evaluate_scratching()
                self.scratchings['side'].append(side[0] if len(set(side)) == 1 else 'both')
                self.scratchings['times'].append(times)
                self.scratchings['intensity'].append(intensity)
                if self.is_continue:
                    self.merge_scratching()
                start_min, start_sec, start_frame = self.frame_index2time(self.scratchings['start'][-1])
                end_min, end_sec, end_frame = self.frame_index2time(self.scratchings['end'][-1])
                duration_min, duration_sec, duration_frame = self.frame_index2time(self.scratchings['duration'][-1])
                gap_min, gap_sec, gap_frame = self.frame_index2time(self.scratchings['gaps'][-1])
                with open(self.save_path, 'a', newline='') as f:
                    f.write(f'{len(self.scratchings["start"])},{start_min}:{start_sec}:{start_frame},{end_min}:{end_sec}:{end_frame},')
                    f.write(f'{duration_min}:{duration_sec}:{duration_frame},{gap_min}:{gap_sec}:{gap_frame},')
                    f.write(f'{self.scratchings["side"][-1]},{self.scratchings["times"][-1]},{self.scratchings["intensity"][-1]}\n')
            else:
                warnings.warn('Warning: scratching behaviour is invalid')
                # remove the last scratching event
                self.scratchings['start'].pop()
                self.scratchings['end'].pop()
                self.scratchings['duration'].pop()
                self.scratchings['gaps'].pop()
            self.keypoints_scratching = []
            self.behaviours_scratching = []
        else:
            pass
        return None
    
    def merge_scratching(self):
        """ Merge the current scratching behaviour with the previous one """
        self.scratchings['start'].pop()
        self.scratchings['end'][-2] = self.scratchings['end'][-1]
        self.scratchings['end'].pop()
        self.scratchings['duration'][-2] += self.scratchings['duration'][-1]
        self.scratchings['duration'][-2] += self.scratchings['gaps'][-1]
        self.scratchings['duration'].pop()
        self.scratchings['gaps'].pop()
        if len(set(self.scratchings['side'][-2:])) == 1:
            self.scratchings['side'].pop()
        else:
            self.scratchings['side'][-2] = 'both'
            self.scratchings['side'].pop()
        self.scratchings['times'][-2] += self.scratchings['times'][-1]
        self.scratchings['times'].pop()
        # change the intensity metric to a more sophisticated one
        self.scratchings['intensity'][-2] += self.scratchings['intensity'][-1]
        self.scratchings['intensity'].pop()
        # delete previous recording in the CSV file
        with open(self.save_path, 'r') as f:
            lines = f.readlines()
        with open(self.save_path, 'w') as f:
            f.writelines(lines[:-1])
        return None
    
    def process_frame(self, frame):
        """ Process a single frame """
        
        return None

    def process_video(self):
        """ Main video processing loop """
        cap = cv2.VideoCapture(self.video_path)
        # self.num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
        self.num_frames = 1800 # or set to the number of frames you want to process
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.i = 0

        while self.i < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            # predict the keypoints and classes from the trained model
            results = self.model(frame, device=self.device, verbose=False)
            r = results[0].cpu().numpy()

            # Clear GPU memory if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            classes, classes_conf, points, points_conf = r.boxes.cls, r.boxes.conf, r.keypoints.xy, r.keypoints.conf
            if self.high_conf_class:
                class_highest_conf = int(classes[np.argmax(classes_conf)])
                keypoints = points[np.argmax(classes_conf)]
            else:
                # class priority: scratching > paw_licking > other
                if 1 in classes:
                    class_highest_conf = 1
                    keypoints = points[classes == 1]
                elif 2 in classes:
                    class_highest_conf = 2
                    keypoints = points[classes == 2]
                else:
                    class_highest_conf = 0
                    keypoints = points[classes == 0]
            if self.plot_prediction:
                #self.plot_results(r, keypoints)
                self.plot_results_together(r, keypoints)

            self.detect_scratching(class_highest_conf, keypoints)
            self.i += 1
        cap.release()
        if self.plot_prediction:
            plt.ioff()
            plt.show()
        return None