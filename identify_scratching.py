import os
import cv2
import csv
import torch
import warnings
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt

def device_diagnosis(verbose=True):
    # use GPU if available, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f'Using device: {device}')
    return device

def frame_index2time(frame_index, fps):
    # return the time in  minutes and seconds
    minutes = int(frame_index / (60 * fps))
    seconds = int(frame_index % (60 * fps) / fps)
    frames = int(frame_index % fps)
    return minutes, seconds, frames

def compute_angle(p1, p2, p3):
    # p1, p2, p3 are arrays of left, vertex, right points
    assert p1.shape == p2.shape == p3.shape, f'Expected same shape, got {p1.shape}, {p2.shape}, {p3.shape}'
    v1 = p1-p2
    v2 = p3-p2
    angle = np.arccos(np.sum(v1 * v2, axis=1)/(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)))
    return np.degrees(angle)

def count_local_minima(data, threshold, include_edges=False):
    count = 0
    for i in range(1, len(data)-1):
        if data[i] < data[i-1] and data[i] < data[i+1] and data[i] < threshold:
            count += 1
    if include_edges:
        if data[0] < data[1] - 5 and data[0] < threshold:
            count += 1
        if data[-1] < data[-2] - 5 and data[-1] < threshold:
            count += 1
    return count

def validate_scratching(keypoints_scratching, behaviours_scratching):
    if len(behaviours_scratching) < 3:
        return False
    else:
        return True

def evaluate_scratching(keypoints_scratching, behaviours_scratching, verbose=True):
    # we group the keypoints in consecutive scratching frames into an array
    # and for a list of arrays for each scratching behaviour within a scratching train
    keypoints_array = np.array(keypoints_scratching)
    licking_index = np.where(np.array(behaviours_scratching) == 'paw_licking')[0]
    # cut into different arrays for scratching intersected by paw licking
    keypoints_scratching = np.array_split(keypoints_array, licking_index)
    # discard (1,4,2) arrays in the list
    keypoints_scratching = [keypoints for keypoints in keypoints_scratching if keypoints.shape[0] > 1]
    if verbose:
        print(f'Found {len(keypoints_scratching)} scratching behaviours')
    # find the side of scratching: side with higher average speed
    # i.e. difference between pre and now keypoints
    speed_right = [np.mean(np.linalg.norm(np.diff(scratch[:,1], axis=0), axis=1)) for scratch in keypoints_scratching]
    speed_left = [np.mean(np.linalg.norm(np.diff(scratch[:,2], axis=0), axis=1)) for scratch in keypoints_scratching]
    side = ['right' if speed_right[i] > speed_left[i] else 'left' for i in range(len(keypoints_scratching))]
    if verbose:
        print(f'Average speed right: {speed_right}, left: {speed_left}')
        print(f'Scratching side: {side}')
    # we count the angles for the side of scratching
    angles = [compute_angle(scratch[:,0], scratch[:,3], scratch[:,1 if side[i] == 'right' else 2]) for i, scratch in enumerate(keypoints_scratching)]
    times = np.sum([count_local_minima(angle, threshold=45) for angle in angles])
    # change here for how to calculate intensity
    intensity = times 
    return side, times, intensity

video_path = '../Scratching_Mouse/videos/A-1.mp4'
pretrained_weights = 'runs/pose/train/weights/best.pt'
save_path = './A-1.csv'
device_available = device_diagnosis(verbose=False)

# load the model
model = YOLO(pretrained_weights)

# we read the video frame by frame and detect the mouse
cap = cv2.VideoCapture(video_path)
# num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total number of frames
num_frames = 1800
fps = cap.get(cv2.CAP_PROP_FPS)
i = 0 # current frame index
scratchings = dict(start=[], end=[], duration=[], gaps=[], side=[], times=[], intensity=[])
keypoints_scratching = []
behaviours_scratching = []

# create a csv file to WRITE IN the scratching results IN LIVE
with open(save_path, 'w', newline='') as f:
    f.write('瘙痒id,开始时间,结束时间,持续时常,潜伏时常,瘙痒侧,瘙痒次数,瘙痒强度\n')

# create 2 extra windows for plotting the prediction results from YOLO
# and plotting the curves for the angle/distance/speed
%matplotlib qt
fig1, ax1 = plt.subplots(figsize=(12,9))
fig2, ax2 = plt.subplots(figsize=(12,6))
# initialize the animation for the angle/distance/speed
# EXAMPLE: angle
line_right, = ax2.plot([], [], label='angle_right', color='red')
line_left, = ax2.plot([], [], label='angle_left', color='blue')
ax2.legend()
ax2.xaxis.set_visible(False)
ax2.set_ylim(0, 90) # angle range from 0 to 90 degrees
ax2.set_title('Angle')
xdata, right_angle_data, left_angle_data = [], [], []

while i < num_frames:
    # read the ith frame
    ret, frame = cap.read()
    if not ret:
        break # end of video
    results = model(frame, device=device_available, verbose=False)
    r = results[0].cpu().numpy()
    
    # clear gpu memory
    if device_available.type == 'cuda':
        torch.cuda.empty_cache()
    classes, classes_conf, points, points_conf = r.boxes.cls, r.boxes.conf, r.keypoints.xy, r.keypoints.conf
    # find the class with the highest confidence
    class_highest_conf = int(classes[np.argmax(classes_conf)]) 
    # Or, if scratching and other are possible, we always choose scratching, similar to the paw licking
    # add code here!!!
    keypoints = points[np.argmax(classes_conf)]
    assert keypoints.shape == (6,2), f'Expected 6 keypoints in (x,y) format, got {keypoints.shape}'
    # plot the bounding box and keypoints for the class with highest confidence
    # Plot results (bounding boxes, keypoints, etc.)
    img = r.plot()  
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # plot the image in the matplotlib window
    ax1.clear()
    ax1.axis('off')
    ax1.imshow(img)
    fig1.canvas.draw() # draw the image on the canvas
    fig1.canvas.flush_events() # update the window to show the new frame

    # plot the curves for the angle/distance/speed
    # EXAMPLE: angle
    right_angle = compute_angle(keypoints[0].reshape(1,2), keypoints[5].reshape(1,2), keypoints[3].reshape(1,2))
    left_angle = compute_angle(keypoints[0].reshape(1,2), keypoints[5].reshape(1,2), keypoints[4].reshape(1,2))
    # dynamically add the new data (current frame, angle) to the plot
    xdata.append(i)
    right_angle_data.append(right_angle)
    left_angle_data.append(left_angle)
    if i >= 50:
        xdata = xdata[-50:]
        right_angle_data = right_angle_data[-50:]
        left_angle_data = left_angle_data[-50:]
    # we smooth the curve by applying a gaussian filter
    # add code here!!!
    line_right.set_data(xdata, right_angle_data)
    line_left.set_data(xdata, left_angle_data)
    ax2.relim()
    ax2.autoscale_view()
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    # find the start of scratching
    if class_highest_conf == 1 and len(scratchings['start']) == len(scratchings['end']):
        # if the gap is too small, we merge this scratching with the previous one
        # add code here!!!
        start_minute, start_second, start_frame = frame_index2time(i, fps)
        scratchings['start'].append(i)
        scratchings['duration'].append(0)
        scratchings['gaps'].append(i-scratchings['end'][-1] if len(scratchings['end']) > 0 else i)
        # print(f'Start scratching at {start_minute} minutes {start_second} seconds {start_frame} frames, gap: {scratchings["gaps"][-1]} frames')
        # we only need head, right hind foot, left hind foot, tail base
        keypoints_scratching.append(keypoints[[0, 3, 4, 5]])
        # what if there's an invalid point?
        # add amendance code here!!!
        behaviours_scratching.append('scratching')
    # find the duration of scratching (behaviour can be 1/2)
    elif (class_highest_conf == 1 or class_highest_conf == 2)  and len(scratchings['start']) > len(scratchings['end']):
        scratchings['duration'][-1] += 1
        keypoints_scratching.append(keypoints[[0, 3, 4, 5]])
        behaviours_scratching.append('scratching' if class_highest_conf == 1 else 'paw_licking')
    # find the end of scratching
    elif class_highest_conf == 0 and len(scratchings['start']) > len(scratchings['end']):
        scratchings['end'].append(i)
        scratchings['duration'][-1] += 1
        end_minute, end_second, end_frame = frame_index2time(i, fps)
        # validate the process is a scratching behaviour
        # validation code here!!!
        if validate_scratching(keypoints_scratching, behaviours_scratching):
            side, times, intensity = evaluate_scratching(keypoints_scratching, behaviours_scratching, verbose=False)
            scratchings['side'].append(side[0] if len(set(side)) == 1 else 'both')
            scratchings['times'].append(times)
            scratchings['intensity'].append(intensity)
            # print(f'Scratching duration: {scratchings["duration"][-1]} frames, side: {scratchings["side"][-1]}, times: {scratchings["times"][-1]}, intensity: {scratchings["intensity"][-1]}')
            # print(f'End scratching at {end_minute} minutes {end_second} seconds {end_frame} frames')
            # print('----------------------------------------------')
            # write the results to the csv file
            with open(save_path, 'a', newline='') as f:
                f.write(f'{len(scratchings["start"])},{start_minute}:{start_second}:{start_frame},{end_minute}:{end_second}:{end_frame},{scratchings["duration"][-1]},{scratchings["gaps"][-1]},{scratchings["side"][-1]},{scratchings["times"][-1]},{scratchings["intensity"][-1]}\n')

        else: 
            # print a warning message in case the scratching behaviour is invalid
            warnings.warn('Warning: scratching behaviour is invalid')
            # remove the last scratching behaviour
            scratchings['start'].pop()
            scratchings['end'].pop()
            scratchings['duration'].pop()
            scratchings['gaps'].pop()
        keypoints_scratching = []
        behaviours_scratching = []
    else:
        pass
    i += 1