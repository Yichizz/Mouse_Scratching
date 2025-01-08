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