# convert data forms from labelme to YOLOv8

import os
import sys
import PIL
from tqdm import tqdm
import numpy as np
import shutil
from utils.json_to_txt import json_write_to_txt
sys.path.append('.')

def create_folders(data_dir):
    """
    Create folders for YOLOv8 training
    Args:
        data_dir (str): data directory path
    
    Returns:
        None
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'labels'), exist_ok=True)
    # train and val subfolders within images and labels
    os.makedirs(os.path.join(data_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'labels/val'), exist_ok=True)
    return None

def create_data_config_yaml(data_dir, num_classes, class_names):
    """
    Create data.yaml file for YOLOv8 training
    Args:
        data_dir (str): data directory path
        num_classes (int): number of classes
        class_names (list): list of class names
    
    Returns:
        None
    """
    with open(os.path.join(data_dir, 'data.yaml'), 'w') as f:
        f.write(f'nc: {num_classes}\n')
        f.write('names: [' + ', '.join([f'"{name}"' for name in class_names]) + ' ]\n')

def json_to_yolo(data_dir, json_dir, num_keypoints, verbose=False):
    """
    Convert json labels to txt labels for YOLOv8 training
    Args:
        data_dir (str): data directory path
        json_dir (str): json directory path
        num_keypoints (int): number of keypoints (4 or 6)
        verbose (bool): print conversion progress
    
    Returns:
        None
    """
    assert num_keypoints in [4, 6], 'num_keypoints must be 4 or 6'
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    for json_file in json_files:
        video, index = json_file.split('.')[0].split('_')
        # for video A we use frames 0-4999 for training and 5000-5964 for validation
        # for video B we use frames 0-994 for training and 995-1650 for validation
        split = 'unused'
        if video == 'A':
            if int(index) < 5000:
                split = 'train'
            elif int(index) < 5965:
                split = 'val'
        else:
            if int(index) < 995:
                split = 'train'
            elif int(index) < 1651:
                split = 'val'
        if split == 'unused':
            break
        else:
            txt_file = os.path.join(data_dir, f'labels/{split}', f'{video}_{index}.txt')
            json_file = os.path.join(json_dir, json_file)
            # convert json to txt and write to txt file
            json_write_to_txt(json_file, txt_file, num_keypoints)
            # copy image file to images folder
            image_file = os.path.join(json_dir, f'{video}_{index}.jpg')
            shutil.copy(image_file, os.path.join(data_dir, f'images/{split}'))
            if verbose:
                print(f'case {video}_{index} converted to YOLOv8 format in {split} folder')
    return None

def json_to_yolo_random_split(data_dir, json_dir, num_keypoints, val_frac=0.2, verbose=False):
    """
    Convert json labels to txt labels for YOLOv8 training and randomly split into train and val
    Args:
        data_dir (str): data directory path
        json_dir (str): json directory path
        num_keypoints (int): number of keypoints (4 or 6)
        val_frac (float): fraction of frame each video to be used for validation
        verbose (bool): print conversion progress
    
    Returns:
        None
    """
    assert num_keypoints in [4, 6], 'num_keypoints must be 4 or 6'
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    video_indexes= list(set([f.split('.')[0].split('_')[0] for f in json_files]))
    print(f'videos found: {video_indexes}')

    for video in tqdm(video_indexes):
        np.random.seed(42)
        video_files = [f for f in json_files if f.split('.')[0].split('_')[0] == video]
        val_files = np.random.choice(video_files, int(len(video_files) * val_frac), replace=False)
        print(f'video {video} has {len(video_files)} frames, {len(val_files)} frames used for validation')
        train_files = [f for f in video_files if f not in val_files]
        for split, files in zip(['train', 'val'], [train_files, val_files]):
            for json_file in files:
                index = json_file.split('.')[0].split('_')[1]
                txt_file = os.path.join(data_dir, f'labels/{split}', f'{video}_{index}.txt')
                json_file = os.path.join(json_dir, json_file)
                # convert json to txt and write to txt file
                json_write_to_txt(json_file, txt_file, num_keypoints)
                # copy image file to images folder
                image_file = os.path.join(json_dir, f'{video}_{index}.jpg')
                shutil.copy(image_file, os.path.join(data_dir, f'images/{split}'))
                if verbose:
                    print(f'case {video}_{index} converted to YOLOv8 format in {split} folder')

    return None
