# convert data forms from labelme to YOLOv8 & vice versa

import os
import sys
import json
from tqdm import tqdm
import numpy as np
import shutil
from utils.json_to_txt import json_write_to_txt

sys.path.append(".")


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
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "labels"), exist_ok=True)
    # train and val subfolders within images and labels
    os.makedirs(os.path.join(data_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "labels/val"), exist_ok=True)
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
    with open(os.path.join(data_dir, "data.yaml"), "w") as f:
        f.write(f"nc: {num_classes}\n")
        f.write("names: [" + ", ".join([f'"{name}"' for name in class_names]) + " ]\n")


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
    assert num_keypoints in [4, 6], "num_keypoints must be 4 or 6"
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    for json_file in json_files:
        video, index = json_file.split(".")[0].split("_")
        # for video A we use frames 0-4999 for training and 5000-5964 for validation
        # for video B we use frames 0-994 for training and 995-1650 for validation
        split = "unused"
        if video == "A":
            if int(index) < 5000:
                split = "train"
            elif int(index) < 5965:
                split = "val"
        else:
            if int(index) < 995:
                split = "train"
            elif int(index) < 1651:
                split = "val"
        if split == "unused":
            break
        else:
            txt_file = os.path.join(data_dir, f"labels/{split}", f"{video}_{index}.txt")
            json_file = os.path.join(json_dir, json_file)
            # convert json to txt and write to txt file
            json_write_to_txt(json_file, txt_file, num_keypoints)
            # copy image file to images folder
            image_file = os.path.join(json_dir, f"{video}_{index}.jpg")
            shutil.copy(image_file, os.path.join(data_dir, f"images/{split}"))
            if verbose:
                print(
                    f"case {video}_{index} converted to YOLOv8 format in {split} folder"
                )
    return None


def json_to_yolo_random_split(
    data_dir, json_dir, num_keypoints, val_frac=0.2, verbose=False
):
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
    assert num_keypoints in [4, 6], "num_keypoints must be 4 or 6"
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    video_indexes = list(set([f.split(".")[0].split("_")[0] for f in json_files]))
    print(f"videos found: {video_indexes}")

    for video in video_indexes:
        np.random.seed(42)
        video_files = [f for f in json_files if f.split(".")[0].split("_")[0] == video]
        val_files = np.random.choice(
            video_files, int(len(video_files) * val_frac), replace=False
        )
        print(
            f"video {video} has {len(video_files)} frames, {len(val_files)} frames used for validation"
        )
        train_files = [f for f in video_files if f not in val_files]
        for split, files in zip(["train", "val"], [train_files, val_files]):
            for json_file in tqdm(files):
                index = json_file.split(".")[0].split("_")[1]
                txt_file = os.path.join(
                    data_dir, f"labels/{split}", f"{video}_{index}.txt"
                )
                json_file = os.path.join(json_dir, json_file)
                # convert json to txt and write to txt file
                try:
                    json_write_to_txt(json_file, txt_file, num_keypoints)
                    # copy image file to images folder
                    image_file = os.path.join(json_dir, f"{video}_{index}.jpg")
                    shutil.copy(image_file, os.path.join(data_dir, f"images/{split}"))
                except:
                    print("error converting json to txt for file: ", json_file)

                if verbose:
                    print(
                        f"case {video}_{index} converted to YOLOv8 format in {split} folder"
                    )

    return None


def json_to_yolo_keypoints(
    frames_path,
    labels_path,
    training_videos,
    testing_videos,
    val_frac=0.1,
    verbose=True,
):
    """
    Convert json labels to txt labels for YOLOv8 training and randomly split into train and val
    Args:
        frames_path (str): path to frames
        labels_path (str): path to labels
        training_videos (list): list of training video names, e.g. ['A', 'C']
        testing_videos (list): list of testing video names, e.g. ['B', 'D']
        val_frac (float): fraction of frame each training video to be used for validation
        verbose (bool): print conversion progress

    Returns:
        None
    """
    assert os.path.exists(frames_path), "frames_path does not exist"
    assert os.path.exists(labels_path), "labels_path does not exist"
    assert len(training_videos) > 0, "training_videos is empty"
    # raise warning if no testing videos
    if len(testing_videos) == 0:
        print("Warning: testing_videos is empty")

    # create folders for datasets of YOLOV8 training
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    if verbose:
        print("Converting json labels to txt labels for YOLOv8 training...")

    if len(testing_videos) > 0:
        if os.path.exists("datasets/images/test") and os.path.exists(
            "datasets/labels/test"
        ):
            # remove existing test data in the test dataset but keep the folder
            shutil.rmtree("datasets/images/test")
            shutil.rmtree("datasets/labels/test")
        os.makedirs("datasets/images/test")
        os.makedirs("datasets/labels/test")
        for sub_dir in os.listdir(frames_path):
            if sub_dir in testing_videos:
                frames = os.listdir(os.path.join(frames_path, sub_dir))
                frames = [frame for frame in frames if frame.endswith(".png")]
                if verbose:
                    print(f"Converting testing data for video {sub_dir}...")
                for frame in tqdm(frames):
                    index = int(frame.split(".")[0].split("img")[1])
                    json_file = os.path.join(
                        labels_path, sub_dir, f"img{index:05d}.json"
                    )
                    txt_file = os.path.join(
                        "datasets", "labels", "test", f"{sub_dir}_{index:05d}.txt"
                    )
                    json_write_to_txt(json_file, txt_file, 6)
                    shutil.copy(
                        os.path.join(frames_path, sub_dir, frame),
                        os.path.join(
                            "datasets", "images", "test", f"{sub_dir}_{index:05d}.png"
                        ),
                    )
        print("Testing data converted to YOLO format")

    if os.path.exists("datasets/images/train") and os.path.exists(
        "datasets/labels/train"
    ):
        # remove existing train data in the train dataset but keep the folder
        shutil.rmtree("datasets/images/train")
        shutil.rmtree("datasets/labels/train")
    os.makedirs("datasets/images/train")
    os.makedirs("datasets/labels/train")
    if os.path.exists("datasets/images/val") and os.path.exists("datasets/labels/val"):
        # remove existing val data in the val dataset but keep the folder
        shutil.rmtree("datasets/images/val")
        shutil.rmtree("datasets/labels/val")
    os.makedirs("datasets/images/val")
    os.makedirs("datasets/labels/val")

    for sub_dir in os.listdir(frames_path):
        if sub_dir in training_videos:
            frames = os.listdir(os.path.join(frames_path, sub_dir))
            frames = [frame for frame in frames if frame.endswith(".png")]
            np.random.seed(42)  # set seed for reproducibility
            val_frames = np.random.choice(
                frames, int(len(frames) * val_frac), replace=False
            )
            train_frames = [frame for frame in frames if frame not in val_frames]
            if verbose:
                print(f"Converting training data for video {sub_dir}...")
            for split, frames in zip(["train", "val"], [train_frames, val_frames]):
                for frame in tqdm(frames):
                    index = int(frame.split(".")[0].split("img")[1])
                    json_file = os.path.join(
                        labels_path, sub_dir, f"img{index:05d}.json"
                    )
                    txt_file = os.path.join(
                        "datasets", "labels", split, f"{sub_dir}_{index:05d}.txt"
                    )
                    json_write_to_txt(json_file, txt_file, 6)
                    shutil.copy(
                        os.path.join(frames_path, sub_dir, frame),
                        os.path.join(
                            "datasets", "images", split, f"{sub_dir}_{index:05d}.png"
                        ),
                    )
    print("Training and validation data converted to YOLO format")

    return None


def check_data(frames_path, labels_path, training_videos, testing_videos):
    """
    Check if: 1. all frames have corresponding labels 2. all labels have corresponding frames
    3. in the label json files, there are in total 6 non-repeated keypoints + 1 bounding box
    4. if there is a group_id for the keypoint, the group_id should be 0,1, or None
    5. if there is a group_id for the bounding box, the group_id should be 1,2, or None
    6. if there's a point (0,0) in the label json files, it should be an invalid point (i.e. group_id = 0)
    """
    print("Checking data...")
    for sub_dir in os.listdir(frames_path):
        if sub_dir in training_videos or sub_dir in testing_videos:
            frames = os.listdir(os.path.join(frames_path, sub_dir))
            frames = [frame for frame in frames if frame.endswith(".png")]
            json_files = os.listdir(os.path.join(labels_path, sub_dir))
            json_files = [
                json_file for json_file in json_files if json_file.endswith(".json")
            ]
            for frame in frames:
                index = int(frame.split(".")[0].split("img")[1])
                json_file = os.path.join(labels_path, sub_dir, f"img{index:05d}.json")
                assert os.path.exists(
                    json_file
                ), f"json file {json_file} does not exist"
            for json_file in tqdm(json_files):
                index = int(json_file.split(".")[0].split("img")[1])
                frame = os.path.join(frames_path, sub_dir, f"img{index:05d}.png")
                assert os.path.exists(frame), f"frame {frame} does not exist"
                json_file = os.path.join(labels_path, sub_dir, json_file)
                with open(json_file, "r") as f:
                    data = json.load(f)
                    assert (
                        len(data["shapes"]) == 7
                    ), f'json file {json_file} has {len(data["shapes"])} shapes'
                    for shape in data["shapes"]:
                        if "group_id" in shape.keys() and shape["label"] != "mouse":
                            assert shape["group_id"] in [
                                0,
                                1,
                                None,
                            ], f'group_id for keypoints in json file {json_file} has {shape["group_id"]}'
                        if shape["label"] == "mouse" and "group_id" in shape.keys():
                            assert shape["group_id"] in [
                                0,
                                1,
                                2,
                                None,
                            ], f'group_id for bbox in json file {json_file} has {shape["group_id"]}'
                        if shape["points"][0] == [0, 0]:
                            assert (
                                shape["group_id"] == 0
                            ), f'point (0,0) in json file {json_file} has group_id {shape["group_id"]}'
    print("Data check complete")
