# convert predicted txt files from YOLOv8 to labelme json format
# Remember to change the order of right and left!
import os
import json
import PIL
import numpy as np

def yolo2coordinates(x_center, y_center, width, height, image_width, image_height):
    top_left_x = (x_center - width / 2) * image_width
    top_left_y = (y_center - height / 2) * image_height
    bottom_right_x = (x_center + width / 2) * image_width
    bottom_right_y = (y_center + height / 2) * image_height
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y

def txt_write_to_json(txt_file, json_file, image_file, num_keypoints=4):
    """
    Convert txt labels to json labels for labelme
    Args:
        txt_file (str): txt file path
        json_file (str): json file path
        image_file (str): image file path
        num_keypoints (int): number of keypoints (4 or 6)
    
    Returns:
        None
    """
    assert txt_file.endswith('.txt'), 'txt_file must be a txt file'
    assert os.path.exists(txt_file), 'txt_file does not exist'
    assert num_keypoints in [4, 6], 'num_keypoints must be 4 or 6'
    with open(txt_file, 'r') as f:
        data = f.read().split()
    x_center, y_center, width, height = [float(x) for x in data[1:5]]
    image = PIL.Image.open(image_file)
    image_width, image_height = image.size
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = yolo2coordinates(x_center, y_center, width, height, image_width, image_height)
    if num_keypoints == 4:
        keypoint_list = [2, 6, 5, 7]
    elif num_keypoints == 6:
        keypoint_list = [2, 4, 3, 6, 5, 7]
    else:
        raise ValueError('Invalid number of keypoints')
    
    shapes = []
    # append mouse rectangle
    shapes.append({
        "label": "mouse",
        "points": [[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]],
        "shape_type": "rectangle",
        "flags": {}
        "mask": None
    })
    for i in range(num_keypoints):
        x, y, group = [float(x) for x in data[keypoint_list[i]:keypoint_list[i]+3]]
        x *= image_width
        y *= image_height
        if group == 2:
            # group_id id Null for normal keypoints
            shapes.append({
                "label": f"{keypoint_list[i]}",
                "points": [[x, y]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {}
                "mask": None
            })
        else:
            shapes.append({
                "label": f"{keypoint_list[i]}",
                "points": [[x, y]],
                "group_id": group,
                "description": "",
                "shape_type": "point",
                "flags": {}
                "mask": None
            })
    # write json file
    with open(json_file, 'w') as f:
        json.dump({
            "version": "5.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(image_file),
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width
        }, f)
    return None