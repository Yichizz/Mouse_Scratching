# convert json labels to txt labels for YOLOv8 training
# txt file format: class x_center y_center width height keypoint1_x keypoint1_y visiblity1 keypoint2_x keypoint2_y visiblity2 ...
# class: 0 - mouse
# visibility: 0 - not visible, 1 - occluded, 2 - visible

# keypoints: 4 - no front feet (i.e. no label 1 & 2), 6 - all keypoints (0-5)

import os
import json
import numpy as np

def coordinates2yolo(x, y, width, height):
    x_center = x + width / 2
    y_center = y + height / 2
    return x_center, y_center, width, height

def json_to_txt(json_file, txt_file, num_keypoints):
    """
    Convert json labels to txt labels for YOLOv8 training
    Args:
        json_file (str): json file path
        txt_file (str): txt file path
        num_keypoints (int): number of keypoints (4 or 6)
    
    Returns:
        None
    """
    assert json_file.endswith('.json'), 'json_file must be a json file'
    assert os.path.exists(json_file), 'json_file does not exist'
    assert num_keypoints in [4, 6], 'num_keypoints must be 4 or 6'
    data = json.load(open(json_file))
    if num_keypoints == 4:
        keypoint_list = [0, 3, 4, 5]
    elif num_keypoints == 6:
        keypoint_list = [0, 1, 2, 3, 4, 5]
    
    with open(txt_file, 'w') as f:
        for obj in data['shapes']:
            height = obj['imageHeight']
            width = obj['imageWidth']
            if obj['label'] == 'mouse':
                # convert json coordinates to real coordinates
                x = np.array(obj['points'])[0][0]
                y = np.array(obj['points'])[0][1]
                w = np.array(obj['points'])[1][0] - x
                h = np.array(obj['points'])[1][1] - y
                x_center, y_center, width, height = coordinates2yolo(x, y, w, h)
                f.write(f'0 {x_center} {y_center} {width} {height} ')
            else:
                for i in keypoint_list:
                    if int(obj['label']) == i:
                        x = obj['points'][i][0] / width
                        y = obj['points'][i][1] / height
                        if obj['group_id'] == 1:
                            f.write(f'{x} {y} 1 ')
                        else:
                            f.write(f'{x} {y} 2 ')
                        break
        f.write('\n')
        f.close()
    return None