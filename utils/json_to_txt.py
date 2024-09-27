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

def json_write_to_txt(json_file, txt_file, num_keypoints):
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
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    labels = [obj['label'] for obj in data['shapes']]
    if num_keypoints == 4 and 'mouse' in labels:
        keypoint_list = [0, 3, 4, 5]
    elif num_keypoints == 4 and 'mouse' not in labels:
        keypoint_list = [2, 6, 5, 7]
    elif num_keypoints == 6 and 'mouse' in labels:
        keypoint_list = [0, 1, 2, 3, 4, 5]
    elif num_keypoints == 6 and 'mouse' not in labels:
        keypoint_list = [2, 4, 3, 6, 5, 7]
    else:
        raise ValueError('Invalid number of keypoints or data format')
    
    with open(txt_file, 'w') as f:
        if 'mouse' in labels:
            index = labels.index('mouse')
        else:
            index = labels.index('1')
        
        if data['shapes'][index]['group_id'] == 0:
            f.write(f'0 {0:.6f} {0:.6f} {0:.6f} {0:.6f} ')
        else:
            x = np.array(data['shapes'][index]['points'])[0][0]
            y = np.array(data['shapes'][index]['points'])[0][1]
            w = np.array(data['shapes'][index]['points'])[1][0] - x
            h = np.array(data['shapes'][index]['points'])[1][1] - y
            x_center, y_center, width, height = coordinates2yolo(x, y, w, h)
            x_center /= image_width
            y_center /= image_height
            width = np.abs(width) / image_width
            height = np.abs(height) / image_height
            f.write(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} ')
                
        for i in keypoint_list:
            for obj in data['shapes']:
                if obj['label'] != 'mouse' and int(obj['label']) == i:
                    x = obj['points'][0][0] / image_width
                    y = obj['points'][0][1] / image_height
                    # in case of no group_id in the obj dictionary, we assume the keypoint is visible
                    if 'group_id' not in obj:
                        f.write(f'{x:.6f} {y:.6f} 2 ')
                    elif obj['group_id'] == 1:
                        f.write(f'{x:.6f} {y:.6f} 1 ')
                    elif obj['group_id'] == 0:
                        # write 0.0 0.0 0 for keypoints that are not valid
                        f.write(f'0.0 0.0 0 ')
                    else:
                        f.write(f'{x:.6f} {y:.6f} 2 ')
                    break
        f.write('\n')
        f.close()
    return None