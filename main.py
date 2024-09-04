# convert data forms from labelme to YOLOv8

from utils.data_preparation import create_folders, json_to_yolo

data_dir = 'dataset_reannotated'
json_dir = 'frame'
num_keypoints = 6

def main():
    create_folders(data_dir)
    json_to_yolo(data_dir, json_dir, num_keypoints, verbose=True)

if __name__ == '__main__':
    main()