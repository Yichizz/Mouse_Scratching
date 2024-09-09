# convert data forms from labelme to YOLOv8

from utils.data_preparation import create_folders, json_to_yolo, json_to_yolo_random_split

data_dir = 'dataset_reannotated'
json_dir = 'frame_good'
num_keypoints = 6

def main():
    create_folders(data_dir)
    #json_to_yolo(data_dir, json_dir, num_keypoints, verbose=True)
    json_to_yolo_random_split(data_dir, json_dir, num_keypoints, val_frac=0.2, verbose=False)

if __name__ == '__main__':
    main()