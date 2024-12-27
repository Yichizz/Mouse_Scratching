# convert data forms from labelme to YOLOv8

from utils.data_preparation import json_to_yolo_keypoints, check_data
from ultralytics import YOLO

def main():
    recreate_data = False # set to True if you want to recreate the dataset
    if recreate_data:
        frames_path = 'C:\\Users\\HIO-A001\\Desktop\\mouse\\asoid_data\\frames'
        labels_path = 'C:\\Users\\HIO-A001\\Desktop\\mouse\\asoid_data\\labels'

        # training: A 1-10, B 1-7, C 1-10 testing: D 1, E 1
        training_videos = ['A-1','B-2','C-6','F-3','I-5']
        testing_videos = []  # leave the list empty if no testing videos
        check_data(frames_path, labels_path, training_videos, testing_videos) # check if the data is prepared correctly

        json_to_yolo_keypoints(frames_path, labels_path, training_videos, testing_videos)
        print('Data preparation complete')
        #'A-1','A-2','A-3','A-4','A-5','A-6','A-7','A-8','A-9','A-10','B-1','B-2','B-3','B-4','B-5','B-6','B-7',
        #'C-1','C-2','C-3','C-4','C-5','C-6','C-7','C-8','C-9','C-10', 'F-1','F-2','F-3','F-4','F-5','F-6','F-7','F-8',
    training_model = True # set to True if you want to train the model
    if training_model:
        # Load pre-trained model for fine-tuning/model testing
        pre_trained_model_path = 'runs/pose/v11_large_well_trained/weights/best.pt' # change the path to the model you want to fine-tune
        model = YOLO(pre_trained_model_path)
        # Train the model, change hyperparameters if needed
        training_results = model.train(data="datasets/mouse-pose.yaml",
                                epochs=300,
                                patience=100, # early stopping patience
                                batch=64,
                                device=0,
                                verbose=False,
                                cos_lr=True, # cosine learning rate schedule
                                lr0=1e-3, # initial learning rate
                                lrf=1e-4, # small final learning rate for finer tuning
                                freeze=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], # freeze all the backbone layers (0-10) AND more (11-20) 
                                label_smoothing=0.1, # label smoothing epsilon
                                dropout=0.3, # add dropout to the head
                                optimizer="AdamW", 
                                save_period=20, # save the model every 50 epochs
                                degrees=15,
                                save=True, # save the model after training
                                box = 3.0,
                                cls = 6.0,
                                dfl = 1.5,
                                pose = 12.0,
                                kobj = 2.0,
                                amp=True)
        # if save to a specific path
        # model.save("weights/last_train.pt")
        print('Training complete')
    
    # Test the model
    testing_model = False # set to True if you want to test the model
    if testing_model:
        # Load pre-trained model for fine-tuning/model testing
        pre_trained_model_path = 'runs/pose/train/weights/best.pt' # change the path to the model you want to test
        model = YOLO(pre_trained_model_path)
        # validate on the test set and compute test performance metrics
        print('Testing the model')
        testing_results = model.val(data = "datasets/mouse-pose.yaml", split='test', save_json=True)
        print(f'mAP50: {testing_results.pose.map50:.4f}, mAP75: {testing_results.pose.map75:.4f}, mAP50-95: {testing_results.pose.map:.4f}')
        print('Testing complete')

if __name__ == '__main__':
    main()