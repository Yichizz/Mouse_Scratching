# convert data forms from labelme to YOLOv8

from utils.data_preparation import json_to_yolo_keypoints, check_data
from ultralytics import YOLO

def main():
    recreate_data = False # set to True if you want to recreate the dataset
    if recreate_data:
        frames_path = '../Scratching_Mouse/frames'
        labels_path = '../Scratching_Mouse/labels'

        # training: A 1-10, C 1-7 testing: B 1, D 1, E 1
        training_videos = ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8', 'A-9', 'A-10', 'B-1', 'C-1', 'C-2', 'C-3', 'C-4', 'C-5', 'C-6', 'C-7']
        testing_videos = ['D-1', 'E-1']  # leave the list empty if no testing videos
        check_data(frames_path, labels_path, training_videos, testing_videos) # check if the data is prepared correctly

        json_to_yolo_keypoints(frames_path, labels_path, training_videos, testing_videos)
        print('Data preparation complete')

    # Load pre-trained model for fine-tuning/model testing
    pre_trained_model_path = 'runs/pose/train14/weights/best.pt'
    model = YOLO(pre_trained_model_path)

    training_model = False # set to True if you want to train the model
    if training_model:
        # Train the model, change hyperparameters if needed
        training_results = model.train(data="datasets/mouse-pose.yaml",
                                epochs=200,
                                patience=50, # early stopping patience
                                batch=64,
                                device=0,
                                verbose=False,
                                cos_lr=True, # cosine learning rate schedule
                                lr0=1e-4, # initial learning rate
                                lrf=1e-6, # small final learning rate for finer tuning
                                freeze=[0,1,2,3,4,5,6,7,8,9], # freeze all the backbone layers
                                label_smoothing=0.1, # label smoothing epsilon
                                dropout=0.3, # add dropout to the head
                                optimizer="AdamW", 
                                save_period=50, # save the model every 50 epochs
                                degrees=15,
                                save=True, # save the model after training
                                pose = 24,
                                amp=True)
        # if save to a specific path
        # model.save("weights/last_train.pt")
        print('Training complete')
    
    # Test the model
    testing_model = True # set to True if you want to test the model
    if testing_model:
        # validate on the test set and compute test performance metrics
        print('Testing the model')
        testing_results = model.val(data = "datasets/mouse-pose.yaml", split='test', save_json=True)
        print('Testing complete')

if __name__ == '__main__':
    main()