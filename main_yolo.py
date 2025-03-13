# convert data forms from labelme to YOLOv8

from utils.data_preparation import json_to_yolo_keypoints, check_data
from utils.biased_sampler import reduce_redundancy
from utils.pose2other import pose2cls, pose2detect
from ultralytics import YOLO


def main():
    recreate_data = False  # set to True if you want to recreate the dataset
    if recreate_data:
        frames_path = "D:\\mouse_120fps\\frames_60fps"
        labels_path = "D:\\mouse_120fps\\labels_60fps"

        # training: A 1-10, B 1-7, C 1-10 testing: D 1, E 1
        training_videos = ["L-2", "M-1"]
        testing_videos = []  # leave the list empty if no testing videos
        check_data(
            frames_path, labels_path, training_videos, testing_videos
        )  # check if the data is prepared correctly

        json_to_yolo_keypoints(
            frames_path, labels_path, training_videos, testing_videos
        )
        print("Data preparation complete")

    biased_sampler = (
        False  # set to True if you want to reduce redundancy in the dataset
    )
    if biased_sampler:
        reduce_redundancy(
            train_images="datasets/images/train", train_labels="datasets/labels/train"
        )
        reduce_redundancy(
            train_images="datasets/images/val", train_labels="datasets/labels/val"
        )
        print("Biased sampling complete")

    pose_2cls = (
        False  # set to True if you want to convert the dataset to classification form
    )
    if pose_2cls:
        # convert dataset in pose form to classfication form
        pose2cls(
            data_dir="datasets/images",
            label_dir="datasets/labels",
            output_dir="datasets/images",
        )
        print("Conversion to classification form complete")

    pose_2detect = (
        False  # set to True if you want to convert the dataset to detection form
    )
    if pose_2detect:
        pose2detect(
            label_dir="datasets/labels",
            background_dir="D:/mouse/mouse_30fps/background/backgrounds",
            image_dir="datasets/images",
            val=0.2,
        )
        print("Converting the dataset to detection form")

    training_pose = False  # set to True if you want to train the model
    if training_pose:
        # Load pre-trained model for fine-tuning/model testing
        pre_trained_model_path = "runs/pose/v11_large_well_trained/weights/best.pt"  # change the path to the model you want to fine-tune
        model = YOLO(pre_trained_model_path)
        # Train the model, change hyperparameters if needed
        training_results = model.train(
            data="datasets/mouse-pose.yaml",
            epochs=300,
            patience=100,  # early stopping patience
            batch=64,
            device=0,
            verbose=False,
            cos_lr=True,  # cosine learning rate schedule
            lr0=1e-3,  # initial learning rate
            lrf=1e-4,  # small final learning rate for finer tuning
            freeze=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
            ],  # freeze all the backbone layers (0-10) AND more (11-20)
            label_smoothing=0.1,  # label smoothing epsilon
            dropout=0.3,  # add dropout to the head
            optimizer="AdamW",
            save_period=50,  # save the model every 50 epochs
            hsv_h=0.1,  # image HSV-Hue augmentation (fraction)
            degrees=20,
            save=True,  # save the model after training
            amp=True,
        )
        # box = 3.0,
        # cls = 3.0,
        # dfl = 1.5,
        # pose = 12.0,
        # kobj = 2.0,

        # if save to a specific path
        # model.save("weights/last_train.pt")
        print("Training complete")

    training_cls = False  # set to True if you want to train the model
    if training_cls:
        pre_trained_model_path = "weights/yolo11m-cls.pt"
        model = YOLO(pre_trained_model_path, task="cls")
        # Train the model, change hyperparameters if needed
        training_results = model.train(
            data="datasets/mouse-cls.yaml",
            epochs=500,
            patience=100,  # early stopping patience
            batch=64,
            device=0,
            verbose=False,
            cos_lr=True,  # cosine learning rate schedule
            lr0=1e-3,  # initial learning rate
            lrf=1e-4,  # small final learning rate for finer tuning
            freeze=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],  # freeze all the backbone layers (0-10) AND more (11-20)
            label_smoothing=0.1,  # label smoothing epsilon
            dropout=0.3,  # add dropout to the head
            optimizer="AdamW",
            save_period=50,  # save the model every 50 epochs
            hsv_h=0.1,  # image HSV-Hue augmentation (fraction)
            degrees=20,
            save=True,  # save the model after training
            amp=True,
        )
        print("Training complete")

    training_detect = False  # set to True if you want to train the model
    if training_detect:
        pre_trained_model_path = "weights/yolo11m.pt"
        model = YOLO(pre_trained_model_path, task="detect")
        # Train the model, change hyperparameters if needed
        training_results = model.train(
            data="datasets/mouse-detect.yaml",
            epochs=300,
            patience=50,  # early stopping patience
            batch=64,
            device=0,
            verbose=False,
            lr0=1e-3,  # initial learning rate
            lrf=1e-4,  # small final learning rate for finer tuning
            freeze=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],  # freeze all the backbone layers (0-10) AND more (11-20)
            label_smoothing=0.1,  # label smoothing epsilon
            dropout=0.3,  # add dropout to the head
            save_period=20,  # save the model every 50 epochs
            hsv_h=0.1,  # image HSV-Hue augmentation (fraction)
            degrees=20,
            cls=1,
            save=True,  # save the model after training
            amp=True,
        )
        print("Training complete")

    # Test the model
    testing_model = True  # set to True if you want to test the model
    if testing_model:
        # Load pre-trained model for fine-tuning/model testing
        pre_trained_model_path = "runs/pose/v11_large_scratch_detect/weights/best.pt"  # change the path to the model you want to test
        model = YOLO(pre_trained_model_path)
        # validate on the test set and compute test performance metrics
        print("Testing the model")
        testing_results = model.val(
            data="datasets/mouse-pose.yaml", split="test", save_json=True
        )
        print(testing_results)
        # print(f'mAP50: {testing_results.pose.map50:.4f}, mAP75: {testing_results.pose.map75:.4f}, mAP50-95: {testing_results.pose.map:.4f}')
        print("Testing complete")


if __name__ == "__main__":
    main()
