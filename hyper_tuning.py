from ultralytics import YOLO


def main():
    # Initialize the YOLO model
    # Load the default YOLOv8 model
    model = YOLO("yolov8n-pose.yaml").load("weights/yolov8n-pose.pt")
    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(
        data="dataset_reannotated/mouse-pose.yaml",
        epochs=30,
        iterations=50,  # number of tuning iterations
        batch=32,  # auto-config based on GPU memory
        optimizer="AdamW",
        freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        label_smoothing=0.1,
        dropout=0.3,
        cache=False,
        amp=True,  # automatic mixed precision for faster training
        plots=False,
        save=False,
        val=False,
    )


if __name__ == "__main__":
    main()
