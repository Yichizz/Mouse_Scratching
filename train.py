from ultralytics import YOLO
import warnings

# Ignore warnings from the ultralytics module
warnings.filterwarnings("ignore", module="ultralytics")

# Load the default YOLOv8 model
model = YOLO("yolov8n-pose.yaml").load('weights/yolov8n-pose.pt')

if __name__ == '__main__':
# Train the model
    results = model.train(data="datasets/mouse-pose.yaml",
                            epochs=2000,
                            patience=50, # early stopping patience
                            batch=-1, # autoselect using 60% GPU memory
                            device=0,
                            verbose=True,
                            cos_lr=True, # cosine learning rate schedule
                            lrf=0.005, # small final learning rate for finer tuning
                            freeze=[0,1,2,3,4,5,6,7,8,9], # freeze all the backbone layers
                            pose=24, # increase the weights of keypoint loss (double the default here)
                            label_smoothing=0.1, # label smoothing epsilon
                            dropout=0.3, # add dropout to the head
                            plots=True,
                            augment=True, # turn on augmentation
                            degrees=15,
                            scale=0.5,
                            flipud=0.5)


    # Save the model
    model.save("runs/pose_new/train0/weights/best.pt")