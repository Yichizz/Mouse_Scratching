from ultralytics import YOLO
import warnings

# Ignore warnings from the ultralytics module
#warnings.filterwarnings("ignore", module="ultralytics")

# Load the default YOLOv8 model
# model = YOLO("yolov8n-pose.yaml").load('weights/yolov8n-pose.pt')
model = YOLO("yolov8n-pose.yaml").load('runs/pose/train3/weights/last.pt')

if __name__ == '__main__':
# Train the model
    results = model.train(data="dataset_reannotated/mouse-pose.yaml",
                            epochs=2000,
                            patience=100, # early stopping patience
                            batch=48, # autoselect using 60% GPU memory
                            device=0,
                            verbose=True,
                            cos_lr=True, # cosine learning rate schedule
                            lr0=1e-4, # initial learning rate
                            lrf=5e-6, # small final learning rate for finer tuning
                            freeze=[0,1,2,3,4,5,6,7,8,9], # freeze all the backbone layers
                            pose=24, # increase the weights of keypoint loss (double the default here)
                            label_smoothing=0.1, # label smoothing epsilon
                            dropout=0.3, # add dropout to the head
                            optimizer="AdamW", 
                            degrees=15,
                            flipud=0.5,
                            save=True, # save the model after training
                            amp=True)# automatic mixed precision for faster training)


    # Save the model
    # model.save("runs/pose/train/weights/best.pt")