#given model, path to images, and path to save json file, this script will automatically annotate the images and save the json file
# Example usage
# from utils.auto_annotater import AutoAnnotater
#
# model_path = './best.pt'
# image_path = './frames'
# save_path = './labels'

# annotater = AutoAnnotater(model_path, image_path, save_path)
# annotater.annotate_images()

import torch
import os
import json
from tqdm import tqdm
from ultralytic import YOLO

class AutoAnnotater:
    def __init__(self, model_path, image_path, save_path):
        assert os.path.exists(model_path), 'Model path does not exist'
        assert os.path.exists(image_path), 'Image path does not exist'

        self.model = YOLO(model_path)
        self.image_path = image_path
        self.save_path = save_path
        self.labels = []
        
        self.create_save_dir()

    def create_save_dir(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print('Created directory', self.save_path)

    def 
        
    def annotate_images(self):
        for image in tqdm(os.listdir(self.image_path)):
            if image.endswith('.jpg') or image.endswith('.png'):
                image_path = os.path.join(self.image_path, image)
            boxes = self.model.predict(image_path)
            self.labels.append({'image': image, 'boxes': boxes})
        
        with open(self.save_path, 'w') as f:
            json.dump(self.labels, f)
            
        print('Labels saved to', self.save_path)