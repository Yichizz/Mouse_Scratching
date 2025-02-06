import os
from tqdm import tqdm
import shutil
import numpy as np

def pose2cls(data_dir, label_dir, output_dir, cls = ['mouse','scratching_mouse','paw_licking_mouse']):
    """
    Convert the data in pose form to classification form
    Args:
        data_dir: str, path to the image data directory, shutil to subdirectories in the directory
        label_dir: str, path to the txt label directory, shutil to subdirectories in the directory
        output_dir: str, path to the output directory
        cls: list, list of classes, subdirectories will be created in the output directory
    """                                                                                                                                                                                                                                      
    if not os.path.exists(output_dir):
        print(f'Creating output directory {output_dir}')
        os.makedirs(output_dir)
    
    for split in ['train','val','test']:
        # create subdirectories for each split
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        # create subdirectories for each class
        for c in cls:
            os.makedirs(os.path.join(output_dir, split, c), exist_ok=True)
        data = os.path.join(data_dir, split)
        labels = os.path.join(label_dir, split)
        for l in tqdm(os.listdir(labels)):
            # open the label file and see which class it belongs to, i.e. the first number in the file
            with open(os.path.join(labels, l), 'r') as f:
                cls_label = int(f.readline().split()[0])
            # move the image to the corresponding class directory
            shutil.move(os.path.join(data, l.replace('txt','png')), os.path.join(output_dir, split, cls[cls_label], l.replace('txt','png')))
    return None

def pose2detect(label_dir, background_dir = None, image_dir = None, val = 0.2, test = 0.1):
    """
    Convert the data in pose form to detection form
    Args:
        label_dir: str, path to the txt label directory, only left first 5 numbers in the label file
        background_dir: str, path to the background image directory, if None, no background images will be added
        image_dir: str, path to the image directory, needed if background_dir is not None
        val: float, proportion of validation data, needed if background_dir is not None
        test: float, proportion of test data, optional
    """                                                                                             
    for split in ['train','val','test']:
        labels = os.path.join(label_dir, split)
        for l in tqdm(os.listdir(labels)):
            with open(os.path.join(labels, l), 'r') as f:
                line = f.readline()
            with open(os.path.join(labels, l), 'w') as f:
                f.write(' '.join(line.split()[:5]))
    
    # randomly add background images to the train, val, and test sets
    if background_dir is not None:
        assert val is not None, 'val proportion must be provided'
        assert os.path.exists(background_dir), 'background directory does not exist'
        assert image_dir is not None, 'image directory must be provided'
        assert os.path.exists(image_dir), 'image directory does not exist'
        background_images = os.listdir(background_dir)
        # background_images = [background_image for background_image in background_images if background_image.endswith('.png')]
        # randomly select background images for each split
        print(len(background_images))
        np.random.seed(0)
        background_images = np.random.permutation(background_images)
        print(len(background_images))
        val_images = background_images[:int(val*len(background_images))]
        test_images = background_images[int(val*len(background_images)):int((val+test)*len(background_images))]
        train_images = background_images[int((val+test)*len(background_images)):]
        # move the images to the corresponding directories
        for i in val_images:
            shutil.copy(os.path.join(background_dir, i), os.path.join(image_dir, 'val', i))
        for i in test_images:
            shutil.copy(os.path.join(background_dir, i), os.path.join(image_dir, 'test', i))
        for i in train_images:
            shutil.copy(os.path.join(background_dir, i), os.path.join(image_dir, 'train', i))
        print('background images added.')
        
    return None                                                                                                             
    
