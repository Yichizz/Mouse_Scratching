import os
from tqdm import tqdm
import shutil

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

def pose2detect(label_dir):
    """
    Convert the data in pose form to detection form
    Args:
        label_dir: str, path to the txt label directory, only left first 5 numbers in the label file
    """                                                                                             
    for split in ['train','val','test']:
        labels = os.path.join(label_dir, split)
        for l in tqdm(os.listdir(labels)):
            with open(os.path.join(labels, l), 'r') as f:
                line = f.readline()
            with open(os.path.join(labels, l), 'w') as f:
                f.write(' '.join(line.split()[:5]))
    return None                                                                                                                                        
    
