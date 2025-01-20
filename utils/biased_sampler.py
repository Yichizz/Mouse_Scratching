import os
import numpy as np

def reduce_redundancy(train_images, train_labels):
    """
    Reduce redundancy in the training data by randomly removing non-scratching images. 
    To keep the number of images for class 0,1,2 balanced, we will remove redundant images in class 0.
    args:
        train_images: directory of training images (.png)
        train_labels: directory of training labels stored in YOLO format (.txt)
    
    return:
        None
    """
    # get the number of images in each class
    class_count = [0, 0, 0]
    for label_file in os.listdir(train_labels):
        with open(os.path.join(train_labels, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_count[int(line.split()[0])] += 1
    print(f'Number of images in each class: {class_count}')

    # get the list of images in class 0
    class_0_images = []
    for label_file in os.listdir(train_labels):
        with open(os.path.join(train_labels, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if int(line.split()[0]) == 0:
                    class_0_images.append(label_file[:-4])
                    break

    # remove redundant images in class 0 to the same number as the maximum number of images in class 1 and 2
    max_count = int(max(class_count[1], class_count[2])*1.5)
    np.random.seed(0)
    class_0_images_to_remove = np.random.choice(class_0_images, len(class_0_images)-max_count, replace=False)
    print(f'Number of images in class 0 after reduction: {len(class_0_images) - len(class_0_images_to_remove)}')

    # remove redundant images and corresponding labels in class 0
    for label_file in os.listdir(train_labels):
        if label_file[:-4] in class_0_images_to_remove:
            os.remove(os.path.join(train_labels, label_file))
            image_file = label_file[:-4] + '.png'
            os.remove(os.path.join(train_images, image_file))
    print('Redundant images removed.')
    return None