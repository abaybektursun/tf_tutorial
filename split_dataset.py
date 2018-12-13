import os
import shutil
import facenet
import numpy as np

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

# t - test or train
def move_to(paths, t):
    for path in paths:
        new_path = path.replace('cropped',t)
        new_dir  = os.path.dirname(new_path)
        if not os.path.isdir(new_dir): os.makedirs(new_dir)
        shutil.move(path, new_path)



data_dir = "data/dataset/cropped"

dataset_tmp = facenet.get_dataset(data_dir)
train_set, test_set = split_dataset(dataset_tmp, 1, 10)

for img in test_set:
    move_to(img.image_paths, 'test')
for img in train_set:
    move_to(img.image_paths, 'train')



