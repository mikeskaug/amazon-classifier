import os
import math
import random

import pandas as pd
import rasterio
import numpy as np

# Prepare the data for training and evaluating the model.

ROOT = os.path.abspath("./data/")
JPEG_DIR = os.path.join(ROOT, 'train-jpg')
TIF_DIR = os.path.join(ROOT, 'train-tif-v2')
LABEL_CSV = os.path.join(ROOT, 'train_v2.csv')

SET_FRACTIONS = {'train': 0.6, 'validation': 0.2, 'test': 0.2}


def unique_labels(labels):
    # Build list of unique labels
    label_list = [label for tag_str in labels.tags.values for label in tag_str.split(' ')]
    unique_set = set(label_list)
    return list(unique_set)


def get_labels(labels_file=LABEL_CSV):
    names_and_labels = pd.read_csv(labels_file)

    # add a 'one-hot' vector of labels to each image
    for label in unique_labels(names_and_labels):
        names_and_labels[label] = names_and_labels['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

    return names_and_labels


def one_hot(tags_str, labels):
    one_hot = [1 if label in tags_str.split(' ') else 0 for label in labels]
    return one_hot


def load_tiff(filename):
    '''Return a 4D (r, g, b, nir) numpy array with the data in the specified TIFF filename.'''
    with rasterio.open(filename) as src:
        b, g, r, nir = src.read()
        return np.dstack([r, g, b, nir])


def get_data_sets(data_dir=TIF_DIR, set_fractions=SET_FRACTIONS):
    # The callable that will return sets of images to be used in training, validation and testing
    #
    # OUTPUT: a dictionary containing a Dataset object for each set [train, validation, test].

    names_and_tags = pd.read_csv(LABEL_CSV)
    labels = unique_labels(names_and_tags)
    num_images = len(names_and_tags)
    data_sets = {}
    start_idx = 0

    for set_name in set_fractions.keys():
        set_size = math.floor(num_images * set_fractions[set_name])
        image_set = []

        for idx in range(start_idx, start_idx + set_size):
            image_path = os.path.join(TIF_DIR, names_and_tags.iloc[idx]['image_name'] + '.tif')
            label_vector = one_hot(names_and_tags.iloc[idx]['tags'], labels)
            image_set.append((image_path, label_vector))

        start_idx += set_size
        random.shuffle(image_set)
        data_sets[set_name] = Dataset(image_set, labels)

    return data_sets


class Dataset:
    '''
    A class that stores image paths and labels and provides useful methods
    for delivering batches of training data
    '''

    def __init__(self, data, labels):
        self.batch_idx = 0
        self.data = data
        self.labels = labels
        self.num_examples = len(data)
        (self.IMG_X, self.IMG_Y, self.IMG_D) = load_tiff(data[0][0]).shape
        self.num_labels = len(data[0][1])

    def get_image_batch(self, batch_size, transform=lambda x: x):
        # return a new batch of images and labels
        if self.batch_idx + batch_size > self.num_examples:
            self.batch_idx = 0
            random.shuffle(self.data)

        image_array = np.empty(shape=(batch_size, self.IMG_X, self.IMG_Y, self.IMG_D), dtype=np.float32)
        label_array = np.empty(shape=(batch_size, self.num_labels), dtype=np.int32)

        for i, j in zip(range(self.batch_idx, self.batch_idx + batch_size), range(batch_size)):
            # read the image file
            image_array[j, :, :, :] = load_tiff(self.data[i][0])
            label_array[j, :] = self.data[i][1]
            self.batch_idx += 1

        return transform((image_array, label_array))

    def batch_generator(self, batch_size, transform=lambda x: x):
        '''
        A generator that will yield batches of training data
        NOTE: this generator never terminates, so don't do something like list(batch_generator)

        transform: an optional function that modifies the raw images and labels returned by get_image_batch
        and returns a new tupel (images, labels)
        '''
        while True:
            (images, labels) = self.get_image_batch(batch_size, transform)
            yield (images, labels)
