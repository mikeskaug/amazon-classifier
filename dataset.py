import os
import pandas as pd

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


def get_labels(labels_file):
    labels = pd.read_csv(labels_file)

    # add a 'one-hot' vector of labels to each image
    for label in unique_labels(labels):
        labels[label] = labels['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

    return labels
