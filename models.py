from sklearn.ensemble import RandomForestClassifier
import numpy as np

from dataset import get_data_sets
from features import summary_stats


def random_forest_clf(features, labels, **kwargs):
    '''
    Builds a random forest classifier
    "features" is a numpy array of shape [num_samples, num_features]
    "labels" is a numpy array of shape [num_sampes, num_categories]
    "kwargs" are additional key word arguments to supply to the RandomForestClassifier() constructor
    '''
    clf = RandomForestClassifier(**kwargs)
    clf = clf.fit(features, labels)

    return clf


def predict():
    data = get_data_sets(data_dir='./data/train-tif-v2')
    (images, labels) = data['train'].get_image_batch(100)
    (rbgn_hists, power_spectra) = summary_stats(images, labels)
    ps_shape = power_spectra.shape
    rbgn_shape = rbgn_hists.shape

    # flatten the color channel dimensions so there is one row per image
    ps_reshaped = power_spectra.reshape((ps_shape[0], ps_shape[1] * ps_shape[2]))
    rbgn_reshaped = rbgn_hists.reshape((rbgn_shape[0], rbgn_shape[1] * rbgn_shape[2]))

    # concatenate the two statistics to get one long row of "features" per image
    features = np.hstack(rbgn_reshaped, ps_reshaped)

    rf_classifier = random_forest_clf(features, labels)

if __name__ == "__main__":
    predict()
