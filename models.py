from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
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


def reshape_features(*features):
    '''
    each "feature" in features is a numpy array of shape [num_samples, num_channels, num_metrics]
    Each of these should be flattened so there is one row for each sample and then concatenated
    horizontally.
    '''
    reshaped = []
    for feature in features:
        shape = feature.shape
        reshaped.append(feature.reshape((shape[0], shape[1] * shape[2])))

    return np.hstack(reshaped)


def predict():
    print('preparing data...')
    data = get_data_sets(data_dir='./data/train-tif-v2')
    (images, labels) = data['train'].get_image_batch(100)
    (rgbn_hists, power_spectra) = summary_stats(images, labels)
    train_features = reshape_features(rgbn_hists, power_spectra)

    print('training classifier...')
    rf_classifier = random_forest_clf(train_features, labels)

    print('making predictions...')
    (eval_images, eval_labels) = data['validation'].get_image_batch(100)
    (rgbn_hists, power_spectra) = summary_stats(eval_images, eval_labels)
    eval_features = reshape_features(rgbn_hists, power_spectra)
    predicted_labels = rf_classifier.predict(eval_features)

    f2 = fbeta_score(eval_labels, predicted_labels, beta=2, average='samples')
    print('F2 score: {}'.format(f2))


if __name__ == "__main__":
    predict()
