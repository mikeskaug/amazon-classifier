from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import numpy as np

from dataset import get_data_sets


def cnn_model():
    # use a pre-trained VGG16 model without the final dense layer
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(256, 256, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add an untrained fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- there are 17 possible labels
    predictions = Dense(17, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def fbeta(y_true, y_pred, beta=2, threshold_shift=0):
    # ensure that predictions are in the range [0, 1]
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))


def find_f2score_threshold(y_true, y_pred, try_all=False):
    best = 0
    best_score = -1
    totry = np.arange(0, 1, 0.005) if try_all is False else np.unique(y_pred)
    for t in totry:
        score = fbeta(y_true, y_pred, beta=2)
        if score > best_score:
            best_score = score
            best = t
    return best


def freeze_layers(model, freeze_layers=[]):
    for i, layer in enumerate(model.layers):
        layer.trainable = False if i in freeze_layers else True

    return model


def train(model, data_dir='./data/train-tif-v2', batch_size=25, num_epochs=10):
    data = get_data_sets(data_dir)
    # rescale input pixel values to match those in VGG16 paper?
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', fbeta])

    def image_mod(batch):
        images, labels = batch
        # drop the last NIR channel
        images = images[:, :, :, :-1]
        return (images, labels)

    model.fit_generator(data['train'].batch_generator(batch_size, mod=image_mod),
                        steps_per_epoch=data['train'].num_examples / batch_size,
                        epochs=num_epochs)
    return model


if __name__ == "__main__":
    model = cnn_model()
    model = freeze_layers(model, list(range(len(model.layers) - 3)))
    model = train(model)
