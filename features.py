import numpy as np


def summary_stats(image_array, label_array):
    rgb_hists = [rgbn_histograms(image_array[i, :, :, :])[0] for i in range(image_array.shape[0])]
    return rgb_hists


def rgbn_histograms(image, bins=32, d_range=2**16):
    '''
    Calculate the distribution of intensity values in each r, g, b, NIR channel
    "image" is a numpy array with shape [height, width, channels]

    Returns an array of histograms with shape [channels, bins]
    '''
    num_channels = image.shape[2]
    bin_labels = [d_range / bins * i for i in range(bins)]
    hists = np.zeros((num_channels, bins))

    for channel in range(num_channels):
        hists[channel, :], _ = np.histogram(image[:, :, channel], bins)

    return (bin_labels, hists)
