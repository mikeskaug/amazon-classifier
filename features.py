import numpy as np


def summary_stats(image_array, label_array):
    rgbn_hists = []
    ps = []
    for i in range(image_array.shape[0]):
        rgbn_hists.append(rgbn_histograms(image_array[i, :, :, :])[1])
        ps.append(power_spectra(image_array[i, :, :, :])[1])
    return (np.array(rgbn_hists), np.array(ps))


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


def radial_average(image, spectrum):
    '''
    Radially average a 2D power spectrum and return a 1D power spectrum that
    is only a function of distance, r
    '''
    L = max(image.shape)
    # get a list of frequencies up to half the image size
    freqs = np.fft.fftfreq(L)[:int(L/2)]
    # radial distances
    dists = np.sqrt(np.fft.fftfreq(image.shape[0])[:, np.newaxis]**2 + np.fft.fftfreq(image.shape[1])**2)
    hist, bins = np.histogram(dists.ravel(), bins=freqs, weights=spectrum.ravel())
    return (hist, bins)


def radially_averaged_power_spectrum(image):
    '''
    Calculate the 1D radially averaged power spectrum of a 2D image
    This describes the spatial frequencies or correlations that make up the image
    '''
    fft = np.fft.fft2(image)
    spectrum = np.real(np.abs(fft))**2
    (raps, bins) = radial_average(image, spectrum)
    return (raps, bins)


def power_spectra(image):
    power_spectra = []

    for channel in range(image.shape[2]):
        ps, bins = radially_averaged_power_spectrum(image[:, :, channel])
        power_spectra.append(ps)

    return (bins, np.array(power_spectra))
