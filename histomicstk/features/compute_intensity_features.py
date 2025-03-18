"""Compute intensity features in labeled image."""
import warnings

import numpy as np


def compute_intensity_features(
        im_label, im_intensity, num_hist_bins=10,
        rprops=None, feature_list=None):
    """Calculate intensity features from an intensity image.

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    im_intensity : array_like
        Intensity image.

    num_hist_bins: int, optional
        Number of bins used to computed the intensity histogram of an object.
        Histogram is used to energy and entropy features. Default is 10.

    rprops : output of skimage.measure.regionprops, optional
        rprops = skimage.measure.regionprops( im_label ). If rprops is not
        passed then it will be computed inside which will increase the
        computation time.

    feature_list : list, default is None
        list of intensity features to return.
        If none, all intensity features are returned.

    Returns
    -------
    fdata: pandas.DataFrame
        A pandas dataframe containing the intensity features listed below for
        each object/label.

    Notes
    -----
    List of intensity features computed by this function:

    Intensity.Min : float
        Minimum intensity of object pixels.

    Intensity.Max : float
        Maximum intensity of object pixels.

    Intensity.Mean : float
        Mean intensity of object pixels

    Intensity.Median : float
        Median intensity of object pixels

    Intensity.MeanMedianDiff : float
        Difference between mean and median intensities of object pixels.

    Intensity.Std : float
        Standard deviation of the intensities of object pixels

    Intensity.IQR: float
        Inter-quartile range of the intensities of object pixels

    Intensity.MAD: float
        Median absolute deviation of the intensities of object pixels

    Intensity.Skewness : float
        Skewness of the intensities of object pixels. Value is 0 when all
        intensity values are equal.

    Intensity.Kurtosis : float
        Kurtosis of the intensities of object pixels. Value is -3 when all
        values are equal.

    Intensity.HistEnergy : float
        Energy of the intensity histogram of object pixels

    Intensity.HistEntropy : float
        Entropy of the intensity histogram of object pixels.

    References
    ----------
    .. [#] Daniel Zwillinger and Stephen Kokoska. "CRC standard probability
       and statistics tables and formulae," Crc Press, 1999.

    """
    import pandas as pd
    from skimage.measure import regionprops

    default_feature_list = [
        'Intensity.Min',
        'Intensity.Max',
        'Intensity.Mean',
        'Intensity.Median',
        'Intensity.MeanMedianDiff',
        'Intensity.Std',
        'Intensity.IQR',
        'Intensity.MAD',
        'Intensity.Skewness',
        'Intensity.Kurtosis',
        'Intensity.HistEnergy',
        'Intensity.HistEntropy',
    ]

    # List of feature names
    if feature_list is None:
        feature_list = default_feature_list
    else:
        assert all(j in default_feature_list for j in feature_list), \
            'Some feature names are not recognized.'

    # compute object properties if not provided
    if rprops is None:
        rprops = regionprops(im_label)

    # collect features for each object in a list
    numLabels = len(rprops)
    results = []

    for i in range(numLabels):
        region = rprops[i]
        if rprops[i] is None:
            continue

        row = {}

        # Get pixel intensities for the current region
        pixelIntensities = im_intensity[region.coords[:, 0], region.coords[:, 1]]

        if pixelIntensities.size == 0:
            # If no pixels, skip or fill NaNs
            results.append({feat: np.nan for feat in feature_list})
            continue

        meanIntensity = np.mean(pixelIntensities)
        medianIntensity = np.median(pixelIntensities)
        stdIntensity = np.std(pixelIntensities)

        # Populate features conditionally
        if 'Intensity.Min' in feature_list:
            row['Intensity.Min'] = np.min(pixelIntensities)
        if 'Intensity.Max' in feature_list:
            row['Intensity.Max'] = np.max(pixelIntensities)
        if 'Intensity.Mean' in feature_list:
            row['Intensity.Mean'] = meanIntensity
        if 'Intensity.Median' in feature_list:
            row['Intensity.Median'] = medianIntensity
        if 'Intensity.MeanMedianDiff' in feature_list:
            row['Intensity.MeanMedianDiff'] = meanIntensity - medianIntensity
        if 'Intensity.Std' in feature_list:
            row['Intensity.Std'] = stdIntensity
        if 'Intensity.IQR' in feature_list:
            row['Intensity.IQR'] = _fast_iqr(pixelIntensities) 
        if 'Intensity.MAD' in feature_list:
            row['Intensity.MAD'] = np.median(np.abs(pixelIntensities - medianIntensity))
        if 'Intensity.Skewness' in feature_list:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                row['Intensity.Skewness'] = _fast_skew(pixelIntensities, meanIntensity, stdIntensity)
        if 'Intensity.Kurtosis' in feature_list:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                row['Intensity.Kurtosis'] = _fast_kurtosis(pixelIntensities, meanIntensity, stdIntensity)

        # Histogram-based features (energy and entropy)
        if any(j in feature_list for j in ['Intensity.HistEntropy', 'Intensity.HistEnergy']):
            hist, _ = np.histogram(pixelIntensities, bins=num_hist_bins)
            prob = hist / np.sum(hist, dtype=np.float32)

            if 'Intensity.HistEntropy' in feature_list:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    row['Intensity.HistEntropy'] = _fast_entropy(prob) 
            if 'Intensity.HistEnergy' in feature_list:
                row['Intensity.HistEnergy'] = np.sum(prob ** 2)

        # Add the row to the results list
        results.append(row)

    # After the loop, create the DataFrame
    fdata = pd.DataFrame(results)

    # Ensure all requested features are in the DataFrame (fill missing with NaNs)
    for feat in feature_list:
        if feat not in fdata.columns:
            fdata[feat] = np.nan

    # Reorder columns to match feature_list
    fdata = fdata[feature_list]

    return fdata

def _fast_iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

def _fast_skew(x, _mean, _std):
    n = len(x)
    if _std == 0: return 0.0
    return np.sum((x - _mean)**3) / n / _std**3

def _fast_kurtosis(x, _mean, _std):
    n = len(x)
    if _std == 0: return -3.0
    return np.sum((x - _mean)**4) / n / _std**4 - 3

def _fast_entropy(prob):
    prob = np.asarray(prob)
    prob = prob[prob > 0]
    log_p = np.log(prob)
    return -np.sum(prob * log_p)