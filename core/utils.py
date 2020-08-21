# -*- coding:utf-8 -*-

import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt


def cart2sph(x, y, z):
    '''
    Transform Cartesian coordinates to spherical

    Parameters
    ----------
    ```txt
    x: float, X coordinate
    y: float, Y coordinate
    z: float, Z coordinate
    ```
    Returns
    -------
    ```txt
    radius, elevation, azimuth: float -> tuple, Transformed polar coordinates
    ```
    '''
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)  # r
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
    az = m.atan2(y, x)  # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    '''
    Transform polar coordinates to Cartesian

    Parameters
    ----------
    ```txt
    theta   : float, angle value
    rho     : float, radius value
    ```
    Returns
    -------
    ```txt
    X, Y    : float -> tuple, projected coordinates
    ```
    '''
    return rho * m.cos(theta), rho * m.sin(theta)


def azim_proj(pos):
    '''
    Computes the Azimuthal Equidistant Projection of input point in
    3D Cartesian Coordinates. Imagine a plane being placed against
    (tangent to) a globe. If a light source inside the globe projects
    the graticule onto the plane the result would be a planar, or
    azimuthal, map projection.

    Parameters
    ----------
    ```txt
    pos     : list or tuple, position in 3D Cartesian coordinates [x, y, z]
    ```
    Returns
    -------
    ```txt
    X, Y    : float -> tuple, projected coordinates using Azimuthal Equidistant Projection
    ```
    '''
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def load_data(datafile, label=True):
    '''
    Loads the data from MAT file. 
    
    MAT file would be two kinds. `'*.mat'` which contains the feature 
    matrix in the shape of `[nTrials, nChannels, nSamples]` and 
    `'*_label.mat'` which contains the output labels as a vector. 
    Label numbers are assumed to start from 0.

    Parameters
    ----------
    ```txt
    datafile        : str, load data or label from *.mat file (* in '*.mat' 
                      and '*_label.mat' are the same, pls let datafile = '*.mat')
    label           : bool, if True: load label, else: load data
    ```
    Returns
    -------
    ```txt
    data or label   : ndarray
    ```
    '''
    if label:
        datafile = datafile[:-4] + '_label.mat'
        print('Loading data from %s' % (datafile))
        dataMat = sio.loadmat(datafile, mat_dtype=True)
        print('Data loading complete. Shape is %r' %
              (dataMat['classlabel'].shape, ))
        # Class labels should start from 0
        return dataMat['classlabel'] - 1
    else:  # [nChannels, nSamples, nTrials]
        print('Loading data from %s' % (datafile))
        dataMat = sio.loadmat(datafile, mat_dtype=True)
        dataMat['s'] = dataMat['s'].swapaxes(1, 2)
        dataMat['s'] = dataMat['s'].swapaxes(0, 1)
        print('Data loading complete. Shape is %r' % (dataMat['s'].shape, ))
        return dataMat['s']  # [nTrials, nChannels, nSamples]


def augment_EEG(data, stdmult, pca=False, nComponents=2):
    '''
    Augment data by adding normal noise to each feature.

    Parameters
    ----------
    ```txt
    data        : EEG feature data as a matrix (nSamples, nFeatures)
    stdmult     : Multiplier for std of added noise
    pca         : if True will perform PCA on data and add noise proportional to PCA components.
    nComponents : Number of components to consider when using PCA.
    ```
    Returns
    -------
    ```txt
    augData     : Augmented data as a matrix (nSamples, nFSeatures)
    ```
    '''
    augData = np.zeros(data.shape)
    if pca:
        pca = PCA(nComponents=nComponents)
        pca.fit(data)
        components = pca.components_
        variances = pca.explained_variance_ratio_
        coeffs = np.random.normal(scale=stdmult,
                                  size=pca.nComponents) * variances
        for s, sample in enumerate(data):
            augData[s, :] = sample + (components * coeffs.reshape(
                (nComponents, -1))).sum(axis=0)
    else:
        # Add Gaussian noise with std determined by weighted std of each feature
        for f, feat in enumerate(data.transpose()):
            augData[:,
                    f] = feat + np.random.normal(scale=stdmult * np.std(feat),
                                                 size=feat.size)
    return augData


def filterbank(data, srate=250, start=4, stop=38, window=4, step=2):
    '''
    Process raw data with filter-bank.

    Parameters
    ----------
    ```txt
    data    : ndarray, raw data, shapes as [nTrials, nChannels, nSamples]
    srate   : int, the sample rate of raw data, default is 250
    start   : int, frequency where the filter-bank begins, default is 4
    stop    : int, frequency where the filter-bank ends, default is 38
    window  : int, the bandwidth of one filter in the filter-bank, default is 4
    step    : int, the interval of each neighbouring filter in the filter-bank, default is 2
    ```
    Returns
    -------
    ```txt
    FBdata  : ndarray, data after filter-bank, shapes (nTrials, nChannels, nSamples, nColors)
    ```
    '''
    nTrials, nChannels, nSamples = data.shape
    FBdata = []
    for beg in range(start, stop - window + 1, step):
        end = beg + window
        b, a = signal.butter(4, [beg / srate * 2, end / srate * 2], 'bandpass')
        FBdata.append(signal.filtfilt(b, a, data, axis=-1))
    #now np.array(FBdata) shapes as[nColors, nTrials, nChannels, nSamples]
    FBdata = np.swapaxes(np.array(FBdata), 0, 1)
    FBdata = np.swapaxes(FBdata, 1, 2)
    FBdata = np.swapaxes(FBdata, 2, 3)
    print('Data filterbank complete. Shape is %r.' % (FBdata.shape, ))
    return FBdata


def load_or_gen_filterbank_data(filepath,
                                beg=0.,
                                end=4.,
                                srate=250,
                                start=4,
                                stop=38,
                                window=4,
                                step=4):
    '''
    load or generate data with filter-bank.

    Parameters
    ----------
    ```txt
    filepath: str, path of raw data file, and data shape is [nTrials, nChannels, nSamples]
    beg     : float, second when imegery tasks begins
    end     : float, second when imegery tasks ends
    srate   : int, the sample rate of raw data, default is 250
    start   : int, frequency where the filter-bank begins, default is 4
    stop    : int, frequency where the filter-bank ends, default is 38
    window  : int, the bandwidth of one filter in the filter-bank, default is 4
    step    : int, the interval of each neighbouring filter in the filter-bank, default is 2
    ```
    Returns
    -------
    ```txt
    FBdata  : ndarray, data after filter-bank, shapes as [nTrials, nChannels, nSamples, nColors]
    ```
    '''
    if os.path.exists(filepath[:-4] + '_fb.mat'):
        print('Loading data from %s' % (filepath[:-4] + '_fb.mat'))
        data = sio.loadmat(filepath[:-4] + '_fb.mat')['fb']
        print('Load filterbank data complete. Shape is %r.' % (data.shape, ))
    else:
        data = filterbank(load_data(filepath, label=False),
                          srate=srate,
                          start=start,
                          stop=stop,
                          window=window,
                          step=step)
        data = data[:, :, round(beg * srate):round(end * srate), :]
        print('Load filterbank data complete. Shape is %r.' % (data.shape, ))
        sio.savemat(filepath[:-4] + '_fb.mat', {'fb': data})
        print('Save filterbank data[\'fb\'] complete. To %s' %
              (filepath[:-4] + '_fb.mat'))

    return data


def load_locs(filepath=None):
    '''
    load data of electrodes' 3D location and name.

    Parameters
    ----------
    ```txt
    filepath: str, path of electrodes' 3D location data file, default is None
    ```
    Returns
    -------
    ```txt
    locs    : ndarray, data of electrodes' 3D location, shapes (nChannels, 3)
    names   : ndarray, name of electrodes, shapes (nChannels, )
    ```
    '''
    if filepath is None:
        filepath = os.path.join('data', '22scan_locs.mat')
    if not os.path.exists(filepath):
        raise FileNotFoundError('File not found in {}'.format(filepath))
    locs = sio.loadmat(filepath)['locs']
    names = sio.loadmat(filepath)['name']
    return locs, names


def interestingband(data, axis=-1, srate=250, swapaxes=True):
    '''
    Filter raw signal to five interesting bands - theta, alpha, low beta, high beta,
     and gamma.
    theta: 2-8Hz
    alpha: 8-12Hz
    low beta: 12-20Hz
    high beta: 20-30Hz
    gamma: 30-60Hz

    Parameters
    ----------
    ```txt
    data    : ndarray, raw data, shapes (nTrials, nChannels, nSamples)
    axis    : int, which axis should be filtered, default is -1
    srate   : int, the sample rate of raw data, default is 250
    swapaxes: bool, deciding the interestingband axis at the end or the begining, default is True
    ```
    Returns
    -------
    ```txt
    IBdata  : ndarray, data after filter-bank, shapes (nTrials, nChannels, nSamples, nColors)
    ```
    '''
    IBdata = []
    b, a = signal.butter(1, [2, 8], 'bandpass', fs=srate)  # theta
    IBdata.append(signal.filtfilt(b, a, data, axis=axis))

    b, a = signal.butter(2, [8, 12], 'bandpass', fs=srate)  # alpha
    IBdata.append(signal.filtfilt(b, a, data, axis=axis))

    b, a = signal.butter(3, [12, 20], 'bandpass', fs=srate)  # low beta
    IBdata.append(signal.filtfilt(b, a, data, axis=axis))

    b, a = signal.butter(3, [20, 30], 'bandpass', fs=srate)  # high beta
    IBdata.append(signal.filtfilt(b, a, data, axis=axis))

    b, a = signal.butter(4, [30, 60], 'bandpass', fs=srate)  # gamma
    IBdata.append(signal.filtfilt(b, a, data, axis=axis))

    # now np.array(IBdata) shapes as[nColors, nTrials, nChannels, nSamples]
    IBdata = np.array(IBdata)
    if swapaxes:
        IBdata = np.swapaxes(IBdata, 0, 1)
        IBdata = np.swapaxes(IBdata, 1, 2)
        IBdata = np.swapaxes(IBdata, 2, 3)
    print('Data filterbank complete. Shape is %r.' % (IBdata.shape, ))
    return IBdata


def load_or_gen_interestingband_data(filepath, beg=0., end=4., srate=250):
    '''
    load or generate data with interesting-band filters.

    Parameters
    ----------
    ```txt
    filepath: str, path of raw data file, and data shapes (nTrials, nChannels, nSamples)
    beg     : float, second when imegery tasks begins
    end     : float, second when imegery tasks ends
    srate   : int, the sample rate of raw data, default is 250
    ```
    Returns
    -------
    ```txt
    IBdata  : ndarray, data after interesting-band filters, shapes (nTrials, nChannels, nSamples, nColors)
    ```
    '''
    if os.path.exists(filepath[:-4] + '_ib.mat'):
        print('Loading data from %s' % (filepath[:-4] + '_ib.mat'))
        data = sio.loadmat(filepath[:-4] + '_ib.mat')['ib']
        print('Load interestingband data complete. Shape is %r.' %
              (data.shape, ))
    else:
        data = interestingband(load_data(filepath, label=False), srate=srate)
        data = data[:, :, round(beg * srate):round(end * srate), :]
        print('Load interestingband data complete. Shape is %r.' %
              (data.shape, ))
        sio.savemat(filepath[:-4] + '_ib.mat', {'ib': data})
        print('Save interestingband data[\'ib\'] complete. To %s' %
              (filepath[:-4] + '_ib.mat'))

    return data


def highpassfilter(data, Wn=4, srate=250):
    b, a = signal.butter(4, Wn=Wn, btype='highpass', fs=srate)
    new_data = []
    for e in data:
        new_data.append(signal.filtfilt(b, a, e))
    return np.asarray(new_data)


def bandpassfilter(data, Wn=[.5, 100], srate=250):
    b, a = signal.butter(4, Wn=Wn, btype='bandpass', fs=srate)
    new_data = []
    for e in data:
        new_data.append(signal.filtfilt(b, a, e))
    return np.asarray(new_data)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma