import mne
from mne.channels import montage
from mne.io import reference
import numpy as np
from scipy.sparse import data
import os
import matplotlib.pyplot as plt
from autoreject import Ransac
import ntpath

EXP_DUR_SECONDS = int(7*60)

EXP_MARKERS = {
    'rest': [2, 4],
    'exp': [2, 6],
    'drmt': [None, 4],
    'vol': [2, 4]
}

def preprocessing(pth, verbose=False):

    # open/load data
    raw = mne.io.read_raw_brainvision(pth, preload=True, verbose=verbose)
    
    # Get the correct start/stop markers the indicate 
    # beginning and end of session
    markers = get_markers_by_path(pth)

    # Crop out the piece of data in raw that is of interest
    raw = crop_by_marker(raw, markers)

    # Map channel types
    raw = map_channels(raw)

    # setting electrode positions
    raw.set_montage('standard_1020',  on_missing='warn', verbose=verbose)  # verbose for less output text

    # first filtering
    raw.filter(0.005, 45, n_jobs=-1, verbose=verbose)
    
    # Resampling for lower computation times
    raw.resample(100)
    
    # Bad channel rejection
    raw = bad_channel_detection_interpolation(raw)
    
    # reference
    raw.set_eeg_reference(['TP9', 'TP10'], verbose=verbose)

    # ICA 
    raw = eog_ica(raw, verbose=verbose)
    
    return raw

def raw_to_epochs(raw):
    ''' Convert raw data object to an epochs data object
    Parameters
    ----------
    raw : mne.io.Raw, the raw data object from MNE
    
    Return
    ------
    epochs : mne.Epochs, the epochs data object from MNE

    '''
    epochs = mne.EpochsArray(np.expand_dims(raw._data, axis=0), raw.info)
    return epochs

def get_data(raw, start_marker, stop_marker, picks):
    ''' Select a range of data and return it as a numpy.ndarray
    Parameters
    ----------
    raw : mne.io.Raw, the mne data object
    start_marker : int, the marker that indicates the start of the segment of interest
    stop_marker : int, the marker that indicates the end of the segment of interest
    picks : list/tuple/str, the channels to pick. Can be list of channel names, channel type(s) etc.

    Return
    ------
    data : numpy.ndarray, the data matrix of the selected time
    '''
    sfreq = raw.info['sfreq']
    events = mne.events_from_annotations(raw)[0]

    if stop_marker is not None:
        stop_latency = events[np.where(events[:, 2]==stop_marker)[0][0]][0]
    else:
        start_latency = events[np.where(events[:, 2]==start_marker)[0][0]][0]
        stop_latency = start_latency + raw.info['sfreq']*EXP_DUR_SECONDS
    
    if start_marker is not None:
        start_latency = events[np.where(events[:, 2]==start_marker)[0][0]][0]
    else:
        # Start marker can be missing
        # In that case use the stop marker and go back 7 minutes
        start_latency = stop_latency - raw.info['sfreq']*EXP_DUR_SECONDS

    print(f'dur: {((stop_latency-start_latency) / sfreq) / 60:.2f} min')

    data = raw.get_data(picks=picks, start=int(start_latency), stop=int(stop_latency))
    
    return data


def map_channels(raw):
    ''' Assign correct channel types (eeg, eog, resp) to various channel names.
    Parameters
    ----------
    raw : mne.io.Raw, holds the EEG data

    Return
    ------
    raw : mne.io.Raw, raw object with correct channel type assignments
    '''
    mapping = dict()
    for channel in raw.ch_names:
        if channel == 'Resp':
            mapping[channel] = 'resp'
        elif channel == 'VEOG':
            mapping[channel] = 'eog'
        else:
            mapping[channel] = 'eeg'

    raw.set_channel_types(mapping)
    return raw

def eog_ica(raw, verbose=False):
    ''' Remove Blink artifacts using ICA.
    
    Parameters
    ----------
    raw : mne.io.Raw, raw object that holds the EEG data

    Return
    ------
    raw : mne.io.Raw, cleaned raw object
    '''
    
    # Filter data to remove drifts which helps ICA
    raw_filt = raw.copy().filter(1, None, verbose=verbose)
    
    # Add an improvised HEOG channel to the data object
    raw_filt = add_heog(raw_filt)

    n_components = 15
    method = "picard"
    ica = mne.preprocessing.ICA(n_components=n_components, 
        method=method, verbose=verbose)
    
    ica.fit(raw_filt, decim=3, reject=dict(eeg=100e-6), 
        verbose=verbose, picks=['eeg'])

    eog_indices, eog_scores = ica.find_bads_eog(raw_filt, 
        verbose=verbose, measure='correlation', threshold=0.6)
        
    ica.exclude = eog_indices
    if verbose:
        print(f'excluding component(s) ', eog_indices)
    return ica.apply(raw, verbose=verbose)


def add_heog(raw, based_on=['Fp1', 'Fp2']):
    ''' Add a improvised HEOG electrode to the raw object using two nearby EEG electrodes.
    Parameters
    ----------
    raw : mne.io.Raw, mne object that holds raw data
    based_on : list, the list of channel names on which HEOG will be emulated upon

    Return
    ------
    raw : mne.io.Raw, the clean raw object

    '''
    
    ch1 = raw._data[raw.ch_names.index(based_on[0]), :]
    ch2 = raw._data[raw.ch_names.index(based_on[1]), :]
    HEOG = ch1-ch2

    info_heog = mne.create_info(['HEOG'], raw.info['sfreq'], ch_types='eog')
    info_heog['description'] = 'HEOG object'
    info_heog['custom_ref_applied'] = raw.info['custom_ref_applied']

    raw_heog = mne.io.RawArray(np.expand_dims(HEOG, axis=0), info_heog)
    raw = raw.add_channels([raw_heog], force_update_info=True)

    return raw

def get_markers_by_path(pth_raw):
    ''' Get the correct start and stop markers within the .vmrk file based 
    on filename
    Parameters
    ----------
    pth_raw : str, the path to the filename
    
    Return
    ------
    marker

    '''
    filename = ntpath.basename(pth_raw)
    for key, val in EXP_MARKERS.items():
        if key in filename:
            break
    marker = EXP_MARKERS[key]
    return marker


def crop_by_marker(raw, markers):
    ''' Crop a raw object based on start/stop markers
    Parameters
    ----------
    raw : mne.io.Raw, the mne data object
    markers : list, list of ints containing the markers 
        that indicate the start and stop of the segment of interest
    Return
    ------
    raw : mne.io.Raw, the cropped raw object
    '''
    start_marker, stop_marker = markers
    sfreq = raw.info['sfreq']
    events = mne.events_from_annotations(raw)[0]
    times = raw.times

    if stop_marker is not None:
        marker_index = np.where(events[:, 2]==stop_marker)[0]
        if marker_index.size == 0:
            # If there is no marker -> take last data point 
            stop_index = raw._data.shape[1] - 1
        else:
            stop_index = events[marker_index[0]][0]
    else:
        start_index = events[np.where(events[:, 2]==start_marker)[0][0]][0]
        stop_index = start_index + sfreq*EXP_DUR_SECONDS
    
    if start_marker is not None:
        marker_index = np.where(events[:, 2]==start_marker)[0]

        if marker_index.size == 0:
            # If there is no marker -> take first data point 
            start_index = 0
        else:
            start_index = events[marker_index[0]][0]

    else:
        # Start marker can be missing
        # In that case use the stop marker and go back 7 minutes
        start_index = stop_index - sfreq*EXP_DUR_SECONDS

    tmin = times[int(start_index)]
    tmax = times[int(stop_index)]

    return raw.crop(tmin=tmin, tmax=tmax, include_tmax=False)

def bad_channel_detection_interpolation(raw, n_resample=25, min_corr=0.6, n_jobs=-1):
    ''' Detect and interpolate bad channels in raw data 
    structure using Ransac.
    
    Parameters
    ----------
    raw : mne.io.Raw, mne Raw data object
    Return
    ------
    raw : mne.io.raw, the cleaned data object

    '''
    rsc = Ransac(n_jobs=n_jobs, verbose='tqdm_notebook', n_resample=n_resample, min_corr=min_corr)
    epochs = raw_to_epochs(raw)   # we have to transform Raw to Epochs bc. the bcr only works with Epochs (Raw is 2D (channels x time), Epochs is 3D (trials x channels x time))
    epochs_clean = rsc.fit_transform(epochs)
    raw._data = np.squeeze(epochs_clean.get_data())  # after cleaning up, we transform Epochs back into Raw by "removing" the empty dimension (here: trials. We only have one trial though. trials = 1. Kann man eigentlich weglassen, deshalb "empty")
    
    return raw