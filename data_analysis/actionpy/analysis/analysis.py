import string
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert, detrend, welch, find_peaks
from scipy.stats import pearsonr


def omega_descriptor(signal, nan_handling='return_nan'):
    ''' Calculates the omega descriptor from a 2D signal.

    Parameters:
    -----------
    signal : numpy.ndarray, 2D signal (channels x time)
    nan_handling : str, policy on handling nans in the signal. 
      'return_nan' : if theres a nan in the signal -> return nan
      'interp_nan' : if theres a  nan in the signal -> interpolate and then perform calculation

    Return:
    -------
    omega : float, the omega (complexity)
    '''
    # Nan handling:
    if np.any(np.isnan(signal)):
        if nan_handling == 'interp_nan':
            signal = interp_nans(signal)
        else:
            return np.nan

    # Ensure its a numpy ndarray
    signal = np.array(signal)
    K, N = signal.shape

    # Zeroing
    signal = np.array([sig-np.mean(sig) for sig in signal])
    # Common average referencing
    signal -= np.mean(signal, axis=0)
    cov = np.cov(signal)
    eigs = np.linalg.eigvals(cov)
    eigs_filtered = eigs[np.where(eigs>0)[0]]
    eigs_filtered /= np.sum(eigs_filtered)
    return np.exp( -np.sum(eigs_filtered*np.log(eigs_filtered)) )

def sigma_descriptor(signal, nan_handling='return_nan'):
    ''' Calculates the sigma descriptor from a 2D signal.

    Parameters:
    -----------
    signal : numpy.ndarray, 2D signal (channels x time)
    nan_handling : str, policy on handling nans in the signal. 
      'return_nan' : if theres a nan in the signal -> return nan
      'interp_nan' : if theres a  nan in the signal -> interpolate and then perform calculation

    Return:
    -------
    sigma : float, the sigma
    '''
    # Nan handling:
    if np.any(np.isnan(signal)):
        if nan_handling == 'interp_nan':
            signal = interp_nans(signal)
        else:
            return np.nan
    # Globa field power

    # Ensure its a numpy ndarray
    signal = np.array(signal)
    K, N = signal.shape
    # Zeroing
    signal = np.stack([sig-np.mean(sig) for sig in signal], axis=0)
    # Common average referencing
    signal -= np.mean(signal, axis=0)
    # Calc sigma
    sigma = np.mean(np.std(signal, axis=0)**2)
    sigma = np.sqrt(np.mean(signal**2))
    return sigma

def phi_descriptor(signal,sfreq, nan_handling='return_nan'):
    ''' Calculates the phi descriptor from a 2D signal.

    Parameters:
    -----------
    signal : numpy.ndarray, 2D signal (channels x time)
    sfreq : int, sampling frequency of the signal
    nan_handling : str, policy on handling nans in the signal. 
      'return_nan' : if theres a nan in the signal -> return nan
      'interp_nan' : if theres a  nan in the signal -> interpolate and then perform calculation

    Return:
    -------
    phi : float, the phi
    '''
    # Nan handling:
    if np.any(np.isnan(signal)):
        if nan_handling == 'interp_nan':
            signal = interp_nans(signal)
        else:
            return np.nan
    # Ensure its a numpy ndarray
    signal = np.array(signal)
    K, N = signal.shape
    # Zeroing
    signal = np.stack([sig-np.mean(sig) for sig in signal], axis=0)
    # Common average referencing
    signal -= np.mean(signal, axis=0)
    # Sum of squared signals per electrode, averaged together
    m_zero = np.mean(signal**2)
    m_one =  np.mean(np.diff(signal, axis=1)**2)
    phi = np.sqrt(m_one/m_zero) * (sfreq/(2*np.pi))

    return phi

def zscore(x):
    return (x - np.nanmean(x)) / np.nanstd(x)

def phase_synchrony(y1, y2):
    ''' Calculate phase synchrony between two signals.
    Parameters:
    -----------
    y1 : float, bandpass-filtered time series data
    y2 : float, bandpass-filtered time series data of same length as y1

    Return:
    -------
    mean_angle : float, phase synchrony index
    '''

    # Get Phase for each signal
    phase1 = np.angle(hilbert(y1))
    phase2 = np.angle(hilbert(y2))
    # Get phase difference
    phasediff = phase1 - phase2
    # Calculate length of average vector

    mean_angle = 0
    for i, th in enumerate(phasediff):
        mean_angle += np.e**(1j*th)

    mean_angle = np.abs(mean_angle / len(phasediff))

    return mean_angle

def average_phase_synchrony(electrodeslist, nan_handling='return_nan'):
    ''' Calculate the average phase synchrony of all electrode pairs.
    Input:
    -----------
    electrodeslist: List of data arrays for each electrode. 
    Structure of array like parameters in phase_synchrony. 
    nan_handling : str, policy on handling nans in the signal. 
        'return_nan' : if theres a nan in the signal -> return nan
        'interp_nan' : if theres a  nan in the signal -> interpolate and then perform calculation
        
        Output:
        --------
        avg_ps: Float, Average phase synchrony index of all pairs of electrodes
    '''
    # Nan handling:
    if np.any(np.isnan(electrodeslist)):
        if nan_handling == 'interp_nan':
            electrodeslist = interp_nans(electrodeslist)
        else:
            return np.nan
    #empty dictionary where key is index of electrode list of each pair
    phasesync_dict = {}

    #get all possible pairs of electrodes
    perm = combinations(range(len(electrodeslist)), 2) 

    #loop through all pairs
    for i, j in list(perm):
        #key of dictionary is index of electrode in list
        key = f'{i}, {j}'
        #calculate the phase synchrony for pair
        phasesync_dict[key] = phase_synchrony(electrodeslist[i], electrodeslist[j])

    #put all phase synchrony values into list
    values = [val for key, val in phasesync_dict.items()] 
    #get average phase synchrony 
    avg_ps = np.mean(values)
    
    return avg_ps

def freq_band_power(data, sfreq, freqband, nan_handling='return_nan'):
    # Nan handling:
    if np.any(np.isnan(data)):
        if nan_handling == 'interp_nan':
            data = interp_nans(data)
        else:
            return np.nan

    freqs, psd = welch(data, sfreq, nperseg=len(data))

    idx_lower, idx_upper = [np.argmin(np.abs(freqs-freq)) for freq in freqband]
    psd_band = np.mean(psd[idx_lower:idx_upper])
    return psd_band

def run_length_encoding(x, gridsize=1, plotme=False, nan_handling='interp_nan', detrend_signal=True):
    ''' Run length encoding for time series data.

    Parameters:
    -----------
    x : list/ndarray, 1D time series data
    gridsize : int,  number of discretizations of the data. The lower the gridsize,
        the higher the runlength will be.
    nan_handling : str, policy on handling nans in the signal. 
        'return_nan' : if theres a nan in the signal -> return nan
        'interp_nan' : if theres a  nan in the signal -> interpolate and then perform calculation
    
    Return:
    -------
    avg_run_len : float, the mean run length
    '''
    # Tests
    assert type(x)==list or type(x)==np.ndarray, 'X is wrong data type. It should be list or numpy ndarray'
    assert type(gridsize)==int, 'gridsize must be of type integer'
    assert type(plotme)==bool, 'plotme can be either True or False.'

    # Nan handling:
    if np.any(np.isnan(x)):
        if nan_handling == 'interp_nan':
            x = interp_nans(x)
        else:
            return np.nan

    abc = string.ascii_lowercase
    if detrend_signal:
        x = detrend(x)

    # Calculate a discretized time series
    discretized = discretize_time_series(x, gridsize)
    # Assign labels to each discrete step
    labels = list(abc[int(np.min(discretized))-1:int(np.max(discretized))])
    # Encode run length
    
    # encoding = ''
    list_of_runs = []
    idx = 0
    while True:
        state = discretized[idx]
        runlen, idx = start_run(discretized, idx)
        list_of_runs.append(runlen)
        # encoding += (labels[int(state-1)] + str(runlen))
        if idx == len(discretized):
            break
    # Plot
    if plotme:
        plt.figure()
        plt.plot(discretized)

    return np.mean(list_of_runs)  #, discretized, encoding 

def autocorrelation(x, corrfun=pearsonr):
    ''' Calculate the autocorrelation of a time series x.
    Paramters:
    ----------
    x : list/ndarray, 1D vector of the time series
    Return:
    -------
    ac : autocorrelation signal
    '''
    assert type(x)==list or type(x)==np.ndarray, 'Input x must be list or numpy array'

    if np.mod(len(x), 2) != 0:
        # if not divisible by 2
        x = np.append(x, x[-1])
        extra = 1
    else:
        extra = 0
    
    # loop through each lag
    ac = []
    mid_idx = (len(x)) / 2
    lags = np.array(np.arange(len(x)) - mid_idx, dtype=int)

    for lag in lags:
        if lag > 0:
            x_tmp = x[lag:]
            x_lag = x[0:-lag]
        elif lag < 0:
            x_tmp = x[0:lag-1]
            x_lag = x[-lag+1:]
        else:
            x_tmp = x
            x_lag = x
        ac.append(corrfun(x_tmp, x_lag)[0])
    
    return np.array(ac)

def start_run(x, idx):
    ''' Loops through time series x starting from given idx as long as the values are the same.
    Returns the length of the run and the index of the first element unequal to x[idx]

    Paramters:
    ----------
    x :  list/ndarray, 1D time series vector
    idx : int, starting index 

    Return:
    -------
    runlen : int, length of the run (== number of iterations in loop)
    stop_idx : index of first element where x[idx] != x[stop_idx]
    '''
    # Tests
    assert idx <= len(x), 'Index is higher than time series x. '
    # initialize runlen
    runlen = 0

    for i in range(idx, len(x)):
        if x[i] == x[idx]:
            runlen += 1
        else:
            stop_idx = i
            return runlen, stop_idx

    stop_idx = i+1
    return runlen, stop_idx

def discretize_time_series(x, gridsize):
    ''' Transform a time series x into discretized values.

    Parameters:
    -----------
    x : list/ndarray, 1D time series vector
    gridsize : number of discrete steps

    Return:
    -------
    discretized : discretized time series
    '''
    # Tests
    assert type(x)==list or type(x)==np.ndarray, 'X is wrong data type. It should be list or numpy ndarray'
    assert type(gridsize)==int, 'gridsize must be of type integer'

    # lower = np.min(x)
    # upper = np.max(x)
    lower = np.percentile(x, 10)
    upper = np.percentile(x, 90)

    span  = upper - lower
    section_height = span/gridsize
    valranges = []
    for i in range(gridsize):
        valranges.append([i*(section_height) + lower, lower + (section_height) * (i+1)])

    discretized = np.zeros((len(x)))
    for i, val in enumerate(x):
        for grid_idx, valrange in enumerate(valranges):
            if val >= valrange[0] and val <= valrange[1]:
                discretized[i] = grid_idx+1
                break
    return discretized

def full_width_half_maximum(x):
    ''' Full width at half maximum calculation.
    Paramters:
    ----------
    x : list/ndarray, time series data, probability distribution
        or anything where fwhm makes sense
    Return:
    -------
    width : float, 
    '''
    half_maximum = np.max(x) / 2
    # Subtract halt maximum so we can search for zero crossings
    x -= half_maximum
    mid_idx = int(round(len(x) / 2))

    # plt.figure()
    # plt.plot(x)

    zero_crossing = np.where(np.diff(np.sign(x)))[0] 
    zero_crossing -= mid_idx  
    if len(zero_crossing) > 2:
        lower_bound = np.abs(np.min(np.abs(zero_crossing[np.where(zero_crossing<0)[0]])))
        upper_bound = np.abs(np.min(np.abs(zero_crossing[np.where(zero_crossing>0)[0]])))
    elif len(zero_crossing) == 1:
        lower_bound = upper_bound = np.abs(zero_crossing)
    elif len(zero_crossing) == 2:
        lower_bound, upper_bound = np.abs(zero_crossing)
    else:
        lower_bound = upper_bound = np.nan
    return lower_bound + upper_bound

def get_acfw(x, nan_handling='interp_nan', detrend_signal=True):
    """ Calculate the Autocorrelation function width
    Parameters:
    -----------
    x : numpy.ndarray, signal
    nan_handling : str, policy on handling nans in the signal. 
        'return_nan' : if theres a nan in the signal -> return nan
        'interp_nan' : if theres a  nan in the signal -> interpolate and then perform calculation
    
    Return:
    -------
    acfw : float, 
    """
    # Nan handling:
    if np.any(np.isnan(x)):
        if nan_handling == 'interp_nan':
            x = interp_nans(x)
        else:
            return np.nan
    if detrend_signal:
        x = detrend(x)
    ac = autocorrelation(x)
    
    fwhm = full_width_half_maximum(ac)
    return fwhm


def get_ptp_amplitudes(x, distance=None, sfreq=100, plot=False):
    ''' Calculate peak-to-peak amplitudes in a signal.
    Parameters
    ----------
    x : numpy.ndarray, 1D time series (e.g. SCP)
    distance : float/int, specifies the minimum distance between two consecutive peaks
    sfreq : int, sampling frequency of the signal x
    
    Return
    ------
    peak_to_peak_amplitudes : numpy.ndarray containing the p2p amplitudes

    '''
    return peak_to_peak_helper(x, distance=distance, sfreq=sfreq)[0]


def get_ptp_latencies(x, distance=None, sfreq=100):
    ''' Calculate peak-to-peak amplitudes in a signal.
    Parameters
    ----------
    x : numpy.ndarray, 1D time series (e.g. SCP)
    distance : float/int, specifies the minimum distance between two consecutive peaks
    sfreq : int, sampling frequency of the signal x
    
    Return
    ------
    peak_to_peak_amplitudes : numpy.ndarray containing the p2p amplitudes
    
    '''

    return peak_to_peak_helper(x, distance=distance, sfreq=sfreq)[1]

def peak_to_peak_helper(x, distance=None, sfreq=100):
    ''' Calculate peak-to-peak amplitudes in a signal.
    Parameters
    ----------
    x : numpy.ndarray, 1D time series (e.g. SCP)
    distance : float/int, specifies the minimum distance between two consecutive peaks
    sfreq : int, sampling frequency of the signal x
    
    Return
    ------
    peak_to_peak_amplitudes : numpy.ndarray containing the p2p amplitudes
    
    '''
    peak_to_peak_latencies = []
    peak_to_peak_amplitudes = []
    
    if distance is not None:
        distance = sfreq*distance

    pos_peak_idc = find_peaks(x, distance=distance)[0]
    neg_peak_idc = find_peaks(-1*x, distance=distance)[0]

    for i, pos_peak_idx in enumerate(pos_peak_idc):
        
        next_negative_idc = np.where(neg_peak_idc>pos_peak_idx)[0]
        
        if next_negative_idc.size==0:
            break
        else:
            next_negative_idx = next_negative_idc[0]
        
        neg_peak_idx = neg_peak_idc[next_negative_idx]
        
        if i!= len(pos_peak_idc)-1:
            if pos_peak_idc[i+1] < neg_peak_idx:
                continue

        idx_pair = (pos_peak_idx, neg_peak_idx)
        diff = np.abs(idx_pair[0]-idx_pair[1])
        
        peak_to_peak_latencies.append( diff / sfreq )
        amp_diff = np.abs(x[pos_peak_idx] - x[neg_peak_idx])
        peak_to_peak_amplitudes.append( amp_diff )

    peak_to_peak_latencies = np.array(peak_to_peak_latencies)
    peak_to_peak_amplitudes = np.array(peak_to_peak_amplitudes)

    return peak_to_peak_amplitudes, peak_to_peak_latencies

def detrended_std(x):
    x = detrend(x)
    return np.std(x)