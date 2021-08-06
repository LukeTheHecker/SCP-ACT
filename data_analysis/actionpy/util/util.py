import numpy as np
from copy import deepcopy
from matplotlib.backends.backend_pdf import PdfPages
import os
import matplotlib.pyplot as plt


filtparams = dict(method='iir', picks=['eeg', 'resp'])


def multipage(filename, figs=None, dpi=300, png=False):
    ''' Saves all open (or list of) figures to filename.pdf with dpi''' 
    pp = PdfPages(filename)
    path = os.path.dirname(filename)
    fn = os.path.basename(filename)[:-4]

    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        print(f'saving fig {fig}\n')
        fig.savefig(pp, format='pdf', dpi=dpi)
        if png:
            fig.savefig(f'{path}\\{i}_{fn}.png', dpi=600)
    pp.close()

def interp_nans(y):
    """ Interpolate nans in a 2d signal along axis 1.

    Parameters:
    -----------
    y : numpy.ndarray, 2D signal

    Return:
    -------
    y_new : numpy.ndarray, 2D signal without nans
    """
    if type(y) == list:
        y = np.array(y)

    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=0)

    y_new = deepcopy(y)
    for i in range(y.shape[0]):
        nans, x = nan_helper(y[i])
        y_new[i, nans] = np.interp(x(nans), x(~nans), y[i, ~nans])
    return y_new

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]