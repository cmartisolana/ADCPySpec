import numpy as np
from scipy.signal import detrend as sci_detrend
from scipy.signal import windows as windows
#import kPyWavelet as wavelet
from scipy.interpolate import griddata as griddata
from scipy import mgrid as mgrid

def zero_pad(y, ax=0):
    """
    Pads the beginning and end of the array `y` along the specified axis `ax` with zeros.
    The total length of the padded axis is adjusted to the next power of 2.
    
    Parameters:
    y : ndarray
        The input array to pad.
    ax : int, optional
        The axis along which to apply the zero-padding. Default is 0.
    
    Returns:
    yn : ndarray
        The zero-padded array with the specified axis length adjusted to a power of 2.
    """
    
    # Get the size of the specified axis
    N = y.shape[ax]
    
    # Calculate the next power of 2 greater than or equal to N
    N2 = int(2**np.ceil(np.log2(N)))
    
    # Calculate the amount of padding needed on each side
    Npad = int(np.ceil(0.5 * (N2 - N)))
    
    # Ensure the padding doesn't exceed the target power of 2 length
    if 2 * Npad + N > N2:
        N2 = int(2**np.ceil(np.log2(N + 2 * Npad)))
        Npad = int(np.ceil(0.5 * (N2 - N)))
    
    # Construct the padding array based on the axis and dimensions of `y`
    if ax == 0 and y.ndim == 1:  # Case for 1D array, padding along axis 0
        pads = np.zeros((Npad,))
    elif ax == 0 and y.ndim >= 2:  # Case for multi-dimensional array, padding along axis 0
        pads = np.zeros((Npad,) + y.shape[1:])
    elif ax == 1 and y.ndim == 2:  # Case for 2D array, padding along axis 1
        pads = np.zeros((y.shape[0], Npad))
    elif ax == 1 and y.ndim >= 3:  # Case for higher-dimensional arrays, padding along axis 1
        pads = np.zeros((y.shape[0], Npad) + y.shape[2:])
    elif ax == 2 and y.ndim == 3:  # Case for 3D array, padding along axis 2
        pads = np.zeros((y.shape[0], y.shape[1], Npad))
    elif ax == 2 and y.ndim == 4:  # Case for 4D array, padding along axis 2
        pads = np.zeros((y.shape[0], y.shape[1], Npad) + y.shape[3:])
    else:
        # Raise an error for invalid axis or unsupported dimensions
        raise ValueError("Too many dimensions to pad or wrong axis choice.")
    
    # Concatenate the padding and the original array along the specified axis
    yn = np.concatenate((pads, y, pads), axis=ax)
    
    return yn

def cross_spec(x, y1, y2, win=None, pad=True, ax=0):
    """
    Compute the non-rotary cross spectrum and related spectra between two signals.
    
    Parameters:
    x : ndarray
        The independent variable, typically time or distance.
    y1, y2 : ndarray
        The two signals to compute the cross spectrum for. Must have the same shape along `ax`.
    win : str or None, optional
        The window function to apply. Options: 'boxcar', 'hanning', 'hamming', 
        'bartlett', 'blackman', 'triang', or None (default: None).
    pad : bool, optional
        If True, zero-pad `y1` and `y2` along `ax` to the next power of 2. Default is True.
    ax : int, optional
        The axis along which the spectrum is computed. Default is 0.
    
    Returns:
    freq : ndarray
        Frequency array.
    py1y2 : ndarray
        Cross spectrum.
    cy1y2 : ndarray
        Coincident spectrum.
    qy1y2 : ndarray
        Quadrature spectrum.
    ay1y2 : ndarray
        Cross amplitude spectrum.
    py1, py2 : ndarray
        Power spectra of `y1` and `y2`.
    phase : ndarray
        Phase spectrum (in radians).
    dofw : float
        Degrees of freedom weighting for the applied window.
    """
    # Zero-pad the signals if requested
    if pad:
        y1 = zero_pad(y1, ax=ax)
        y2 = zero_pad(y2, ax=ax)
    
    # Ensure y1 and y2 have the same shape along the specified axis
    if y1.shape[ax] != y2.shape[ax]:
        raise ValueError("y1 and y2 must have the same size along the specified axis.")

    # Calculate sampling interval
    d = np.diff(x, axis=ax).mean()
    N = y1.shape[ax]  # Length of the specified axis

    # Validate the window choice
    valid_windows = ['boxcar', 'hanning', 'hamming', 'bartlett', 'blackman', 'triang', None]
    if win not in valid_windows:
        raise ValueError(f"Invalid window choice. Options are: {valid_windows}")

    # Apply window function if specified
    if win:
        win_func = getattr(windows, win)(N, sym=False)
        dofw = len(win_func) / np.sum(win_func**2)  # Degrees of freedom weighting

        # Reshape window to match the axis and dimensions of y1 and y2
        win_shape = [1] * y1.ndim
        win_shape[ax] = N
        win_func = win_func.reshape(win_shape)
        
        # Apply the window and detrend the signals
        y1 = sci_detrend(y1 * win_func, axis=ax)
        y2 = sci_detrend(y2 * win_func, axis=ax)
    else:
        dofw = 1.0

    # Perform FFT and FFT shift
    fy1 = np.fft.fft(y1, axis=ax)
    fy2 = np.fft.fft(y2, axis=ax)
    fy1 = np.fft.fftshift(np.sqrt(dofw) * fy1, axes=ax)
    fy2 = np.fft.fftshift(np.sqrt(dofw) * fy2, axes=ax)
    freq = np.fft.fftshift(np.fft.fftfreq(N, d))

    # Calculate power and cross spectra
    py1 = (d / N) * np.abs(fy1)**2  # Power spectrum of y1
    py2 = (d / N) * np.abs(fy2)**2  # Power spectrum of y2
    py1y2 = (d / N) * (fy1.conj() * fy2)  # Cross spectrum
    cy1y2 = (d / N) * (fy1.real * fy2.real + fy1.imag * fy2.imag)  # Coincident spectrum
    qy1y2 = (d / N) * (fy1.real * fy2.imag - fy1.imag * fy2.real)  # Quadrature spectrum

    # Calculate cross amplitude and phase
    ay1y2 = np.sqrt(cy1y2**2 + qy1y2**2)  # Cross amplitude
    phase = np.arctan2(-qy1y2, cy1y2)  # Phase spectrum

    return freq, py1y2, cy1y2, qy1y2, ay1y2, py1, py2, phase, dofw

import numpy as np
from scipy.signal import windows

def rot_cross_spec(x, y1, y2, win=None, pad=True, ax=0):
    """
    Compute the rotary cross spectrum and related spectra.
    
    Parameters:
    x : ndarray
        Independent variable (e.g., time or spatial coordinate).
    y1, y2 : ndarray
        Two signals to analyze. Must have the same shape along the specified axis.
    win : str or None, optional
        Window function to apply. Options: 'boxcar', 'hann', 'hamming', 'bartlett', 
        'blackman', 'triang', or None (default: None).
    pad : bool, optional
        If True, zero-pad `y1` and `y2` along the specified axis. Default is True.
    ax : int, optional
        Axis along which to compute the spectrum. Default is 0.
    
    Returns:
    freqs : ndarray
        Frequencies corresponding to the computed spectra.
    ipy1y2 : ndarray
        Inner cross spectrum.
    ipy1, ipy2 : ndarray
        Inner power spectra of `y1` and `y2`.
    iphase_y1y2 : ndarray
        Inner phase spectrum (in radians).
    opy1y2, opy2y1 : ndarray
        Outer cross spectra.
    opy1, opy2 : ndarray
        Outer power spectra of `y1` and `y2`.
    ophase_y1y2 : ndarray
        Outer phase spectrum (in radians).
    dofw : float
        Degrees of freedom weighting for the applied window.
    """
    # Zero-pad the signals if requested
    if pad:
        y1 = zero_pad(y1, ax=ax)
        y2 = zero_pad(y2, ax=ax)
    
    # Ensure `y1` and `y2` have the same shape along the specified axis
    if y1.shape[ax] != y2.shape[ax]:
        raise ValueError("y1 and y2 must have the same size along the specified axis.")

    # Calculate sampling interval and number of points
    d = np.diff(x, axis=ax).mean()
    N = y1.shape[ax]

    # Validate and apply the window function
    valid_windows = ['boxcar', 'hann', 'hamming', 'bartlett', 'blackman', 'triang', None]
    if win not in valid_windows:
        raise ValueError(f"Invalid window choice. Options are: {valid_windows}")
    
    if win:
        win_func = getattr(windows, win)(N)
        dofw = len(win_func) / np.sum(win_func**2)

        # Reshape the window to match the axis
        win_shape = [1] * y1.ndim
        win_shape[ax] = N
        win_func = win_func.reshape(win_shape)

        # Apply the window
        y1 = y1 * win_func
        y2 = y2 * win_func
    else:
        dofw = 1.0

    # Transpose the specified axis to the first position
    if ax != 0:
        y1 = np.moveaxis(y1, ax, 0)
        y2 = np.moveaxis(y2, ax, 0)

    # Perform FFT and FFT shift
    fy1 = np.fft.fft(y1, axis=0)
    fy2 = np.fft.fft(y2, axis=0)
    fy1 = np.fft.fftshift(np.sqrt(dofw) * fy1, axes=0)
    fy2 = np.fft.fftshift(np.sqrt(dofw) * fy2, axes=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d))

    # Compute inner power spectra and cross spectrum
    ipy1 = (d / N) * np.abs(fy1)**2  # Inner power spectrum of y1
    ipy2 = (d / N) * np.abs(fy2)**2  # Inner power spectrum of y2
    ipy1y2 = (d / N) * (fy1.conj() * fy2)  # Inner cross spectrum
    iphase_y1y2 = np.arctan2(-ipy1y2.imag, ipy1y2.real)  # Inner phase spectrum

    # Compute outer power spectra and cross spectra
    ipy1_flip = np.flip(ipy1, axis=0)
    ipy2_flip = np.flip(ipy2, axis=0)

    opy1 = ipy1 * ipy1_flip  # Outer auto-spectrum of y1
    opy2 = ipy2 * ipy2_flip  # Outer auto-spectrum of y2
    opy1y2 = ipy1 * ipy2_flip  # Outer cross spectrum
    opy2y1 = ipy2 * ipy1_flip  # Outer cross spectrum (reversed)
    
    # Compute outer phase spectrum
    ophase_y1y2 = np.arctan2(
        -np.concatenate([ipy2_flip.imag[freqs > 0], ipy1y2.imag[freqs >= 0]]),
        np.concatenate([ipy2_flip.real[freqs > 0], ipy1y2.real[freqs >= 0]])
    )

    # Transpose the results back to the original axis
    if ax != 0:
        ipy1y2 = np.moveaxis(ipy1y2, 0, ax)
        iphase_y1y2 = np.moveaxis(iphase_y1y2, 0, ax)
        opy1y2 = np.moveaxis(opy1y2, 0, ax)
        ophase_y1y2 = np.moveaxis(ophase_y1y2, 0, ax)

    return freqs, ipy1y2, ipy1, ipy2, iphase_y1y2, opy1y2, opy2y1, opy1, opy2, ophase_y1y2, dofw