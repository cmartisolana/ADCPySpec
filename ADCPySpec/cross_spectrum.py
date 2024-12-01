import numpy as np
from scipy.signal import detrend as sci_detrend
from scipy.signal import windows as windows
#import kPyWavelet as wavelet
from scipy.interpolate import griddata as griddata
from scipy import mgrid as mgrid

def zero_pad(y, ax = 0):
    """
    pads the beguining and end of y along the axis ax with zeros
    """

    N = y.shape[ax]
    N2 = 2**(np.ceil(np.log2(N)))
    Npad = np.ceil(.5 * (N2 - N))
    if 2*Npad + N > N2:
        N2 = 2**(np.ceil(np.log2(N+2*Npad)))
        Npad = np.ceil(.5 * (N2 - N))
    if ax == 0 and y.ndim == 1:
        pads = np.zeros((Npad,))
    elif ax == 0 and y.ndim >= 2:
        pads = np.zeros((Npad,) + y.shape[1:])
    elif ax == 1 and y.ndim ==2:
        pads = np.zeros((len(y), Npad))
    elif ax == 1 and y.ndim >=3:
        pads = np.zeros((len(y), Npad) + y.shape[2:])
    elif ax == 2 and y.ndim ==3:
        pads = np.zeros((len(y), y.shape[1], Npad))
    elif ax == 2 and y.ndim == 4:
        pads = np.zeros((len(y), y.shape[1], Npad) + y.shape[3:])
    else:
        raise ValueError("Too many dimensions to pad or wrong axis choice.")
    
    yn = np.concatenate((pads, y, pads), axis = ax)
    return yn

def cross_spec(x, y1, y2, win = None, pad = True, ax = 0):
    """
    Cross spectrum, non-rotary 
    """
    if pad == True:
        y1 = zero_pad(y1, ax = ax)
        y2 = zero_pad(y2, ax = ax)
    
    d = np.diff(x, axis = ax).mean()
    N = y1.shape[ax]
    if not win in ['boxcar', 'hanning', 'hamming', 'bartlett', 'blackman', 'triang', None]:
        raise ValueError("Window choice is invalid")
    
    if win != None:
        win = eval('windows.' + win + '(N' + ', sym = False)')
        dofw = len(win) / np.sum(win**2)
        win.resize((N,) + tuple(np.int8(np.ones(y1.ndim - 1))))
        if ax != 0 and ax != -1:
            win = np.rollaxis(win, 0, start = ax + 1)
        elif ax != 0 and ax == -1:
            win = np.rollaxis(win, 0, start = y1.dim)
        elif ax == 0:
            win = win
        else:
            raise ValueError("Pick your axis better.")
        
        y1 = sci_detrend(y1 * win, axis = ax)
        y2 = sci_detrend(y2 * win, axis = ax)
    else:
        dofw = 1.0
    
    fy1, fy2 = map(np.fft.fft, (y1, y2), (None, None), (ax, ax))
    fy1, fy2 = map(np.fft.fftshift, (np.sqrt(dofw)*fy1, np.sqrt(dofw)*fy2), (ax, ax))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d))

    py1 = (d/N)*np.abs(fy1)**2
    py2 = (d/N)*np.abs(fy2)**2
    py1y2 = (d/N)*(fy1.conj() * fy2) # cross spectrum
    cy1y2 = (d/N)*( fy1.real*fy2.real + fy1.imag*fy2.imag ) # coincident spectrum
    qy1y2 = (d/N)*( fy1.real*fy2.imag - fy2.real*fy1.imag ) # quadrature spectrum

    ay1y2 = np.sqrt( cy1y2**2 + qy1y2**2 ) # cross amplitude
    phase = np.arctan2(-1.*qy1y2, cy1y2)

    return freq, py1y2, cy1y2, qy1y2, ay1y2, py1, py2, phase, dofw 


def rot_cross_spec(x, y1, y2, win = None, pad = True, ax = 0):
    """
    Rotary Cross spectrum: apply to dim = ax by transposing, ffting dim=0
    and then tranposing back
    """
    if pad == True:
        y1 = zero_pad(y1, ax = ax)
        y2 = zero_pad(y2, ax = ax)
    
    d = np.diff(x, axis = ax).mean()
    N = y1.shape[ax]
    if not win in ['boxcar', 'hanning', 'hamming', 'bartlett', 'blackman', 'triang', None]:
        raise ValueError("Window choice is invalid")
    
    if win != None:
        win = eval('windows.' + win + '(N)')
        dofw = len(win) / np.sum(win**2)
        win.resize((N,) + tuple(np.int8(np.ones(y1.ndim - 1))))
        if ax != 0 and ax != -1:
            win = np.rollaxis(win, 0, start = ax + 1)
        elif ax != 0 and ax == -1:
            win = np.rollaxis(win, 0, start = y1.dim)
        elif ax == 0:
            win = win
        else:
            raise ValueError("Pick your axis better.")
        
        y1 = y1 * win
        y2 = y2 * win
    else:
        dofw = 1.0
    
    if ax != 0:
        y1 = np.rollaxis(y1, ax, start = 0)
        y2 = np.rollaxis(y2, ax, start = 0)
    
    fy1, fy2 = map(np.fft.fft, (y1, y2), (None, None), (0, 0))
    fy1, fy2 = map(np.fft.fftshift, (np.sqrt(dofw)*fy1, np.sqrt(dofw)*fy2))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d))

    ipy1 = (d/N)*np.abs(fy1)**2
    ipy2 = (d/N)*np.abs(fy2)**2
    ipy1y2 = (d/N)*(fy1.conj() * fy2) # inner cross spectrum
    iphase_y1y2 = np.arctan2(-1.*ipy1y2.imag, ipy1y2.real)
    
    opy1 = ipy1 * np.flipud( ipy1 ) # outer autospectrum (not a "real" spec: complex)
    opy2 = ipy2 * np.flipud( ipy2 )
    opy1y2 = ipy1 * np.flipud( ipy2 )
    opy2y1 = ipy2 * np.flipud( ipy1 )
    ophase_y1y2 = np.arctan2( -1.* np.concatenate(np.flipud(opy2y1[freqs>0].imag), opy1y2[freqs>=0].imag, axis = 0), np.concatenate( np.flipud(opy2y1[freqs>0].real), opy1y2[freqs>=0].real, axis = 0) )
    
    if ax != 0:
        ipy1y2 = np.rollaxis(ipy1y2, 0, start = ax)
        iphase_y1y2 = np.rollaxis(iphase_y1y2, 0, start = ax)
        opy1y2 = np.rollaxis(opy1y2, 0, start = ax)
        ophase_y1y2 = np.rollaxis(ophase_y1y2, 0, start = ax)
    
    return freqs, ipy1y2, ipy1, ipy2, iphase_y1y2, opy1y2, opy2y1, opy1, opy2, ophase_y1y2, dofw 
