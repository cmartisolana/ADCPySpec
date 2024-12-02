import numpy as np
from scipy.signal import detrend as sci_detrend
from scipy.signal import windows as windows

class SpectrumProcessor:
    def __init__(self, x, y1=None, y2=None, win=None, pad=True, ax=0):
        """
        Initialize the SpectrumProcessor with input signals.

        Parameters:
        x (array-like): Independent variable, typically time.
        y1 (array-like): First input signal (optional).
        y2 (array-like): Second input signal (optional, for cross-spectrum).
        win (str): Type of window to use. Default is None.
        pad (bool): Whether to zero pad the signals. Default is True.
        ax (int): Axis along which to calculate. Default is 0.
        """
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.win = win
        self.pad = pad
        self.ax = ax

    def zero_pad(self, y):
        """
        Pads the beginning and end of y along the specified axis with zeros.

        Parameters:
        y (array-like): Input signal to be padded.

        Returns:
        array-like: Zero-padded signal.
        """
        # Determine the length of the input along the specified axis
        N = y.shape[self.ax]
        # Calculate the next power of 2 for efficient FFT computation
        N2 = 2**(np.ceil(np.log2(N)))
        # Calculate the amount of padding needed on each side
        Npad = np.ceil(0.5 * (N2 - N))
        # Recalculate padding if the initial estimate exceeds the next power of 2
        if 2 * Npad + N > N2:
            N2 = 2**(np.ceil(np.log2(N + 2 * Npad)))
            Npad = np.ceil(0.5 * (N2 - N))
        # Create padding arrays based on the dimensions of the input
        if self.ax == 0 and y.ndim == 1:
            pads = np.zeros((int(Npad),))
        elif self.ax == 0 and y.ndim >= 2:
            pads = np.zeros((int(Npad),) + y.shape[1:])
        elif self.ax == 1 and y.ndim == 2:
            pads = np.zeros((len(y), int(Npad)))
        elif self.ax == 1 and y.ndim >= 3:
            pads = np.zeros((len(y), int(Npad)) + y.shape[2:])
        elif self.ax == 2 and y.ndim == 3:
            pads = np.zeros((len(y), y.shape[1], int(Npad)))
        elif self.ax == 2 and y.ndim == 4:
            pads = np.zeros((len(y), y.shape[1], int(Npad)) + y.shape[3:])
        else:
            raise ValueError("Too many dimensions to pad or wrong axis choice.")

        # Concatenate the padding arrays to the beginning and end of the input
        yn = np.concatenate((pads, y, pads), axis=self.ax)
        return yn

    def compute_cross_spectrum(self):
        """
        Compute the cross-spectrum between two signals.

        Returns:
        tuple: Frequencies, inner and outer cross-spectra, and other spectral properties.
        """
        if self.y1 is None or self.y2 is None:
            raise ValueError("Both y1 and y2 must be provided for cross-spectrum calculation.")

        # Zero pad the signals if specified
        if self.pad:
            self.y1 = self.zero_pad(self.y1)
            self.y2 = self.zero_pad(self.y2)

        # Calculate the sampling interval (assumes uniform spacing)
        d = np.diff(self.x, axis=self.ax).mean()
        # Determine the length of the input along the specified axis
        N = self.y1.shape[self.ax]

        # Validate the window choice
        if self.win not in ['boxcar', 'hanning', 'hamming', 'bartlett', 'blackman', 'triang', None]:
            raise ValueError("Window choice is invalid")

        # Apply the window to the signals if specified
        if self.win is not None:
            # Get the window function from scipy.signal.windows
            win = getattr(windows, self.win)(N, sym=False)
            # Calculate the degree of freedom weight
            dofw = len(win) / np.sum(win**2)
            # Reshape the window to match the dimensions of the input signals
            win = win.reshape((N,) + (1,) * (self.y1.ndim - 1))
            # Roll the axis if necessary to match the input axis
            if self.ax != 0 and self.ax != -1:
                win = np.rollaxis(win, 0, start=self.ax + 1)
            elif self.ax != 0 and self.ax == -1:
                win = np.rollaxis(win, 0, start=self.y1.ndim)
            # Apply the window to both input signals
            self.y1 *= win
            self.y2 *= win
        else:
            dofw = 1.0

        # Roll the axis of the input signals if necessary
        if self.ax != 0:
            self.y1 = np.rollaxis(self.y1, self.ax, start=0)
            self.y2 = np.rollaxis(self.y2, self.ax, start=0)

        # Compute the FFT of both signals
        fy1, fy2 = map(np.fft.fft, (self.y1, self.y2), (None, None), (0, 0))
        # Apply FFT shift and scale by the degree of freedom weight
        fy1, fy2 = map(np.fft.fftshift, (np.sqrt(dofw) * fy1, np.sqrt(dofw) * fy2))
        # Compute the frequencies corresponding to the FFT output
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d))

        # Calculate the power spectral density of each signal
        ipy1 = (d / N) * np.abs(fy1)**2
        ipy2 = (d / N) * np.abs(fy2)**2
        # Calculate the cross-spectrum between the two signals
        ipy1y2 = (d / N) * (fy1.conj() * fy2)
        # Calculate the phase difference between the two signals
        iphase_y1y2 = np.arctan2(-ipy1y2.imag, ipy1y2.real)

        # Roll the axis back to the original position if necessary
        if self.ax != 0:
            ipy1y2 = np.rollaxis(ipy1y2, 0, start=self.ax)
            iphase_y1y2 = np.rollaxis(iphase_y1y2, 0, start=self.ax)

        # Return the computed frequencies, cross-spectrum, and other spectral properties
        return freqs, ipy1y2, ipy1, ipy2, iphase_y1y2, dofw

    def calc_ispec(self, k, l, E, ndim=2):
        """
        Calculates the azimuthally-averaged spectrum.

        Parameters:
        k (array-like): Wavenumber in the x-direction.
        l (array-like): Wavenumber in the y-direction.
        E (array-like): Two-dimensional spectrum.
        ndim (int): Number of dimensions. Default is 2.

        Returns:
        tuple: Radial wavenumber and azimuthally-averaged spectrum.
        """
        # Calculate the wavenumber spacing in both directions
        dk = np.abs(k[2] - k[1])
        dl = np.abs(l[2] - l[1])

        # Create a meshgrid of wavenumber values
        k, l = np.meshgrid(k, l)
        # Compute the magnitude of the wavenumber
        wv = np.sqrt(k**2 + l**2)

        # Determine the maximum wavenumber value to consider
        if k.max() > l.max():
            kmax = l.max()
        else:
            kmax = k.max()

        # Calculate the radial wavenumber bins
        dkr = np.sqrt(dk**2 + dl**2)
        kr = np.arange(dkr / 2., kmax + dkr / 2., dkr)
        Er = np.zeros((kr.size,))

        # Loop through each radial wavenumber bin and calculate the average energy
        for i in range(kr.size):
            fkr = (wv >= kr[i] - dkr / 2) & (wv <= kr[i] + dkr / 2)
            dth = np.pi / (fkr.sum() - 1)  # Angular spacing
            if ndim == 2:
                Er[i] = (E[fkr] * (wv[fkr] * dth)).sum()  # Sum energy in each bin
            elif ndim == 3:
                Er[i] = (E[fkr] * (wv[fkr] * dth)).sum(axis=(0, 1))

        # Return the radial wavenumber and the azimuthally-averaged spectrum
        return kr, Er.squeeze()

    def spectral_slope(self, k, E, kmin, kmax, stdE):
        """
        Compute spectral slope in log space in a wavenumber subrange [kmin, kmax].

        Parameters:
        k (array-like): Wavenumber values.
        E (array-like): Spectrum values.
        kmin (float): Minimum wavenumber for slope calculation.
        kmax (float): Maximum wavenumber for slope calculation.
        stdE (float): Standard error of the spectrum.

        Returns:
        tuple: Spectral slope and uncertainty.
        """
        # Find the indices of the wavenumber range for slope calculation
        fr = np.where((k >= kmin) & (k <= kmax))

        # Convert the wavenumber and spectrum to log space
        ki = np.matrix((np.log10(k[fr]))).T
        Ei = np.matrix(np.log10(np.real(E[fr]))).T
        # Create the diagonal error matrix
        dd = np.matrix(np.eye(ki.size) * ((np.abs(np.log10(stdE)))**2))

        # Set up the linear system to solve for the slope
        G = np.matrix(np.append(np.ones((ki.size, 1)), ki, axis=1))
        Gg = ((G.T * G).I) * G.T  # Pseudo-inverse
        m = Gg * Ei  # Solve for slope and intercept
        mm = np.sqrt(np.array(Gg * dd * Gg.T)[1, 1])  # Calculate uncertainty
        m = np.array(m)[1]  # Extract the slope value

        # Return the spectral slope and its uncertainty
        return m, mm
