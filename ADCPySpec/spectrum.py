import numpy as np
from scipy.signal import detrend as sci_detrend
from scipy.signal import windows as windows
from scipy.special import gammainc

class SpectrumProcessor:
    def __init__(self, x, y1=None, y2=None, win=None, pad=True, ax=0):
        """
        Initialize the SpectrumProcessor with input signals.

        Parameters:
        x (array-like): Independent variable, typically time.
        dt (float): Sampling interval.
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
        if self.win not in ['boxcar', 'hann', 'hamming', 'bartlett', 'blackman', 'triang', None]:
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
        py1 = (d / N) * np.abs(fy1)**2
        py2 = (d / N) * np.abs(fy2)**2
        # Calculate the cross-spectrum between the two signals
        py1y2 = (d / N) * (fy1.conj() * fy2)

        cy1y2 = (d/N)*( fy1.real*fy2.real + fy1.imag*fy2.imag ) # coincident spectrum
        qy1y2 = (d/N)*( fy1.real*fy2.imag - fy2.real*fy1.imag ) # quadrature spectrum
        
        ay1y2 = np.sqrt( cy1y2**2 + qy1y2**2 ) # cross amplitude
        # Calculate the phase difference between the two signals
        phase_y1y2 = np.arctan2(-py1y2.imag, py1y2.real)

        # Roll the axis back to the original position if necessary
        if self.ax != 0:
            py1y2 = np.rollaxis(py1y2, 0, start=self.ax)
            phase_y1y2 = np.rollaxis(phase_y1y2, 0, start=self.ax)

        # Return the computed frequencies, cross-spectrum, and other spectral properties
        return freqs, py1y2, py1, py2, cy1y2, qy1y2, ay1y2, phase_y1y2, dofw
    
    def yNlu(sn,yN,ci):
        """ compute yN[l] yN[u], that is, the lower and
                    upper limit of yN """

        # cdf of chi^2 dist. with 2*sn DOF
        cdf = gammainc(sn,sn*yN)

        # indices that delimit the wedge of the conf. interval
        fl = np.abs(cdf - ci).argmin()
        fu = np.abs(cdf - 1. + ci).argmin()

        return yN[fl],yN[fu]

    def spec_error(E,sn,ci=.95):

        """ Computes confidence interval for one-dimensional spectral
            estimate E.

            Parameters
            ===========
            - sn is the number of spectral realizations;
                    it can be either an scalar or an array of size(E)
            - ci = .95 for 95 % confidence interval

            Output
            ==========
            lower (El) and upper (Eu) bounds on E """

        dbin = .005
        yN = np.arange(0,2.+dbin,dbin)

        El, Eu = np.empty_like(E), np.empty_like(E)

        try:
            n = sn.size
        except AttributeError:
            n = 0

        if n:
            assert n == E.size, " *** sn has different size than E "

            for i in range(n):
                yNl,yNu = yNlu(sn[i],yN=yN,ci=ci)
                El[i] = E[i]/yNl
                Eu[i] = E[i]/yNu

        else:
            yNl,yNu = yNlu(sn,yN=yN,ci=ci)
            El = E/yNl
            Eu = E/yNu

        return El, Eu

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