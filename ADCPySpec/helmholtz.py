import numpy as np
from numpy import pi, sinh, cosh
from scipy import integrate

# Attempt to use Intel MKL for faster numpy operations if available
try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass

class HelmholtzDecomposition:
    """
    A class to compute the Helmholtz decomposition of KE spectra into rotational
    and divergent components based on Buhler et al. (JFM 2014, JFM2015).
    """

    def __init__(self, k, Cu, Cv, Cuv, u=None, v=None, theta=None):
        """
        Initialize the decomposition class.

        Parameters:
        k : array-like
            Wavenumber array.
        Cu : array-like
            Spectrum of across-track velocity.
        Cv : array-like
            Spectrum of along-track velocity.
        Cuv : array-like
            Cross-spectrum of velocities.
        u, v : array-like, optional
            Velocity components (for model3_decomposition).
        theta : float, optional
            Angle parameter (for model3_decomposition).
        """
        self.k = k
        self.Cu = Cu
        self.Cv = Cv
        self.Cuv = Cuv
        self.u = u
        self.v = v
        self.theta = theta

    @staticmethod
    def diff_central(x, y):
        """
        Compute the central difference of y with respect to x.

        Parameters:
        x : array-like
            Independent variable.
        y : array-like
            Dependent variable.

        Returns:
        array-like
            Central difference values.
        """
        x0, x1, x2 = x[:-2], x[1:-1], x[2:]
        y0, y1, y2 = y[:-2], y[1:-1], y[2:]
        f = (x2 - x1) / (x2 - x0)
        # Weighted central difference formula
        return (1 - f) * (y2 - y1) / (x2 - x1) + f * (y1 - y0) / (x1 - x0)

    def isotropic_decomposition(self):
        """
        Perform the Helmholtz decomposition for isotropic spectra.

        Returns:
        tuple
            Rotational and divergent components, and other GM-specific values if GM=True.
        """
        s = np.log(self.k)
        Cu, Cv = self.Cu, self.Cv

        # Initialize arrays for results
        Fphi = np.zeros_like(Cu)
        Fpsi = np.zeros_like(Cu)
        Cphi = np.zeros_like(Cu)
        Cpsi = np.zeros_like(Cu)

        # Loop over wavenumber bins to compute integrals
        for i in range(s.size - 1):
            sh, ch = sinh(s[i] - s[i:]), cosh(s[i] - s[i:])
            Fp = Cv[i:] * sh + Cu[i:] * ch
            Fs = Cu[i:] * sh + Cv[i:] * ch

            # Integrate using Simpson's rule
            Fpsi[i] = integrate.simpson(Fs, x=s[i:])
            Fphi[i] = integrate.simpson(Fp, x=s[i:])

            # Set negative values to "Nan" (should be np.nan for numerical work)
            Fpsi[Fpsi < 0.] = "Nan"
            Fphi[Fphi < 0.] = "Nan"

        # Compute derivatives using central difference
        dFphi = self.diff_central(self.k, Fphi)
        dFphi = np.interp(self.k, self.k[1:-1], dFphi.real)

        dFpsi = self.diff_central(self.k, Fpsi)
        dFpsi = np.interp(self.k, self.k[1:-1], dFpsi.real)

        # Compute rotational and divergent KE spectra
        Kpsi = (Fpsi - self.k * dFpsi) / 2
        Kphi = (Fphi - self.k * dFphi) / 2

        # Compute wave energy
        E_w = Fphi - self.k * dFphi

        return Fpsi, Fphi, Kpsi, Kphi, E_w

    def model3_decomposition(self):
        """
        Perform the Helmholtz decomposition using model 3 (with cross-spectra and optional angle).

        Returns:
        tuple
            Rotational and divergent components, and other GM-specific values if GM=True.
        """
        s = self.k
        Cu, Cv, Cuv = self.Cu, self.Cv, self.Cuv
        u, v = self.u, self.v

        # Initialize arrays for results
        Kpsi = np.zeros_like(Cu)
        Kphi = np.zeros_like(Cu)

        # Compute mean squared velocities and cross-term
        w = np.ones(len(u))
        Eu2 = np.nansum((np.dot(u**2, w))) / np.sum(w)
        Ev2 = np.nansum((np.dot(v**2, w))) / np.sum(w)
        Euv = np.nansum((np.dot(u * v, w))) / np.sum(w)

        # Compute E parameter based on theta or velocity moments
        if self.theta is None:
            E = (Eu2 - Ev2) / Euv
        else:
            E = 2 * (1 / np.tan(2 * self.theta))

        # Loop over wavenumber bins to compute integrals
        for i in range(s.size - 1):
            K = Cv[i:] - Cu[i:] + Cuv[i:] * E

            Kpsi[i] = .5 * (Cv[i] + ((1 / s[i]) * integrate.simpson(K, x=s[i:])))
            Kphi[i] = .5 * (Cu[i] - ((1 / s[i]) * integrate.simpson(K, x=s[i:])))

            # Set negative values to "Nan" (should be np.nan for numerical work)
            Kpsi[Kpsi < 0.] = "Nan"
            Kphi[Kphi < 0.] = "Nan"

        # Compute derivatives using central difference
        dKpsi = self.diff_central(Kpsi, s)
        dKpsi = np.interp(self.k, self.k[1:-1], dKpsi)

        dCv = self.diff_central(Cv, s)
        dCv = np.interp(self.k, self.k[1:-1], dCv)

        # Compute cross-spectral components
        Cpsi_u = 4 * Kpsi - Cv + E * Cuv + s * (2 * dKpsi - dCv)
        Cphi_v = Cpsi_u + Cv - 2 * Kpsi

        dCphi_v = self.diff_central(Cphi_v, s)
        dCphi_v = np.interp(self.k, self.k[1:-1], dCphi_v)

        Cphi_uv = (1 / E) * (Cphi_v - Cpsi_u + s * dCphi_v)

        # Compute final spectral components
        Cphi_u = Cu - Cpsi_u
        Cpsi_v = Cv - Cphi_v
        Cpsi_uv = Cuv - Cphi_uv

        return Cpsi_u, Cphi_u, Cpsi_v, Cphi_v, Cpsi_uv, Cphi_uv, Kpsi, Kphi, Eu2, Ev2, Euv