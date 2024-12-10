import numpy as np
from numpy import pi, sinh, cosh
from scipy import integrate

try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass


class HelmholtzDecomposition:
    """
    A class to compute the Helmholtz decomposition of KE spectra into rotational
    and divergent components based on Buhler et al. (JFM 2014).
    """

    def __init__(self, k, Cu, Cv, Cuv, u=None, v=None):
        """
        Initialize the decomposition class.

        Parameters:
        k : array-like
            Wavenumber array.
        Cu : array-like
            Spectrum of across-track velocity.
        Cv : array-like
            Spectrum of along-track velocity.
        GM : bool, optional
            Use GM decomposition if True. Default is False.
        gm_file_path : str, optional
            Path to the GM data file, required if GM=True.
        """
        self.k = k
        self.Cu = Cu
        self.Cv = Cv
        self.Cuv = Cuv
        self.u = u
        self.v = v

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
        return (1 - f) * (y2 - y1) / (x2 - x1) + f * (y1 - y0) / (x1 - x0)

    def isotropic_decomposition(self):
        """
        Perform the Helmholtz decomposition.

        Returns:
        tuple
            Rotational and divergent components, and other GM-specific values if GM=True.
        """
        s = np.log(self.k)
        Cu, Cv = self.Cu, self.Cv

        Fphi = np.zeros_like(Cu)
        Fpsi = np.zeros_like(Cu)
        Cphi = np.zeros_like(Cu)
        Cpsi = np.zeros_like(Cu)

        for i in range(s.size - 1):
            sh, ch = sinh(s[i] - s[i:]), cosh(s[i] - s[i:])
            Fp = Cu[i:] * sh + Cv[i:] * ch
            Fs = Cv[i:] * sh + Cu[i:] * ch

            Fpsi[i] = integrate.simpson(Fs,x=s[i:])
            Fphi[i] = integrate.simpson(Fp,x=s[i:])

            Fpsi[Fpsi < 0.] = "Nan"
            Fphi[Fphi < 0.] = "Nan"

        Cpsi = Fpsi - Fphi + Cu
        Cphi = Fphi - Fpsi + Cv

        dFphi = self.diff_central(self.k, Fphi)
        dFphi = np.interp(self.k, self.k[1:-1], dFphi.real)

        E_w = Fphi - self.k * dFphi

        return Cpsi, Cphi, E_w
    
    def model3_decomposition(self):
        """
        Perform the Helmholtz decomposition.

        Returns:
        tuple
            Rotational and divergent components, and other GM-specific values if GM=True.
        """
        s = self.k
        Cu, Cv, Cuv = self.Cu, self.Cv, self.Cuv
        u,v = self.u,self.v

        Kpsi = np.zeros_like(Cu)
        Kphi = np.zeros_like(Cu)

        w = np.ones(len(u))
        Eu2 = np.nansum((np.dot(u**2,w))) / np.sum(w)
        Ev2 = np.nansum((np.dot(v**2,w))) / np.sum(w)
        Euv = np.nansum((np.dot(u*v,w))) / np.sum(w)

        E = (Eu2 - Ev2) / Euv

        for i in range(s.size - 1):
            K = Cv[i:] - Cu[i:] + Cuv[i:]*E

            Kpsi[i] = .5 * (Cv[i] + (1/s[i]) * integrate.simpson(K,x=s[i:]))
            Kphi[i] = .5 * (Cu[i] - (1/s[i]) * integrate.simpson(K,x=s[i:]))

            Kpsi[Kpsi < 0.] = "Nan"
            Kphi[Kphi < 0.] = "Nan"

        dKpsi = self.diff_central(Kpsi,s)
        dKpsi = np.interp(self.k, self.k[1:-1], dKpsi)

        dCv = self.diff_central(Cv,s)
        dCv = np.interp(self.k, self.k[1:-1], dCv)

        Cpsi_u = 4*Kpsi - Cv + E*Cuv + s * ( 2*dKpsi - dCv)
        Cphi_v = Cpsi_u + Cv - 2*Kpsi

        dCphi_v = self.diff_central(Cphi_v,s)
        dCphi_v = np.interp(self.k, self.k[1:-1], dCphi_v)
        
        Cphi_uv = (1/E) * (Cphi_v - Cpsi_u + s * dCphi_v)

        Cphi_u = Cu - Cpsi_u
        Cpsi_v = Cv - Cphi_v
        Cpsi_uv = Cuv - Cphi_uv


        return Cpsi_u,Cphi_u,Cpsi_v,Cphi_v,Cpsi_uv,Cphi_uv, Kpsi, Kphi, Eu2, Ev2, Euv
