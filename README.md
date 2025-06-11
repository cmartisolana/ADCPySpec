# ADCPySpec

## A Python package for ADCP spectral analysis

ADCPySpec is a Python package for spectral analysis of Acoustic Doppler Current Profiler (ADCP) and velocity field data. It provides tools for computing power spectra, cross-spectra, and performing Helmholtz decomposition of velocity fields.

---

## Features

- **Power Spectrum Estimation:** Compute power spectral density of velocity signals.
- **Cross-Spectrum Analysis:** Analyze relationships between velocity components.
- **Helmholtz Decomposition:** Decompose velocity spectra into rotational and divergent components.
- **Easy-to-use API:** Simple classes and functions for common spectral analysis tasks.

---

## Installation

Clone the repository and install with pip:

```sh
git clone https://github.com/cmartisolana/ADCPySpec.git
cd ADCPySpec
pip install .
```

---

## Requirements

- Python >= 3.6
- numpy
- scipy
- pytest (for running tests)

---

## Usage

### Helmholtz Decomposition Example

See [examples/helmholtz_decomposition_example.ipynb](examples/helmholtz_decomposition_example.ipynb) for a full notebook.

```python
import numpy as np
from ADCPySpec.helmholtz import HelmholtzDecomposition

k = np.linspace(0, 5, 5000)
Cu = np.exp(-np.pi*k**2)
Cv = 0.5 * np.exp(-np.pi*k**2)
Cuv = np.zeros_like(k)

helmholtz = HelmholtzDecomposition(k, Cu, Cv, Cuv)
Fpsi, Fphi, Kpsi, Kphi, E_w = helmholtz.isotropic_decomposition()
```

### Spectrum Processor Example

See [examples/spectrum_processor_example.ipynb](examples/spectrum_processor_example.ipynb) for a full notebook.

```python
import numpy as np
from ADCPySpec.spectrum import SpectrumProcessor

dt = 1.0
N = 1024
t = np.arange(N) * dt
u = 2 * np.sin(2 * np.pi * 0.05 * t) + 1 * np.sin(2 * np.pi * 0.15 * t) + 0.5 * np.random.randn(N)

spec_proc = SpectrumProcessor(dt=dt, nfft=256, window='hann')
freqs, Su = spec_proc.compute_spectrum(u)
```

---

## Testing

Run all tests with:

```sh
pytest
```

---

## Project Structure

```
ADCPySpec/
    __init__.py
    helmholtz.py
    spectrum.py
examples/
    helmholtz_decomposition_example.ipynb
    spectrum_processor_example.ipynb
tests/
    test_helmholtz.py
    test_spectrum.py
```

---

## License

This project is licensed under the MIT License.

---

## Author

Cristina Martí-Solana 
[cmarti@imedea.uib-csic.es](mailto:cmarti@imedea.uib-csic.es)
