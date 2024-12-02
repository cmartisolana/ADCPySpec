import pytest
import numpy as np
from ADCPySpec.spectrum import SpectrumProcessor

# Fixture for SpectrumProcessor sample data
@pytest.fixture
def spectrum_sample_data():
    # Create sample x, y1, y2 data for testing SpectrumProcessor
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)  # Example input signal 1
    y2 = np.cos(x)  # Example input signal 2
    return x, y1, y2

# Tests for SpectrumProcessor

def test_initialize_spectrum_processor(spectrum_sample_data):
    x, y1, y2 = spectrum_sample_data
    processor = SpectrumProcessor(x, y1, y2)
    assert processor.x is not None
    assert processor.y1 is not None
    assert processor.y2 is not None
    assert processor.win is None
    assert processor.pad is True
    assert processor.ax == 0

def test_compute_cross_spectrum(spectrum_sample_data):
    x, y1, y2 = spectrum_sample_data
    processor = SpectrumProcessor(x, y1, y2, win='hanning', pad=True, ax=0)
    freqs, ipy1y2, ipy1, ipy2, iphase_y1y2, dofw = processor.compute_cross_spectrum()
    assert len(freqs) == len(y1)
    assert ipy1y2.shape == y1.shape
    assert ipy1.shape == y1.shape
    assert ipy2.shape == y2.shape
    assert iphase_y1y2.shape == y1.shape

def test_calc_ispec():
    k = np.linspace(0.1, 10, 100)
    l = np.linspace(0.1, 10, 100)
    E = np.random.rand(100, 100)  # Example spectrum
    processor = SpectrumProcessor(k, E)
    kr, Er = processor.calc_ispec(k, l, E, ndim=2)
    assert len(kr) > 0
    assert len(Er) == len(kr)

def test_spectral_slope(spectrum_sample_data):
    x, y1, _ = spectrum_sample_data
    k = np.linspace(0.1, 10, 100)
    E = np.exp(-k)  # Example spectrum values
    kmin, kmax = 1, 5
    stdE = 0.1
    processor = SpectrumProcessor(x, y1)
    slope, uncertainty = processor.spectral_slope(k, E, kmin, kmax, stdE)
    assert isinstance(slope, float)
    assert isinstance(uncertainty, float)

if __name__ == "__main__":
    pytest.main()