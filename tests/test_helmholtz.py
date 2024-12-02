import pytest
import numpy as np
from ADCPySpec.helmholtz import HelmholtzProcessor

@pytest.fixture
def sample_data():
    # Create sample x, y, u, v data for testing
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    u = np.sin(np.outer(y, x))  # Example zonal velocity component
    v = np.cos(np.outer(y, x))  # Example meridional velocity component
    return x, y, u, v

def test_initialize_helmholtz_processor(sample_data):
    x, y, u, v = sample_data
    processor = HelmholtzProcessor(u, v, x, y)
    assert processor.u is not None
    assert processor.v is not None
    assert processor.x is not None
    assert processor.y is not None
    assert processor.method == 'linear'

def test_interpolate_velocity(sample_data):
    x, y, u, v = sample_data
    processor = HelmholtzProcessor(u, v, x, y, method='linear')
    grid_x = np.linspace(0, 10, 100)
    grid_y = np.linspace(0, 10, 100)
    u_interp, v_interp = processor.interpolate_velocity(grid_x, grid_y)
    assert u_interp.shape == (100, 100)
    assert v_interp.shape == (100, 100)
    assert not np.isnan(u_interp).all()
    assert not np.isnan(v_interp).all()

def test_compute_divergence(sample_data):
    x, y, u, v = sample_data
    processor = HelmholtzProcessor(u, v, x, y)
    divergence = processor.compute_divergence()
    assert divergence.shape == u.shape
    assert not np.isnan(divergence).all()

def test_compute_vorticity(sample_data):
    x, y, u, v = sample_data
    processor = HelmholtzProcessor(u, v, x, y)
    vorticity = processor.compute_vorticity()
    assert vorticity.shape == u.shape
    assert not np.isnan(vorticity).all()

if __name__ == "__main__":
    pytest.main()
