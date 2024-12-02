import numpy as np
from scipy.interpolate import griddata

class HelmholtzProcessor:
    def __init__(self, u, v, x, y, method='linear'):
        """
        Initialize the HelmholtzProcessor with velocity field components.

        Parameters:
        u (array-like): Zonal velocity component.
        v (array-like): Meridional velocity component.
        x (array-like): x-coordinates of the velocity field.
        y (array-like): y-coordinates of the velocity field.
        method (str, optional): Interpolation method to use ('linear', 'nearest', 'cubic'). Default is 'linear'.
        """
        self.u = u
        self.v = v
        self.x = x
        self.y = y
        self.method = method

    def interpolate_velocity(self, grid_x, grid_y):
        """
        Interpolate the velocity field to a new grid.

        Parameters:
        grid_x (array-like): x-coordinates of the new grid.
        grid_y (array-like): y-coordinates of the new grid.

        Returns:
        tuple: Interpolated zonal and meridional velocity components (u_interp, v_interp).
        """
        # Flatten the input coordinates and velocity components
        points = np.array([self.x.flatten(), self.y.flatten()]).T
        u_values = self.u.flatten()
        v_values = self.v.flatten()

        # Create a meshgrid for the new coordinates
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_points = np.array([grid_x.flatten(), grid_y.flatten()]).T

        # Interpolate the zonal (u) and meridional (v) velocity components
        u_interp = griddata(points, u_values, grid_points, method=self.method).reshape(grid_x.shape)
        v_interp = griddata(points, v_values, grid_points, method=self.method).reshape(grid_y.shape)

        return u_interp, v_interp

    def compute_divergence(self):
        """
        Compute the divergence of the velocity field.

        Returns:
        array-like: Divergence of the velocity field.
        """
        # Compute the partial derivatives of u and v with respect to x and y
        du_dx = np.gradient(self.u, axis=1)
        dv_dy = np.gradient(self.v, axis=0)

        # Divergence is the sum of the partial derivatives
        divergence = du_dx + dv_dy

        return divergence

    def compute_vorticity(self):
        """
        Compute the vorticity of the velocity field.

        Returns:
        array-like: Vorticity of the velocity field.
        """
        # Compute the partial derivatives of u and v with respect to y and x
        dv_dx = np.gradient(self.v, axis=1)
        du_dy = np.gradient(self.u, axis=0)

        # Vorticity is the difference between the partial derivatives
        vorticity = dv_dx - du_dy

        return vorticity