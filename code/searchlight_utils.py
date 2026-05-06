import numpy as np
from scipy.stats import spearmanr
import pingouin as pg
import pandas as pd

def create_sphere(center, radius, shape):
    """
    Create a boolean mask for a sphere centered at 'center' with the given 'radius'.

    Parameters:
    center (tuple): The center of the sphere (x, y, z coordinates).
    radius (int): The radius of the sphere.
    shape (tuple): The shape of the 3D data (width, height, depth).

    Returns:
    numpy.ndarray: A boolean mask where voxels within the sphere are True.
    """
    xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance_squared = (xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2
    return distance_squared <= radius**2

def compute_rdm(data):
    """
    Compute a lower-triangular RDM vector (1 - Pearson correlation).
    Input shape should be conditions x voxels.
    """
    correlation_distances = np.round(1 - np.corrcoef(data), 6)
    n = correlation_distances.shape[0]
    rdm_vector = [correlation_distances[i, j] for i in range(n) for j in range(n) if i > j]
    return rdm_vector

def process_voxel(args):
    """
    Compute RSA correlation for a spherical searchlight centered on voxel (x, y, z).
    Returns: (x, y, z, correlation with behavioral RDM)
    """
    x, y, z, fmri_data, mask, radius, behavioral_rdm = args
    if mask.get_fdata()[x, y, z]:
        sphere_mask = create_sphere((x, y, z), radius, fmri_data.shape[:3])
        sphere_data = fmri_data.get_fdata()[sphere_mask]
        neural_rdm = compute_rdm(sphere_data.T)  # shape: conditions x voxels
        return x, y, z, spearmanr(neural_rdm, behavioral_rdm).correlation
    return x, y, z, np.nan

def process_voxel_uniqvar(args):
    """
    Compute semi-partial Spearman correlations for a spherical searchlight.
    Returns: (x, y, z, unique variance for model1, unique variance for model2)
    """
    x, y, z, fmri_data, mask, radius, model1_rdm, model2_rdm = args

    if mask.get_fdata()[x, y, z]:
        sphere_mask = create_sphere((x, y, z), radius, fmri_data.shape[:3])
        sphere_data = fmri_data.get_fdata()[sphere_mask]
        neural_rdm = compute_rdm(sphere_data.T)

        if np.sum(~np.isnan(neural_rdm)) < 3:
            return x, y, z, np.nan, np.nan  # not enough valid values for reliable correlation

        data = pd.DataFrame({'X': neural_rdm, 'Y1': model1_rdm, 'Y2': model2_rdm})
        sr1 = pg.partial_corr(data=data, x='X', y='Y1', y_covar=['Y2'], method='spearman')['r'].iloc[0]
        sr2 = pg.partial_corr(data=data, x='X', y='Y2', y_covar=['Y1'], method='spearman')['r'].iloc[0]
        return x, y, z, sr1, sr2

    return x, y, z, np.nan, np.nan
