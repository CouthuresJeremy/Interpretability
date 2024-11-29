import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma


def hypersphere_approximation(points, bounding_box, norm=True):
    """
    Approximate Voronoi volumes for points in D-dimensional space using a hypersphere.
    :param points: ndarray of shape (n_points, dimensions)
    :param bounding_box: ndarray of shape (2, dimensions), [[min_x, ..., min_D], [max_x, ..., max_D]]
    :return: List of approximate Voronoi cell volumes
    """
    dimensions = points.shape[1]
    kdtree = cKDTree(points)
    volumes = []

    for point in points:
        distances, indices = kdtree.query(point, k=2)
        nearest_distance = distances[1]

        # Approximate the Voronoi region as a hypersphere
        hypersphere_volume = (
            np.pi ** (dimensions / 2) / gamma(dimensions / 2 + 1)
        ) * nearest_distance**dimensions

        # Clip the volume to the bounding box if necessary
        box_volume = np.prod(bounding_box[1] - bounding_box[0])
        volumes.append(min(hypersphere_volume, box_volume))

    if norm:
        volumes = volumes / np.sum(volumes)

    return volumes
