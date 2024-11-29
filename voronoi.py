import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.special import gamma
from shapely.geometry import LineString, Polygon


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


def voronoi_volumes(vor, bounding_box):
    """
    Calculate the volume of clipped Voronoi cells.
    :param vor: The Voronoi object containing the regions and vertices
    :param bounding_box: The bounding box to clip the regions to
    :return: List of volumes for each Voronoi cell
    """
    # clipped_regions = clip_infinite_regions(vor, bounding_box)
    clipped_regions = clip_regions(vor, bounding_box)
    # print(clipped_regions)
    volumes = []

    for region in clipped_regions:
        if len(region) == 0:
            volumes.append(0)
            continue

        # Compute the convex hull of the clipped region
        try:
            hull = ConvexHull(np.array(region))
            volumes.append(hull.volume)
        except:
            volumes.append(
                0
            )  # If the region doesn't form a valid hull, assign zero volume

    return volumes


def clip_regions(vor, bounding_box, verbose=False):
    """
    Clip the Voronoi regions to a finite bounding box.
    :param vor: The Voronoi object containing the regions and vertices
    :param bounding_box: The bounding box to clip the regions to
    :return: List of clipped regions
    """
    clipped_regions = []

    # Handle regions with infinite ridges
    infinite_ridge_point_indices_list = find_infinite_ridge_sharing_regions(
        vor, bounding_box
    )

    # Get regions corresponding to the infinite ridges indices
    found = 0
    infinite_ridge_regions = {}
    for ridge_point_indices in infinite_ridge_point_indices_list:

        ridge_vertices = vor.ridge_dict[tuple(ridge_point_indices)]

        for region_index in range(len(vor.regions)):
            region = vor.regions[region_index]

            if len(region) == 0:
                continue

            if set(ridge_vertices).issubset(set(region)):
                found += 1
                infinite_ridge_regions.setdefault(
                    tuple(ridge_point_indices), []
                ).append(region_index)

    reversed_infinite_ridge_regions = {}
    for key, value in infinite_ridge_regions.items():
        for region_index in value:
            reversed_infinite_ridge_regions.setdefault(region_index, []).append(key)

    if verbose:
        print(f"Found: {found}")

        print(f"Infinite Ridge Regions: {infinite_ridge_regions}")
        print(f"Reversed Infinite Ridge Regions: {reversed_infinite_ridge_regions}")

    all_infinite_ridge_regions = set(reversed_infinite_ridge_regions.keys())
    if verbose:
        print(f"All Infinite Ridge Regions: {all_infinite_ridge_regions}")

    # Handle infinite regions without infinite ridges and finite regions
    for region_index in range(len(vor.regions)):
        if region_index in all_infinite_ridge_regions:
            continue

        region = vor.regions[region_index]

        if len(region) == 0:
            continue

        # Remove the point at infinity (-1)
        region = [r for r in region if r != -1]

        vertices = vor.vertices[region]
        # vertices = clip_to_bounding_box(vertices, bounding_box)
        vertices = clip_vertex_region(vertices, bounding_box)
        clipped_regions.append(vertices)

    bounded_ridge_points = get_bounded_ridge_points(vor, bounding_box)

    # Get regions corresponding to the infinite ridges indices
    for region_index in reversed_infinite_ridge_regions.keys():
        ridge_point_indices = reversed_infinite_ridge_regions[region_index]
        assert len(ridge_point_indices) == 1, "Only one ridge point index"
        ridge_point_indices = ridge_point_indices[0]
        region = vor.regions[region_index]

        # Replace the point at infinity (-1) with the bounded ridge point
        # Get the position of the infinite point in the region
        infinite_point_index = region.index(-1)
        region = [r for r in region if r != -1]
        vertices = vor.vertices[region]
        # Add the new bounded ridge point
        vertices = np.insert(
            vertices,
            infinite_point_index,
            bounded_ridge_points.get(tuple(ridge_point_indices)),
            axis=0,
        )
        vertices = clip_vertex_region(vertices, bounding_box)
        clipped_regions.append(vertices)

    return clipped_regions


def find_infinite_ridge_sharing_regions(vor, bounding_box):
    """
    Identify regions that share an infinite ridge in the Voronoi diagram.
    A ridge with -1 as one of its vertices is an infinite ridge, and we find
    which regions share it.

    :param vor: The Voronoi object containing the diagram
    :return: A list of tuples, where each tuple contains two regions (indexes)
             that share an infinite ridge
    """
    infinite_ridge_sharing_regions = []

    for ridge_point_indices, ridge_vertices in vor.ridge_dict.items():
        if not -1 in ridge_vertices:
            continue

        valid_ridge_vertices = np.array([v for v in ridge_vertices if v != -1])
        valid_ridge_vertices = vor.vertices[valid_ridge_vertices]

        # Ensure that the valid ridge vertex is inside the bounding box
        doSkip = False
        for axis in range(bounding_box.shape[1]):
            if not all(
                bounding_box[0, axis] <= v[axis] <= bounding_box[1, axis]
                for v in valid_ridge_vertices
            ):
                doSkip = True
                break

        if doSkip:
            continue

        infinite_ridge_sharing_regions.append(ridge_point_indices)

    return infinite_ridge_sharing_regions


def clip_vertex_region(vertices, bounding_box):
    """
    Clip the infinite Voronoi region to a finite bounding box.
    :param vor: The Voronoi object containing the regions and vertices
    :param region: The region to clip
    :param bounding_box: The bounding box to clip the region to
    :return: List of clipped vertices
    """
    if len(vertices) == 0:
        return []

    # Make a polygon from the vertices
    polygon = Polygon(vertices)

    # print(f"Polygon: {polygon}")

    # Make a bounding box (hypercube)
    box = Polygon(
        [
            (bounding_box[0, 0], bounding_box[0, 1]),
            (bounding_box[1, 0], bounding_box[0, 1]),
            (bounding_box[1, 0], bounding_box[1, 1]),
            (bounding_box[0, 0], bounding_box[1, 1]),
        ]
    )

    # Find the intersection of the polygon with the bounding box
    try:
        intersection = polygon.intersection(box)
    except Exception as e:
        print(e)
        return []

    if intersection.is_empty:
        return vertices

    assert intersection.geom_type != "Point", "Intersection should not be a point"

    assert (
        intersection.geom_type == "LineString" or intersection.geom_type == "Polygon"
    ), "Intersection should be a LineString or Polygon"

    if intersection.geom_type == "LineString":
        print(list(intersection.coords))
        return []
    else:
        return np.array(list(intersection.exterior.coords))


def get_bounded_ridge_points(vor, bounding_box):
    infinite_ridge_point_indices_list = find_infinite_ridge_sharing_regions(
        vor, bounding_box
    )

    bounded_ridge_points = {}

    for ridge_point_indices in infinite_ridge_point_indices_list:
        ridge_vertices = vor.ridge_dict[tuple(ridge_point_indices)]

        valid_ridge_vertices = np.array([v for v in ridge_vertices if v != -1])
        valid_ridge_vertices = vor.vertices[valid_ridge_vertices]

        ridge_points = []
        for index in ridge_point_indices:
            ridge_points.append(vor.points[index])

        print(f"Ridge Points: {ridge_points}, {ridge_point_indices}, {ridge_vertices}")

        # Compute the midpoint of the ridge
        midpoint = np.mean(ridge_points, axis=0)
        print(f"Midpoint: {midpoint}")

        # Determine the intersection of the line (valid_ridge_vertices[0], midpoint) with the bounding box
        line = LineString([valid_ridge_vertices[0], midpoint])

        # Extend the line by a factor of 10
        extended_line = LineString(
            [
                (
                    midpoint[0] + 1000 * (midpoint[0] - valid_ridge_vertices[0][0]),
                    midpoint[1] + 1000 * (midpoint[1] - valid_ridge_vertices[0][1]),
                ),
                (
                    midpoint[0] - 1000 * (midpoint[0] - valid_ridge_vertices[0][0]),
                    midpoint[1] - 1000 * (midpoint[1] - valid_ridge_vertices[0][1]),
                ),
            ]
        )

        box = Polygon(
            [
                (bounding_box[0, 0], bounding_box[0, 1]),
                (bounding_box[1, 0], bounding_box[0, 1]),
                (bounding_box[1, 0], bounding_box[1, 1]),
                (bounding_box[0, 0], bounding_box[1, 1]),
            ]
        )

        box_boundary = box.boundary

        # Make box boundary coordinates * 2
        # box_boundary = LineString([(-b[0] * 4, b[1] * 4) for b in box_boundary.coords])
        # box_boundary = LineString([(b[0] * 4, b[1] * 4) for b in box_boundary.coords])

        box_boundary = LineString(
            [
                (-3, -3),
                (4, -3),
                (4, 4),
                (-3, 4),
                (-3, -3),
            ]
        )

        print(f"Box Boundary: {box_boundary}")

        intersection = extended_line.intersection(box_boundary)

        if intersection.is_empty:
            continue

        if intersection.geom_type == "Point":
            bounded_ridge_points[tuple(ridge_point_indices)] = np.array(
                intersection.coords[0]
            )
            continue

        print(f"Intersection: {intersection}")

        if intersection.geom_type == "MultiPoint":
            # Find the closest point to the midpoint
            closest_point = min(
                intersection.geoms,
                key=lambda p: np.linalg.norm(midpoint - p.coords[0]),
            )
            bounded_ridge_points[tuple(ridge_point_indices)] = closest_point.coords[0]

        if intersection.geom_type == "LineString":
            assert False, "LineString intersection"

    return bounded_ridge_points
