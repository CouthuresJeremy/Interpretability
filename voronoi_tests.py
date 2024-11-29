import numpy as np
from scipy.spatial import Voronoi
from scipy.special import gamma
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from voronoi import hypersphere_approximation, voronoi_volumes


def generate_random_points(n_points, dimensions, seed=None):
    """
    Generate random points in D-dimensional space.
    :param n_points: Number of points
    :param dimensions: Number of dimensions
    :param seed: Random seed
    :return: ndarray of shape (n_points, dimensions)
    """
    rng = np.random.default_rng(seed)
    return rng.random((n_points, dimensions))


def clip_to_bounding_box(vertices, bounding_box):
    """
    Clip vertices to the bounding box.
    :param vertices: ndarray of Voronoi vertices
    :param bounding_box: ndarray of shape (2, D), [[min_x, ..., min_D], [max_x, ..., max_D]]
    :return: ndarray of clipped vertices
    """
    return np.clip(vertices, bounding_box[0], bounding_box[1])


def clip_line_segment(p1, p2, bounding_box):
    """
    Clip the line segment defined by points p1 and p2 to the bounding box using Shapely.
    :param p1: Starting point of the line segment
    :param p2: Ending point of the line segment
    :param bounding_box: The bounding box to clip the line segment to
    :return: List of clipped vertices
    """
    box = Polygon(
        [
            (bounding_box[0, 0], bounding_box[0, 1]),
            (bounding_box[1, 0], bounding_box[0, 1]),
            (bounding_box[1, 0], bounding_box[1, 1]),
            (bounding_box[0, 0], bounding_box[1, 1]),
        ]
    )

    line = LineString([p1, p2])
    intersection = line.intersection(box)

    if intersection.is_empty:
        return []

    if intersection.geom_type == "Point":
        return [np.array(intersection.coords[0])]

    if intersection.geom_type == "LineString":
        return np.array(list(intersection.coords))


def clip_infinite_regions(vor, bounding_box):
    """
    Clip the infinite Voronoi regions to a finite bounding box.
    :param vor: The Voronoi object containing the regions and vertices
    :param bounding_box: The bounding box to clip the regions to
    :return: List of clipped regions
    """
    clipped_regions = []

    for region_index in range(len(vor.regions)):
        region = vor.regions[region_index]

        if len(region) == 0:
            continue

        clipped_vertices = clip_infinite_region(vor, region, bounding_box)

        clipped_regions.append(clipped_vertices)

        # if -1 in region:
        #     # This region is infinite, clip it to the bounding box
        #     clipped_vertices = []
        #     for ridge in vor.ridge_vertices:
        #         if -1 in ridge:
        #             # If the ridge has an infinite edge, we need to clip it
        #             ridge_vertices = np.array(
        #                 [vor.vertices[r] for r in ridge if r != -1]
        #             )
        #             if len(ridge_vertices) == 2:
        #                 p1, p2 = ridge_vertices
        #                 # Clip the line segment between p1 and p2 to the bounding box
        #                 clipped_vertices.append(clip_line_segment(p1, p2, bounding_box))
        #     clipped_regions.append(clipped_vertices)
        # else:
        #     # Valid, finite region
        #     vertices = vor.vertices[region]
        #     clipped_vertices = clip_to_bounding_box(vertices, bounding_box)
        #     clipped_regions.append(clipped_vertices)

    # print(f"Infinite ridge: {find_infinite_ridge_sharing_regions(vor)}")

    return clipped_regions


def clip_infinite_region(vor, region, bounding_box):
    """
    Clip the infinite Voronoi region to a finite bounding box.
    :param vor: The Voronoi object containing the regions and vertices
    :param region: The region to clip
    :param bounding_box: The bounding box to clip the region to
    :return: List of clipped vertices
    """
    if len(region) == 0:
        return []

    if -1 in region:
        # Remove the point at infinity (-1)
        region = [r for r in region if r != -1]

        # Valid, finite region
        vertices = vor.vertices[region]

        # print(list(list(v) for v in vertices))

        # return vertices

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

        return vertices
    else:
        # return []
        # Valid, finite region
        vertices = vor.vertices[region]

        # Make a polygon from the vertices
        polygon = Polygon(vertices)

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
        intersection = polygon.intersection(box)

        if intersection.is_empty:
            return vertices

        assert intersection.geom_type != "Point", "Intersection should not be a point"

        assert (
            intersection.geom_type == "LineString"
            or intersection.geom_type == "Polygon"
        ), "Intersection should be a LineString or Polygon"

        if intersection.geom_type == "LineString":
            print(list(intersection.coords))
            return []
        else:
            return np.array(list(intersection.exterior.coords))

        # clipped_vertices = []
        # for i in range(len(vertices)):
        #     p1 = vertices[i]

        #     line = LineString([p1, p2])
        #     intersection = line.intersection(box)

        #     if intersection.is_empty:
        #         return []

        #     if intersection.geom_type == "Point":
        #         return [np.array(intersection.coords[0])]

        #     if intersection.geom_type == "LineString":
        #         return np.array(list(intersection.coords))
        return clip_to_bounding_box(vertices, bounding_box)


def compare_methods(dimensions, n_points=50, bounding_box=[[0, 0, 0], [1, 1, 1]]):
    bounding_box = np.array([[0] * dimensions, [1] * dimensions])
    print(f"Bounding Box: {bounding_box}")
    points = generate_random_points(n_points, dimensions, seed=42)
    print(f"Points: {points}")
    vor = Voronoi(points)
    # print(f"Vor: {vor.points}")
    # print(len(vor.regions))
    # print(vor.regions)

    try:
        exact_volumes = voronoi_volumes(vor, bounding_box)
    except Exception as e:
        print(e)
        exact_volumes = [0] * len(points)
    approximate_volumes = hypersphere_approximation(points, bounding_box)

    print(f"Exact Volumes sum: {sum(exact_volumes)}")
    print(f"Approximate Volumes sum: {sum(approximate_volumes)}")

    # Renormalize the approximate volumes
    approximate_volumes = np.array(approximate_volumes) / sum(approximate_volumes)

    if dimensions == 2:
        # Plot the Voronoi diagram and compare with the clipped regions
        # Make a subplot for unclipped and clipped regions
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the Voronoi diagram
        for region in vor.regions:
            if len(region) == 0:
                continue
            vertices = vor.vertices[region]
            ax[0].fill(*zip(*vertices), alpha=0.5, edgecolor="black")
        # for regions in find_infinite_ridge_sharing_regions(vor):
        #     for region in regions:
        #         region = vor.regions[region]
        #         if len(region) == 0:
        #             continue
        #         vertices = vor.vertices[region]
        #         ax[0].fill(*zip(*vertices), alpha=0.5, edgecolor="black")

        # # Plot the ridge vertices
        # for ridge in vor.ridge_vertices:
        #     if -1 in ridge:
        #         continue
        #     vertices = vor.vertices[ridge]
        #     ax[0].plot(*zip(*vertices), color="black")

        ax[0].scatter(*zip(*points), color="red")
        ax[0].scatter(*zip(*vor.vertices), color="black", marker="x")

        print(f"Vor ridge dict: {vor.ridge_dict}")

        ax[0].set_title("Voronoi Diagram")
        ax[0].set_xlim(bounding_box[0, 0], bounding_box[1, 0])
        ax[0].set_ylim(bounding_box[0, 1], bounding_box[1, 1])

        # Plot the clipped regions

        # # Handle infinite regions
        # infinite_ridge_point_indices_list = find_infinite_ridge_sharing_regions(
        #     vor, bounding_box
        # )

        # # Get regions corresponding to the infinite ridges indices
        # found = 0
        # infinite_ridge_regions = {}
        # for ridge_point_indices in infinite_ridge_point_indices_list:

        #     ridge_vertices = vor.ridge_dict[tuple(ridge_point_indices)]

        #     for region_index in range(len(vor.regions)):
        #         region = vor.regions[region_index]

        #         if len(region) == 0:
        #             continue

        #         if set(ridge_vertices).issubset(set(region)):
        #             found += 1
        #             infinite_ridge_regions.setdefault(
        #                 tuple(ridge_point_indices), []
        #             ).append(region_index)

        # reversed_infinite_ridge_regions = {}
        # for key, value in infinite_ridge_regions.items():
        #     for region_index in value:
        #         reversed_infinite_ridge_regions.setdefault(region_index, []).append(key)

        # print(f"Found: {found}")

        # print(f"Infinite Ridge Regions: {infinite_ridge_regions}")
        # print(f"Reversed Infinite Ridge Regions: {reversed_infinite_ridge_regions}")

        # all_infinite_ridge_regions = set(reversed_infinite_ridge_regions.keys())
        # print(f"All Infinite Ridge Regions: {all_infinite_ridge_regions}")

        # # Handle infinite regions without infinite ridges and finite regions
        # for region_index in range(len(vor.regions)):
        #     if region_index in all_infinite_ridge_regions:
        #         continue

        #     region = vor.regions[region_index]

        #     if len(region) == 0:
        #         continue

        #     # Remove the point at infinity (-1)
        #     region = [r for r in region if r != -1]

        #     vertices = vor.vertices[region]
        #     # vertices = clip_to_bounding_box(vertices, bounding_box)
        #     vertices = clip_vertex_region(vertices, bounding_box)
        #     ax[1].fill(*zip(*vertices), alpha=0.5, edgecolor="black")

        # bounded_ridge_points = get_bounded_ridge_points(vor, bounding_box)
        # for ridge_point_indices in infinite_ridge_point_indices_list:
        #     bounded_ridge_point = bounded_ridge_points.get(tuple(ridge_point_indices))
        #     if bounded_ridge_point is not None:
        #         ax[1].scatter(*bounded_ridge_point, color="cyan")

        # # Get regions corresponding to the infinite ridges indices
        # for region_index in reversed_infinite_ridge_regions.keys():
        #     ridge_point_indices = reversed_infinite_ridge_regions[region_index]
        #     assert len(ridge_point_indices) == 1, "Only one ridge point index"
        #     ridge_point_indices = ridge_point_indices[0]
        #     region = vor.regions[region_index]

        #     # Replace the point at infinity (-1) with the bounded ridge point
        #     # Get the position of the infinite point in the region
        #     infinite_point_index = region.index(-1)
        #     region = [r for r in region if r != -1]
        #     vertices = vor.vertices[region]
        #     # Add the new bounded ridge point
        #     vertices = np.insert(
        #         vertices,
        #         infinite_point_index,
        #         bounded_ridge_points.get(tuple(ridge_point_indices)),
        #         axis=0,
        #     )
        #     vertices = clip_vertex_region(vertices, bounding_box)
        #     ax[1].fill(*zip(*vertices), alpha=0.5, edgecolor="black")

        # ax[1].scatter(*zip(*points), color="red")
        # ax[1].set_title("Clipped Regions")
        # ax[1].set_xlim(bounding_box[0, 0], bounding_box[1, 0])
        # ax[1].set_ylim(bounding_box[0, 1], bounding_box[1, 1])

        # plt.show()

        #    for region in vor.regions:
        #     if len(region) == 0:
        #         continue
        #     vertices = vor.vertices[region]
        #     print(vertices.shape)
        #     # vertices = clip_infinite_regions(vor, bounding_box)
        #     # print(np.array(vertices).shape)
        #     vertices = clip_to_bounding_box(vertices, bounding_box)
        #     print(vertices.shape)
        #     plt.fill(*zip(*vertices), edgecolor="black", alpha=0.5)
        # plt.scatter(*zip(*points), color="red")
        # plt.xlim(bounding_box[0, 0], bounding_box[1, 0])
        # plt.ylim(bounding_box[0, 1], bounding_box[1, 1])
        # plt.title("Voronoi Diagram")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.grid()
        # plt.show()

        for vertices in clip_regions(vor=vor, bounding_box=bounding_box):
            ax[1].fill(*zip(*vertices), alpha=0.5, edgecolor="black")

        bounded_ridge_points = get_bounded_ridge_points(vor, bounding_box)
        for ridge_point_indices in bounded_ridge_points:
            bounded_ridge_point = bounded_ridge_points.get(tuple(ridge_point_indices))
            if bounded_ridge_point is not None:
                ax[1].scatter(*bounded_ridge_point, color="cyan")

        ax[1].scatter(*zip(*points), color="red")
        ax[1].set_title("Clipped Regions")
        ax[1].set_xlim(bounding_box[0, 0], bounding_box[1, 0])
        ax[1].set_ylim(bounding_box[0, 1], bounding_box[1, 1])

        plt.show()

    return exact_volumes, approximate_volumes


# Main Simulation
dimensions_range = range(2, 9)
# dimensions_range = range(2, 4)
# dimensions_range = range(2, 7)
mean_absolute_errors = []

for dimensions in dimensions_range:
    # if dimensions > 4:
    #     break
    print(f"Running simulation for {dimensions} dimensions")
    # exact, approximate = compare_methods(dimensions, n_points=50)
    # exact, approximate = compare_methods(dimensions, n_points=80)
    exact, approximate = compare_methods(dimensions, n_points=1000)
    print(f"Exact volumes: {exact}")
    print(f"Approximate volumes: {approximate}")
    error = np.abs(np.array(exact) - np.array(approximate)).mean()
    mean_absolute_errors.append(error)

# Plot Results
plt.figure(figsize=(8, 6))
plt.plot(
    list(dimensions_range)[: len(mean_absolute_errors)],
    mean_absolute_errors,
    marker="o",
)
plt.title("Mean Absolute Error Between Exact and Approximate Volumes (Clipped)")
plt.xlabel("Dimensions")
plt.ylabel("Mean Absolute Error")
plt.grid()
plt.show()
