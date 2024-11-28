from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt


def kde_marginal_projection(data, kde_bandwidth=0.1, grid_points=100):
    """
    Project the KDE predictions onto each dimension.
    :param data: Numpy array of shape (n_samples, n_features)
    :param kde_bandwidth: Bandwidth for KDE
    :param grid_points: Number of points in the grid for evaluation
    :return: Marginal density predictions for each dimension
    """
    n_features = data.shape[1]
    print(f"n_features: {n_features}")
    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth)
    kde.fit(data)

    projections = {}
    for dim in range(n_features):
        # Create grid along the current dimension
        min_val, max_val = data[:, dim].min(), data[:, dim].max()
        grid = np.linspace(min_val, max_val, grid_points + 1)
        print(f"Grid: {grid}")
        # Evaluate KDE for the data
        log_density = kde.score_samples(data)
        print(f"Log Density: {log_density}")
        density = np.exp(log_density)

        # Check if the sum of density is close to 1
        print(f"Sum of density: {np.sum(density)}")

        # Sum the density along the current dimension
        marginal_density = []
        for i in range(grid_points):
            mask = (data[:, dim] >= grid[i]) & (data[:, dim] < grid[i + 1])
            marginal_density.append(np.sum(density[mask]))

        # Check if the sum of marginal density is close to 1
        print(f"Sum of marginal density: {np.sum(marginal_density)}")

        print(f"Marginal Density: {marginal_density}")

        # Renormalize the marginal density
        # marginal_density = np.array(marginal_density) / np.sum(marginal_density)

        # Multiply by the number of samples
        # marginal_density = np.array(marginal_density) * len(data)  # / grid_points

        marginal_density = np.array(marginal_density) * (
            grid[1] - grid[0]
        )  # / grid_points

        print(f"Sum of marginal density: {np.sum(marginal_density)}")

        projections[dim] = (grid[:-1], np.array(marginal_density), data[:, dim])

        plt.figure()
        plt.plot(grid[:-1], marginal_density, label=f"Dimension {dim + 1}")
        # Plot the data points
        plt.hist(data[:, dim], bins=grid[:-1], alpha=0.5, density=True, label="Data")
        plt.title(f"KDE Projection for Dimension {dim + 1}")
        plt.xlabel(f"Dimension {dim + 1}")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    return projections


def kde_marginal_projection(data, kde_bandwidth=0.1, grid_points=100):
    """
    Project the KDE predictions onto each dimension.
    :param data: Numpy array of shape (n_samples, n_features)
    :param kde_bandwidth: Bandwidth for KDE
    :param grid_points: Number of points in the grid for evaluation
    :return: Marginal density predictions for each dimension
    """
    n_features = data.shape[1]
    print(f"n_features: {n_features}")
    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth)
    kde.fit(data)

    projections = {}
    for dim in range(n_features):
        # Create grid along the current dimension
        min_val, max_val = data[:, dim].min(), data[:, dim].max()
        grid = np.linspace(min_val, max_val, grid_points + 1)
        print(f"Grid: {grid}")
        # Evaluate KDE for the data
        x = kde.sample(n_samples=1000)

        # Sum the density along the current dimension
        marginal_density = []
        for i in range(grid_points):
            mask = (x[:, dim] >= grid[i]) & (x[:, dim] < grid[i + 1])
            marginal_density.append(np.sum(mask))

        # Check if the sum of marginal density is close to 1
        print(f"Sum of marginal density: {np.sum(marginal_density)}")

        print(f"Marginal Density: {marginal_density}")

        # Renormalize the marginal density
        # marginal_density = np.array(marginal_density) / np.sum(marginal_density)

        # Multiply by the number of samples
        # marginal_density = np.array(marginal_density) * len(data)  # / grid_points

        projections[dim] = (grid[:-1], np.array(marginal_density), data[:, dim])

        y = kde.score_samples(grid[:-1])

        # Print the sum of the density
        print(f"Sum of density: {np.sum(np.exp(y))}")

        # y = np.exp(y) / np.sum(np.exp(y))

        plt.figure()
        # plt.plot(grid[:-1], marginal_density, label=f"Dimension {dim + 1}")
        plt.hist(x[:, dim], bins=grid[:-1], alpha=0.5, density=True, label="Sampled")
        plt.plot(grid[:-1], np.exp(y), label="KDE")
        # plt.plot(grid[:-1], y, label="KDE")
        # Plot the data points
        plt.hist(data[:, dim], bins=grid[:-1], alpha=0.5, density=True, label="Data")
        plt.title(f"KDE Projection for Dimension {dim + 1}")
        plt.xlabel(f"Dimension {dim + 1}")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    return projections


def kde_marginal_projection(data, kde_bandwidth=0.1, grid_points=10):
    """
    Compute the chi-square goodness-of-fit test for KDE.
    :param data: Numpy array of shape (n_samples, n_features)
    :param kde_bandwidth: Bandwidth for KDE
    :param bins: Number of bins for the histogram
    :return: Chi-square statistic and p-value
    """
    # Fit KDE
    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth)
    kde.fit(data)

    # Create histogram (observed frequencies)
    hist, edges = np.histogramdd(data, bins=grid_points)
    bin_volumes = np.prod([edges[i][1] - edges[i][0] for i in range(len(edges))])

    # Compute expected frequencies using KDE
    centers = [0.5 * (edges[i][1:] + edges[i][:-1]) for i in range(len(edges))]
    grid = np.meshgrid(*centers)
    grid_points = np.stack([g.ravel() for g in grid], axis=-1)

    log_density = kde.score_samples(grid_points)
    expected = np.exp(log_density) * bin_volumes
    expected = expected.reshape(hist.shape)
    # print(expected[expected != 0])
    # print(np.sum(np.exp(log_density) != 0))
    # print(expected)

    # print(f"Sums: {np.sum(observed)}, {np.sum(expected)}")
    # print(f"Sum of observed: {np.sum(log_density)}")
    # print(f"Sum of observed: {np.sum(np.exp(log_density))*bin_volumes}")
    # print(f"Sum of data: {np.sum(np.exp(kde.score_samples(data)))}")
    # print(f"Sum of data: {np.sum(np.exp(kde.score_samples(data))) * bin_volumes}")
    # print(f"Bin Volumes: {bin_volumes}")
    # print(f"Dim sum: {np.sum(np.sum(expected, axis=1), axis=0)}")
    # print(f"Dim sum: {np.sum(np.sum(expected, axis=0), axis=0)}")

    # Plot for each dimension data and expected
    for dim in range(data.shape[1]):
        sumed_axis = tuple(range(dim)) + tuple(range(dim + 1, data.shape[1]))
        # Swap 0 and 1 axis
        if dim == 0:
            sumed_axis = (0,) + tuple(range(2, data.shape[1]))
        elif dim == 1:
            sumed_axis = (1,) + tuple(range(2, data.shape[1]))
        # print(f"Summed axis: {sumed_axis}")
        # assert not dim in sumed_axis, "Dimension should not be summed over"
        plt.figure()
        plt.hist(data[:, dim], bins=edges[dim], alpha=0.5, density=True, label="Data")
        plt.plot(
            centers[dim],
            np.sum(
                expected,
                axis=sumed_axis,
            )
            / (edges[dim][1] - edges[dim][0]),
            label="Expected",
        )
        plt.title(f"KDE Projection for Dimension {dim + 1}")
        plt.xlabel(f"Dimension {dim + 1}")
        plt.ylabel("Density")
        plt.legend()
        plt.show()


# Data is 2D gaussian
np.random.seed(0)

# data = np.random.multivariate_normal(mean=[2, 0], cov=[[1, 0.5], [0.5, 1]], size=20000)
# data = np.random.multivariate_normal(
#     mean=[2, 0, 0], cov=[[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]], size=2000
# )

# Data is 4D gaussian
data = np.random.multivariate_normal(
    mean=[2, 0, 4, 0],
    cov=[
        [1, 0.5, 0.3, 0.2],
        [0.5, 1, 0.2, 0.1],
        [0.3, 0.2, 1, 0.1],
        [0.2, 0.1, 0.1, 1],
    ],
    size=2000,
)


print(f"Data: {data}")

# Plot the data
plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.title("Data")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

grid_points = 100

# Project the data onto each dimension
for dim in range(data.shape[1]):
    # Create grid along the current dimension
    min_val, max_val = data[:, dim].min(), data[:, dim].max()
    grid = np.linspace(min_val, max_val, grid_points + 1)

    # Sum the density along the current dimension
    marginal_density = []
    for i in range(grid_points):
        mask = (data[:, dim] >= grid[i]) & (data[:, dim] < grid[i + 1])
        marginal_density.append(np.sum(mask))

    plt.figure()
    plt.plot(grid[:-1], marginal_density, label=f"Dimension {dim + 1}")
    # Plot the data points
    plt.hist(data[:, dim], bins=grid[:-1], alpha=0.5, density=False, label="Data")
    plt.title(f"KDE Projection for Dimension {dim + 1}")
    plt.xlabel(f"Dimension {dim + 1}")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# Data is 1D gaussian
# data = np.random.normal(loc=0, scale=1, size=1000).reshape(-1, 1)

kde_marginal_projection(data, kde_bandwidth=0.1, grid_points=30)
