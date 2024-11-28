import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from load_data import match_input_data, load_csv_data
from sklearn.neighbors import KernelDensity

event = 101


# Calculate mutual information between two features
def mutual_information(df, feature1, feature2):
    # Calculate the joint probability distribution
    joint_prob = df.groupby([feature1, feature2]).size() / len(df)
    # Calculate the marginal probability distribution
    marginal_prob1 = df[feature1].value_counts() / len(df)
    marginal_prob2 = df[feature2].value_counts() / len(df)
    # Calculate the mutual information
    mi = 0
    for i, j in joint_prob.index:
        mi += joint_prob[i, j] * np.log(
            joint_prob[i, j] / (marginal_prob1[i] * marginal_prob2[j])
        )
    return mi


# Function to calculate conditional entropy using kernel density estimation
def conditional_entropy(x, y):
    """

    Estimate conditional entropy H(Y|X) using kernel density estimation.

    """
    # Reshape for sklearn
    x_reshaped = x.reshape(-1, 1)
    y_reshaped = y.reshape(-1, 1)

    # Fit kernel density models for joint and marginal distributions
    kde_joint = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(
        np.hstack([x_reshaped, y_reshaped])
    )
    kde_x = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(x_reshaped)

    # Calculate log density estimates
    log_joint_density = kde_joint.score_samples(np.hstack([x_reshaped, y_reshaped]))
    log_x_density = kde_x.score_samples(x_reshaped)

    # Compute conditional entropy
    conditional_entropy_value = -np.mean(log_joint_density - log_x_density)

    return conditional_entropy_value


def entropy(x):
    """

    Estimate entropy H(X) using kernel density estimation.

    """
    # Reshape for sklearn
    x_reshaped = x.reshape(-1, 1)

    # Fit kernel density model for marginal distribution
    kde_x = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(x_reshaped)

    # Calculate log density estimates
    log_x_density = kde_x.score_samples(x_reshaped)

    # Compute entropy
    entropy_value = -np.mean(log_x_density)

    return entropy_value


# Plotting Distribution Function
def plot_feature_distributions(df, scales=[]):
    if scales is None or len(scales) == 0:
        scales = [1] * len(df.columns)
    if len(df.columns) != len(scales):
        raise ValueError(
            f"Length of scales {len(scales)} must be equal to the number of features {len(df.columns)}"
        )
    for i, feature_name in enumerate(df.columns):
        plt.hist(df[feature_name] * scales[i], bins=100, alpha=0.5)
        plt.xlabel(feature_name)
        plt.show()


# Neuron Output Calculation Function
def calculate_neuron_output(df, weights, biases):
    return df.to_numpy() @ weights.T + biases


# Plot Neuron Output Function
def plot_neuron_output(neuron_output, neuron_label):
    plt.hist(neuron_output, bins=100, alpha=0.5, label=f"Neuron {neuron_label}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Scatter Plot for Feature Pairs with Color Gradient
def plot_scatter_with_color(df, feature_x, feature_y, color_data, xlabel, ylabel):
    plt.scatter(df[feature_x], df[feature_y], c=color_data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.show()


# Plot Isocurves Function
def plot_isocurves(df, feature_x, feature_y, color_data):
    plt.tricontourf(df[feature_x], df[feature_y], color_data, levels=20)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.colorbar()
    plt.show()


# Plot Neuron Output Correlation Function
def plot_output_correlation(output1, output2, label1, label2):
    plt.scatter(output1, output2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.show()


# Cartesian and Spherical Coordinates Calculation Function
def add_coordinate_transformations(df):
    df["x"] = df["r"] * np.cos(df["phi"])
    df["y"] = df["r"] * np.sin(df["phi"])
    df["theta"] = np.arctan2(df["r"], df["z"])
    df["rho"] = np.sqrt(df["r"] ** 2 + df["z"] ** 2)
    df["eta"] = -np.log(np.tan(df["theta"] / 2))
    return df


# Multi-feature Plot Against Neuron Output
def plot_neuron_output_vs_features(neuron_output, df, features):
    for feature in features:
        plt.plot(neuron_output, df[feature], "o", label=feature)
    plt.legend()
    plt.grid(True)
    plt.show()


# Load the particles
particles = load_csv_data(
    file_name="event000000101-hard-cut-particles.csv", directory="csv"
)

# Load the truth
truth = load_csv_data(file_name="event000000101-hard-cut-truth.csv", directory="csv")


# Get particles corresponding to the truth
truth_particle_ids = truth["particle_id"].unique()
truth_particles = particles[
    particles["particle_id"].isin(truth_particle_ids)
    | particles["particle_id"].isin(truth["particle_id_1"])
    | particles["particle_id"].isin(truth["particle_id_2"])
]

# Assign particle information to the truth
truth_particles = truth.merge(
    particles, left_on="particle_id", right_on="particle_id", suffixes=("", "_particle")
)

# Compute r, phi, z for the truth particles
truth_particles["r"] = np.sqrt(truth_particles["x"] ** 2 + truth_particles["y"] ** 2)
truth_particles["phi"] = np.arctan2(truth_particles["y"], truth_particles["x"])
truth_particles["z"] = truth_particles["z"]

input_df = match_input_data(truth_particles, load_data=True)

# Add various coordinate transformations
input_df = add_coordinate_transformations(input_df)

# Map particle_id to [0..N]
particle_id_map = {
    particle_id: i for i, particle_id in enumerate(input_df["particle_id"].unique())
}
input_df["particle_id_mapped"] = input_df["particle_id"].map(particle_id_map)


from sklearn.feature_selection import mutual_info_regression

n_hits = input_df.shape[0]

# Read activations
event = 100
event = 101
activations = torch.load(
    f"activations/activations_event{event:09d}.pt", map_location=torch.device("cpu")
)
# print(activations)

for key in activations:
    print(key, activations[key].shape)

keys = list(activations)
# print(keys)
layer_1_name = keys[0]
layer_3_name = keys[2]
layer_4_name = keys[3]

# Transpose the activation to get the neuron distributions
neuron_activations = {
    layer_name: layer_activations.T
    for layer_name, layer_activations in activations.items()
}

activations_1 = neuron_activations[layer_1_name].numpy()
activations_3 = neuron_activations[layer_3_name].numpy()
activations_4 = neuron_activations[layer_4_name].numpy()
# print(activations_3.shape)
# print(neuron_935_weights.T.shape)

# Do input*weights + biases
# neuron_935_output = activations_3.T @ neuron_935_weights.T + neuron_935_biases

neuron_86_output = activations_1[86]
neuron_44_output = activations_1[44]
neuron_935_output = activations_4[935]

# for feature in input_df.columns:
#     feature_values = input_df[feature].to_numpy()
#     # If the feature is not continuous, skip
#     if not np.issubdtype(feature_values.dtype, np.number):
#         print(f"Feature {feature} is not continuous")
#         continue
#     print(f"Feature: {feature}")
#     print(
#         f"Conditional Entropy: {conditional_entropy(neuron_935_output, feature_values)}"
#     )


# Remove non-continuous features
df_continuous = input_df.select_dtypes(include=[np.number])

import os
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np
import time

# Compute mutual information for all neuron outputs in a layer and save them into a CSV file
# mutual_information_df = pd.DataFrame()

# Start from the last layer
reversed_neuron_activations = list(neuron_activations.keys())[::-1]
# for layer_index, layer in enumerate(reversed_neuron_activations):
#     start_time = time.time()
#     layer_index = len(neuron_activations) - layer_index

#     # Get all neuron outputs of the current layer as a 2D array (shape: num_samples x num_neurons)
#     layer_outputs = np.array(
#         neuron_activations[layer]
#     ).T  # Transpose to match mutual_info_regression input format

#     print(f"Computing mutual information for Layer {layer_index}")

#     # File path for the current layer's CSV
#     layer_file_path = f"mutual_information_event{event:09d}_layer{layer_index}.csv"

#     # Initialize or load existing data for the layer
#     if os.path.exists(layer_file_path):
#         mutual_information_df_layer = pd.read_csv(layer_file_path)
#         processed_neurons = mutual_information_df_layer["neuron"].tolist()
#         print(
#             f"Found existing file for Layer {layer_index}. Resuming from Neuron {len(processed_neurons)}"
#         )
#     else:
#         mutual_information_df_layer = pd.DataFrame()
#         processed_neurons = []

#     for neuron_idx in range(layer_outputs.shape[1]):
#         if neuron_idx in processed_neurons:
#             continue  # Skip already processed neurons

#         print(f"Computing mutual information for Neuron {neuron_idx}")

#         mutual_information_values = mutual_info_regression(
#             df_continuous.to_numpy(), layer_outputs[:, neuron_idx], random_state=42
#         )

#         # # Use Adjusted Mutual Information
#         # from sklearn.metrics import adjusted_mutual_info_score

#         # adjusted_mutual_information_values = []
#         # for i, feature in enumerate(df_continuous.columns):
#         #     adjusted_mutual_information_values.append(
#         #         adjusted_mutual_info_score(
#         #             df_continuous[feature].to_numpy(), layer_outputs[:, neuron_idx]
#         #         )
#         #     )

#         # Create a new row for the current neuron
#         new_row = {
#             "layer": layer_index,
#             "neuron": neuron_idx,
#             **{
#                 feature: mutual_information_values[i]
#                 for i, feature in enumerate(df_continuous.columns)
#             },
#         }

#         # Append the new row to the existing DataFrame for the layer
#         mutual_information_df_layer = pd.concat(
#             [mutual_information_df_layer, pd.DataFrame([new_row])],
#             ignore_index=True,
#         )

#         # Save the updated layer data to the CSV file
#         mutual_information_df_layer.to_csv(layer_file_path, index=False)

#     # Append the processed layer to the global DataFrame
#     mutual_information_df = pd.concat(
#         [mutual_information_df, mutual_information_df_layer], ignore_index=True
#     )

#     # Save the combined mutual information to a global CSV file
#     mutual_information_df.to_csv(
#         f"mutual_information_event{event:09d}.csv", index=False
#     )

#     end_time = time.time()
#     print(f"Time taken for Layer {layer_index}: {end_time - start_time} seconds")

# # Final save of the mutual information to a CSV file
# mutual_information_df.to_csv(f"mutual_information_event{event:09d}.csv", index=False)
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy as scipy_entropy
import numpy as np


def entropy_discrete(y):
    """
    Compute entropy for a discrete variable Y.
    :param y: Numpy array of shape (n_samples,), the discrete variable Y
    :return: Estimated entropy
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log(probabilities))


def entropy_discrete(y):
    """
    Compute entropy for a discrete variable Y using scipy.
    :param y: Numpy array of shape (n_samples,), the discrete variable Y
    :return: Estimated entropy
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return scipy_entropy(probabilities)


def entropy_kde(data, bandwidth=0.5):
    """
    Estimate entropy for continuous variables using Kernel Density Estimation (KDE).
    :param data: Numpy array of shape (n_samples, n_features)
    :param bandwidth: Bandwidth for KDE
    :return: Estimated entropy
    """
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(data)

    # Estimate log density
    log_density = kde.score_samples(data)

    # Safeguard: Avoid numerical issues by ensuring log densities are valid
    log_density = np.clip(log_density, -1e10, np.log(np.finfo(float).max))

    # Compute entropy as the negative mean log-density
    entropy = -np.mean(log_density)
    return entropy


def conditional_entropy(data_y, data_x, y_is_discrete=False, bandwidth=0.5):
    """
    Estimate conditional entropy H(Y | X) for mixed discrete-continuous data.
    :param data_y: Numpy array of shape (n_samples, 1), the target variable Y
    :param data_x: Numpy array of shape (n_samples, n_features), the conditioning variables X
    :param y_is_discrete: Whether Y is discrete (True) or continuous (False)
    :param bandwidth: Bandwidth for KDE
    :return: Estimated conditional entropy H(Y | X)
    """
    if y_is_discrete:
        # Compute joint entropy H(Y, X) for discrete-continuous
        joint_entropy_yx = 0
        unique_y, counts_y = np.unique(data_y, return_counts=True, axis=0)
        total_samples = len(data_y)

        for y_val, count in zip(unique_y, counts_y):
            # Extract samples where Y = y_val
            mask = (data_y == y_val).flatten()
            sub_data_x = data_x[mask]

            if len(sub_data_x) > 1:
                joint_entropy_yx += (count / total_samples) * entropy_kde(
                    sub_data_x, bandwidth
                )

        joint_entropy_yx += entropy_discrete(data_y)

    else:
        # Continuous Y: Joint entropy H(Y, X)
        data_yx = np.hstack((data_y, data_x))
        joint_entropy_yx = entropy_kde(data_yx, bandwidth)

    # Entropy H(X)
    h_x = entropy_kde(data_x, bandwidth)

    # Conditional entropy H(Y | X)
    h_y_given_x = joint_entropy_yx - h_x
    return h_y_given_x


def entropy_histogram(data, bins=10):
    """
    Estimate entropy using a histogram-based approach.
    :param data: Numpy array of shape (n_samples, n_features)
    :param bins: Number of bins for the histogram
    :return: Estimated entropy
    """
    # If bins is an integer, take min between bins and 2
    if isinstance(bins, int):
        bins = min(bins, 2)
    hist, edges = np.histogramdd(data, bins=bins, density=True)
    probabilities = hist / np.sum(hist)
    probabilities = probabilities[probabilities > 0]  # Exclude zero probabilities
    return -np.sum(probabilities * np.log(probabilities))


def conditional_entropy_histogram(data_y, data_x, y_is_discrete=False, bins=2):
    """
    Estimate conditional entropy H(Y | X) using a histogram-based approach.
    :param data_y: Numpy array of shape (n_samples, 1), the target variable Y
    :param data_x: Numpy array of shape (n_samples, n_features), the conditioning variables X
    :param y_is_discrete: Whether Y is discrete (True) or continuous (False)
    :param bins: Number of bins for the histogram
    :return: Estimated conditional entropy H(Y | X)
    """
    if y_is_discrete:
        print(f"Bins: {[min(np.unique(data_y).size, 50)] + [bins] * data_x.shape[1]}")
        # Compute joint entropy H(Y, X)
        joint_hist, edges = np.histogramdd(
            np.hstack((data_y, data_x)),
            bins=[min(np.unique(data_y).size, 50)] + [bins] * data_x.shape[1],
            density=True,
        )
        joint_probabilities = joint_hist / np.sum(joint_hist)
        joint_entropy_yx = -np.sum(
            joint_probabilities[joint_probabilities > 0]
            * np.log(joint_probabilities[joint_probabilities > 0])
        )

        # Marginal entropy H(X)
        hist_x, _ = np.histogramdd(data_x, bins=bins, density=True)
        probabilities_x = hist_x / np.sum(hist_x)
        entropy_x = -np.sum(
            probabilities_x[probabilities_x > 0]
            * np.log(probabilities_x[probabilities_x > 0])
        )

        # Marginal entropy H(Y)
        unique, counts = np.unique(data_y, return_counts=True)
        probabilities_y = counts / len(data_y)
        entropy_y = -np.sum(probabilities_y * np.log(probabilities_y))

        # Conditional entropy
        return max(joint_entropy_yx - entropy_x, 0)

    else:
        # Continuous Y: Joint entropy H(Y, X)
        data_yx = np.hstack((data_y, data_x))
        joint_entropy_yx = entropy_histogram(data_yx, bins=bins)

        # Entropy H(X)
        entropy_x = entropy_histogram(data_x, bins=bins)

        # Conditional entropy H(Y | X)
        return max(joint_entropy_yx - entropy_x, 0)


# Example dataset: Continuous variables
np.random.seed(42)

# feature_values = df_continuous["pt"].to_numpy()  # Convert feature column to NumPy array

# # Conditional entropy H(Z | X, Y)
# data_y = feature_values.reshape(-1, 1)  # Dependent variable (Z)
# data_x = df_continuous[["px", "py"]].to_numpy()  # Conditioning variables (X, Y)
# data_x = df_continuous[["x", "z"]].to_numpy()  # Conditioning variables (X, Y)
# conditional_entropy_val = conditional_entropy_knn(data_y, data_x)

# # Compute the entropy of the feature
# data_y_entropy = entropy_knn(data_y)

# print(
#     f"Conditional Entropy H(Z | X, Y): {conditional_entropy_val} for feature  with entropy H(Z): {data_y_entropy}"
# )
# exit()

import numpy as np
import pandas as pd
from itertools import combinations

joint_mutual_information_values = {}

# Assuming `layer_index` is intended to be the last layer
layer_index = len(neuron_activations) - 1
layer = list(neuron_activations)[layer_index]
layer_outputs = np.array(
    neuron_activations[layer]
).T  # Transpose for shape (samples, neurons)

# Remove duplicates [r, phi, z] in feature
df_continuous = df_continuous.drop_duplicates(subset=["r", "phi", "z"])

# # Iterate over all features and neuron pairs
# for feature in df_continuous.columns:
#     # print(
#     #     f"Computing joint mutual information for Neuron pairs with feature '{feature}'"
#     # )
#     feature_values = df_continuous[
#         feature
#     ].to_numpy()  # Convert feature column to NumPy array
#     # Check if the feature is discrete integer
#     y_is_discrete = False
#     if (df_continuous[feature].abs() == df_continuous[feature].abs().astype(int)).all():
#         y_is_discrete = True
#         print(f"Feature '{feature}' is discrete")
#     else:
#         print(f"Feature '{feature}' is continuous")

#     # Conditional entropy H(Z | X, Y)
#     data_y = feature_values.reshape(-1, 1)  # Dependent variable (Z)
#     data_x = layer_outputs  # Conditioning variables (X, Y)
#     # conditional_entropy_val = conditional_entropy_histogram(
#     #     data_y, data_x, y_is_discrete=y_is_discrete
#     # )

#     # # Compute the entropy of the feature
#     # data_y_entropy = (
#     #     entropy_histogram(data_y)
#     #     if not y_is_discrete
#     #     else entropy_discrete(feature_values)
#     # )
#     conditional_entropy_val = conditional_entropy(
#         data_y, data_x, y_is_discrete=y_is_discrete
#     )

#     # Compute the entropy of the feature
#     data_y_entropy = (
#         entropy_kde(data_y) if not y_is_discrete else entropy_discrete(feature_values)
#     )

#     print(
#         f"Conditional Entropy H({feature} | X): {conditional_entropy_val} for feature '{feature}' with entropy H({feature}): {data_y_entropy}"
#     )

# #     for neuron_idx1, neuron_idx2 in combinations(range(layer_outputs.shape[1]), 2):

# #         x1 = layer_outputs[:, neuron_idx1].reshape(
# #             -1, 1
# #         )  # Reshape for sklearn compatibility
# #         x2 = layer_outputs[:, neuron_idx2].reshape(-1, 1)
# #         x1_x2 = np.column_stack((x1, x2))  # Combine neuron outputs into a 2D array

# #         # Compute joint mutual information MI(Y;X1,X2)
# #         mi_x1 = mutual_info_regression(x1, feature_values, random_state=42)[
# #             0
# #         ]  # MI(Y;X1)
# #         mi_x2 = mutual_info_regression(x2, feature_values, random_state=42)[
# #             0
# #         ]  # MI(Y;X2)
# #         mi_x1_x2 = mutual_info_regression(
# #             x1_x2, feature_values, random_state=42
# #         ).sum()  # MI(Y;X1,X2)

# #         joint_mutual_information_values[(neuron_idx1, neuron_idx2)] = (
# #             mi_x1 + mi_x2 - mi_x1_x2
# #         )
# #         print(  # Print the joint mutual information value
# #             f"Neuron pair ({neuron_idx1}, {neuron_idx2}): {joint_mutual_information_values[(neuron_idx1, neuron_idx2)]}"
# #         )

# conditional_entropy_dict = {}

# # Iterate over all features and neuron pairs
# for feature in df_continuous.columns:
#     # print(
#     #     f"Computing joint mutual information for Neuron pairs with feature '{feature}'"
#     # )
#     feature_values = df_continuous[
#         feature
#     ].to_numpy()  # Convert feature column to NumPy array
#     # Check if the feature is discrete integer
#     y_is_discrete = False
#     if (df_continuous[feature].abs() == df_continuous[feature].abs().astype(int)).all():
#         y_is_discrete = True
#         print(f"Feature '{feature}' is discrete")
#     else:
#         print(f"Feature '{feature}' is continuous")

#     # Conditional entropy H(Z | X, Y)
#     data_y = feature_values.reshape(-1, 1)  # Dependent variable (Z)
#     data_x = layer_outputs  # Conditioning variables (X, Y)
#     # conditional_entropy_val = conditional_entropy_histogram(
#     #     data_y, data_x, y_is_discrete=y_is_discrete
#     # )

#     # # Compute the entropy of the feature
#     # data_y_entropy = (
#     #     entropy_histogram(data_y)
#     #     if not y_is_discrete
#     #     else entropy_discrete(feature_values)
#     # )
#     conditional_entropy_val = conditional_entropy(
#         data_y, data_x, y_is_discrete=y_is_discrete
#     )

#     # Compute the entropy of the feature
#     data_y_entropy = (
#         entropy_kde(data_y) if not y_is_discrete else entropy_discrete(feature_values)
#     )

#     print(
#         f"Conditional Entropy H({feature} | X): {conditional_entropy_val} for feature '{feature}' with entropy H({feature}): {data_y_entropy}"
#     )

#     print(f"Information coverage: {1 - conditional_entropy_val / data_y_entropy}")

#     # for neuron_idx1, neuron_idx2 in combinations(range(layer_outputs.shape[1]), 2):

#     #     # x1 = layer_outputs[:, neuron_idx1].reshape(
#     #     #     -1, 1
#     #     # )  # Reshape for sklearn compatibility
#     #     # x2 = layer_outputs[:, neuron_idx2].reshape(-1, 1)
#     #     # x1_x2 = np.column_stack((x1, x2))  # Combine neuron outputs into a 2D array

#     #     conditional_entropy_val_12 = conditional_entropy(
#     #         data_y,
#     #         layer_outputs[:, [neuron_idx1, neuron_idx2]],
#     #         y_is_discrete=y_is_discrete,
#     #     )

#     #     conditional_entropy_dict[(neuron_idx1, neuron_idx2)] = (
#     #         conditional_entropy_val_12
#     #     )

#     #     print(  # Print the joint mutual information value
#     #         f"Neuron pair ({neuron_idx1}, {neuron_idx2}): {conditional_entropy_dict[(neuron_idx1, neuron_idx2)]}; Information coverage: {1 - conditional_entropy_dict[(neuron_idx1, neuron_idx2)] / data_y_entropy}"
#     #     )


def compute_full_conditional_entropy(
    layer_index,
    event,
    particles,
    neuron_activations,
    df_continuous,
    bandwidth=0.5,
    conditional_entropy=conditional_entropy,
    entropy_discrete=entropy_discrete,
    entropy_continuous=entropy_kde,
):
    layer = list(neuron_activations)[layer_index]
    layer_outputs = np.array(
        neuron_activations[layer]
    ).T  # Transpose for shape (samples, neurons)

    conditional_entropy_df = pd.DataFrame()
    information_coverage_df = pd.DataFrame()

    output_dir = Path("conditional_entropy")
    output_dir.mkdir(exist_ok=True)

    file_path_full_conditional_entropy = f"conditional_entropy_event{event:09d}_full_layer{layer_index}_bandwidth{bandwidth}.csv"
    file_path_full_information_coverage = f"information_coverage_event{event:09d}_full_layer{layer_index}_bandwidth{bandwidth}.csv"

    file_path_full_conditional_entropy = output_dir / file_path_full_conditional_entropy
    file_path_full_information_coverage = (
        output_dir / file_path_full_information_coverage
    )

    conditional_entropy_df["layer"] = [layer_index]
    information_coverage_df["layer"] = [layer_index]

    # Load the file if it exists
    if os.path.exists(file_path_full_conditional_entropy):
        conditional_entropy_df = pd.read_csv(file_path_full_conditional_entropy)
        print(
            f"Found existing file for Layer {layer_index}. Resuming from Neuron {len(conditional_entropy_df.columns) - 1}"
        )

    if os.path.exists(file_path_full_information_coverage):
        information_coverage_df = pd.read_csv(file_path_full_information_coverage)

    # Iterate over all features and neuron pairs
    # for feature in df_continuous.columns:
    for feature in particles.columns:
        # Skip the feature if it is not in the continuous DataFrame
        if feature not in df_continuous.columns:
            continue

        # Skip the feature if it is already in the DataFrame
        if feature in conditional_entropy_df.columns:
            continue

        # print(
        #     f"Computing joint mutual information for Neuron pairs with feature '{feature}'"
        # )
        feature_values = df_continuous[
            feature
        ].to_numpy()  # Convert feature column to NumPy array
        # Check if the feature is discrete integer
        y_is_discrete = False
        if (
            df_continuous[feature].abs() == df_continuous[feature].abs().astype(int)
        ).all():
            y_is_discrete = True
            print(f"Feature '{feature}' is discrete")
        else:
            print(f"Feature '{feature}' is continuous")

        # Conditional entropy H(Z | X, Y)
        data_y = feature_values.reshape(-1, 1)  # Dependent variable (Z)
        data_x = layer_outputs  # Conditioning variables (X, Y)

        # Compute the entropy of the feature
        conditional_entropy_val = conditional_entropy(
            data_y, data_x, y_is_discrete=y_is_discrete, bandwidth=bandwidth
        )

        # Compute the entropy of the feature
        data_y_entropy = (
            entropy_continuous(data_y, bandwidth=bandwidth)
            if not y_is_discrete
            else entropy_discrete(feature_values)
        )

        print(
            f"Conditional Entropy H({feature} | X): {conditional_entropy_val} for feature '{feature}' with entropy H({feature}): {data_y_entropy}"
        )

        print(f"Information coverage: {1 - conditional_entropy_val / data_y_entropy}")

        # Create a new column for the current feature
        conditional_entropy_df[feature] = [conditional_entropy_val]
        information_coverage_df[feature] = [
            1 - conditional_entropy_val / data_y_entropy
        ]

        # Save the updated data to the CSV file
        conditional_entropy_df.to_csv(file_path_full_conditional_entropy, index=False)
        information_coverage_df.to_csv(file_path_full_information_coverage, index=False)


layer_index = len(neuron_activations) - 3
compute_full_conditional_entropy(
    layer_index, event, particles, neuron_activations, df_continuous
)


def compute_doublet_conditional_entropy(
    layer_index,
    event,
    particles,
    neuron_activations,
    df_continuous,
    bandwidth=0.5,
    conditional_entropy=conditional_entropy,
    entropy_discrete=entropy_discrete,
    entropy_continuous=entropy_kde,
):
    layer = list(neuron_activations)[layer_index]
    layer_outputs = np.array(
        neuron_activations[layer]
    ).T  # Transpose for shape (samples, neurons)

    conditional_entropy_df = pd.DataFrame()
    information_coverage_df = pd.DataFrame()

    output_dir = Path("conditional_entropy")
    output_dir.mkdir(exist_ok=True)

    file_path_full_conditional_entropy = f"conditional_entropy_event{event:09d}_comb2_layer{layer_index}_bandwidth{bandwidth}.csv"
    file_path_full_information_coverage = f"information_coverage_event{event:09d}_comb2_layer{layer_index}_bandwidth{bandwidth}.csv"

    file_path_full_conditional_entropy = output_dir / file_path_full_conditional_entropy
    file_path_full_information_coverage = (
        output_dir / file_path_full_information_coverage
    )

    # Load the file if it exists
    if os.path.exists(file_path_full_conditional_entropy):
        conditional_entropy_df = pd.read_csv(file_path_full_conditional_entropy)
        print(
            f"Found existing file for Layer {layer_index}. Resuming from Neuron {len(conditional_entropy_df.columns) - 1}"
        )

    if os.path.exists(file_path_full_information_coverage):
        information_coverage_df = pd.read_csv(file_path_full_information_coverage)

    # Compute entropy of each feature and store it
    feature_entropy = {}
    for feature in particles.columns:
        # Skip the feature if it is not in the continuous DataFrame
        if feature not in df_continuous.columns:
            continue

        # Compute the entropy of the feature
        feature_values = df_continuous[
            feature
        ].to_numpy()  # Convert feature column to NumPy array
        # Check if the feature is discrete integer
        y_is_discrete = (
            df_continuous[feature].abs() == df_continuous[feature].abs().astype(int)
        ).all()

        data_y = feature_values.reshape(-1, 1)

        # Compute the entropy of the feature
        data_y_entropy = (
            entropy_continuous(data_y, bandwidth=bandwidth)
            if not y_is_discrete
            else entropy_discrete(feature_values)
        )

        feature_entropy[feature] = data_y_entropy

        print(
            f"Feature {feature} has entropy H({feature}): {data_y_entropy} "
            + ("(discrete)" if y_is_discrete else "(continuous)")
        )

    # Iterate over all features and neuron pairs
    for neuron_idx1, neuron_idx2 in combinations(range(layer_outputs.shape[1]), 2):
        new_row_conditional_entropy = {
            "layer": layer_index,
            "neuron1": neuron_idx1,
            "neuron2": neuron_idx2,
        }

        new_row_information_coverage = {
            "layer": layer_index,
            "neuron1": neuron_idx1,
            "neuron2": neuron_idx2,
        }

        # Skip the neuron pair if it is already in the DataFrame and has all columns filled
        if not conditional_entropy_df.empty and (
            conditional_entropy_df[
                (conditional_entropy_df["neuron1"] == neuron_idx1)
                & (conditional_entropy_df["neuron2"] == neuron_idx2)
                & (conditional_entropy_df["layer"] == layer_index)
            ].shape[0]
            > 0
        ):
            continue

        # Append new row for the combination of neurons
        conditional_entropy_df = pd.concat(
            [conditional_entropy_df, pd.DataFrame([new_row_conditional_entropy])],
            ignore_index=True,
        )

        information_coverage_df = pd.concat(
            [information_coverage_df, pd.DataFrame([new_row_information_coverage])],
            ignore_index=True,
        )

        # for feature in df_continuous.columns:
        for feature in particles.columns:
            # Skip the feature if it is not in the continuous DataFrame
            if feature not in df_continuous.columns:
                continue

            feature_values = df_continuous[
                feature
            ].to_numpy()  # Convert feature column to NumPy array
            # Check if the feature is discrete integer
            y_is_discrete = False
            if (
                df_continuous[feature].abs() == df_continuous[feature].abs().astype(int)
            ).all():
                y_is_discrete = True

            # Conditional entropy H(Z | X, Y)
            data_y = feature_values.reshape(-1, 1)  # Dependent variable (Z)
            data_x = layer_outputs  # Conditioning variables (X, Y)

            # Compute the entropy of the feature
            data_y_entropy = feature_entropy[feature]

            # Compute the entropy of the feature
            conditional_entropy_val_12 = conditional_entropy(
                data_y,
                layer_outputs[:, [neuron_idx1, neuron_idx2]],
                y_is_discrete=y_is_discrete,
                bandwidth=bandwidth,
            )

            information_coverage_val_12 = (
                1 - conditional_entropy_val_12 / data_y_entropy
            )

            # Set the value of the feature for this combination of neurons
            conditional_entropy_df.loc[
                (conditional_entropy_df["neuron1"] == neuron_idx1)
                & (conditional_entropy_df["neuron2"] == neuron_idx2)
                & (conditional_entropy_df["layer"] == layer_index),
                feature,
            ] = conditional_entropy_val_12

            information_coverage_df.loc[
                (information_coverage_df["neuron1"] == neuron_idx1)
                & (information_coverage_df["neuron2"] == neuron_idx2)
                & (information_coverage_df["layer"] == layer_index),
                feature,
            ] = information_coverage_val_12

            print(
                f"Neuron pair ({neuron_idx1}, {neuron_idx2}): H({feature}|{neuron_idx1}, {neuron_idx2})/H({feature}): {conditional_entropy_val_12}/{data_y_entropy}; Information coverage: {information_coverage_val_12 * 100} %"
            )

            # Save the updated data to the CSV file
            conditional_entropy_df.to_csv(
                file_path_full_conditional_entropy, index=False
            )
            information_coverage_df.to_csv(
                file_path_full_information_coverage, index=False
            )


layer_index = len(neuron_activations) - 1
# compute_doublet_conditional_entropy(
#     layer_index, event, particles, neuron_activations, df_continuous
# )

# layer_index = len(neuron_activations) - 2
# bandwidth = 8
# compute_full_conditional_entropy(
#     layer_index, event, particles, neuron_activations, df_continuous, bandwidth=bandwidth
# )

exit()
# for neuron_idx1, neuron_idx2 in combinations(range(layer_outputs.shape[1]), 2):

#     # x1 = layer_outputs[:, neuron_idx1].reshape(
#     #     -1, 1
#     # )  # Reshape for sklearn compatibility
#     # x2 = layer_outputs[:, neuron_idx2].reshape(-1, 1)
#     # x1_x2 = np.column_stack((x1, x2))  # Combine neuron outputs into a 2D array

#     conditional_entropy_val_12 = conditional_entropy(
#         data_y,
#         layer_outputs[:, [neuron_idx1, neuron_idx2]],
#         y_is_discrete=y_is_discrete,
#     )

#     conditional_entropy_dict[(neuron_idx1, neuron_idx2)] = (
#         conditional_entropy_val_12
#     )

#     print(  # Print the joint mutual information value
#         f"Neuron pair ({neuron_idx1}, {neuron_idx2}): {conditional_entropy_dict[(neuron_idx1, neuron_idx2)]}; Information coverage: {1 - conditional_entropy_dict[(neuron_idx1, neuron_idx2)] / data_y_entropy}"
#     )


# Plot kde fit for the features
for feature in df_continuous.columns:
    feature_values = df_continuous[feature].to_numpy()
    # If the feature is not continuous, skip
    if not np.issubdtype(feature_values.dtype, np.number):
        print(f"Feature {feature} is not continuous")
        continue

    if (
        not feature
        in [
            # "norm_x_1",
            # "norm_y_1",
            # "norm_z_1",
            "eta_angle",
            "phi_angle",
            "z",
            "cluster_z",
            "vz",
            "px",
            "py",
            "pz",
            "pt",
            "eta",
            "phi",
            "theta",
        ]
        and False
    ):
        continue

    bandwidth = 0.5
    bandwidth = 0.1
    bandwidth = 1
    bandwidth = 2
    bandwidth = 4
    bandwidth = 8
    # Compute the entropy of the feature
    data_y = feature_values.reshape(-1, 1)
    data_y_entropy = entropy_kde(data_y, bandwidth=bandwidth)
    # Compare with discrete entropy if the feature is discrete
    if (df_continuous[feature].abs() == df_continuous[feature].abs().astype(int)).all():
        data_y_entropy_discrete = entropy_discrete(feature_values)
        print(
            f"Entropy of feature '{feature}': {data_y_entropy} (continuous) vs {data_y_entropy_discrete} (discrete)"
        )
    else:
        print(f"Entropy of feature '{feature}': {data_y_entropy}")

    # Plot the KDE fit for the feature
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(data_y)
    x = np.linspace(min(feature_values), max(feature_values), 1000).reshape(-1, 1)
    log_density = kde.score_samples(x)
    density = np.exp(log_density)
    plt.plot(x, density, label=feature)
    # Compare with the histogram
    plt.hist(feature_values, bins=1000, density=True, alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # Discrete features are not modeled well by KDE
    # Normed coordinates are not modeled well by KDE
    # eta_angle is not modeled well by KDE
    # phi_angle is not modeled well by KDE
    # z, cluster_z and vz are not modeled well by KDE
    # px, py, pz are not modeled well by KDE
    # pt could be modeled better by KDE
    # eta and phi are okay
    # theta is not modeled well by KDE (better with a smaller bandwidth)

# Plot the KDE fit for (z, r)
kde = KernelDensity(kernel="gaussian", bandwidth=0.5)
kde.fit(df_continuous[["z", "r"]].to_numpy())
x = np.linspace(min(df_continuous["z"]), max(df_continuous["z"]), 1000)
y = np.linspace(min(df_continuous["r"]), max(df_continuous["r"]), 1000)
X, Y = np.meshgrid(x, y)
positions = np.vstack([X.ravel(), Y.ravel()]).T
log_density = kde.score_samples(positions)
density = np.exp(log_density)
plt.contourf(X, Y, density.reshape(X.shape))
# Compare with the histogram
plt.hist2d(df_continuous["z"], df_continuous["r"], bins=1000, density=True)
plt.xlabel("z")
plt.ylabel("r")
# Show color bar
plt.colorbar()
plt.show()


# # Save the joint mutual information values to a CSV file
# joint_mutual_information_df = pd.DataFrame(
#     [
#         {
#             "neuron1": neuron_idx1,
#             "neuron2": neuron_idx2,
#             "joint_mutual_information": value,
#         }
#         for (neuron_idx1, neuron_idx2), value in joint_mutual_information_values.items()
#     ]
# )
exit()
output_filename = f"joint_mutual_information_event{event:09d}.csv"
joint_mutual_information_df.to_csv(output_filename, index=False)
print(f"Joint mutual information values saved to '{output_filename}'")


# # Print the feature names and their mutual information with neuron 935
# for i, feature in enumerate(df_continuous.columns):
#     conditional_entropy_value = conditional_entropy(
#         neuron_935_output, df_continuous[feature].to_numpy()
#     )
#     # Compute the entropy of the feature
#     # compute density probability pk of feature
#     # Use kernel density estimation
#     pk = 1
#     entropy_value = -np.sum(pk * np.log(pk))
#     # Compute entropy with sklearn
#     print(
#         f"Feature: {feature}, Mutual Information: {mutual_information_values[i]}, Conditional Entropy: {conditional_entropy_value}, Sum: {mutual_information_values[i] + conditional_entropy_value}, Entropy: {entropy_value} or {entropy(df_continuous[feature].to_numpy())}"
#     )
exit()

# # Plot r vs z with the color of the points representing the output of neuron 86
# plt.scatter(input_df["z"], input_df["r"], c=neuron_86_output)
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar()
# plt.savefig(f"neuron_86_output_event{event:09d}_r_z.png")
# plt.show()

# # Plot isocurves of the output of neuron 86 with a lot of points
# plt.tricontourf(input_df["z"], input_df["r"], neuron_86_output, levels=20)
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar()
# plt.savefig(f"neuron_86_output_event{event:09d}_r_z_isocurves.png")
# plt.show()

# # Plot the output of neuron 44
# plt.hist(
#     neuron_44_output,
#     bins=100,
#     alpha=0.5,
#     label="44",
# )
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot r vs z with the color of the points representing the output of neuron 44
# plt.scatter(input_df["z"], input_df["r"], c=neuron_44_output)
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar()
# plt.savefig(f"neuron_44_output_event{event:09d}_r_z.png")
# plt.show()

# # Plot isocurves of the output of neuron 44 with a lot of points
# plt.tricontourf(input_df["z"], input_df["r"], neuron_44_output, levels=20)
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar()
# plt.savefig(f"neuron_44_output_event{event:09d}_r_z_isocurves.png")
# plt.show()

# # Plot the output of neuron 44 vs the output of neuron 86
# plt.scatter(neuron_86_output, neuron_44_output)
# plt.xlabel("Neuron 86")
# plt.ylabel("Neuron 44")
# plt.savefig("neuron_44_output_vs_neuron_86_output.png")
# plt.show()

# # Plot the output of neuron 935
# plt.hist(
#     neuron_935_output,
#     bins=100,
#     alpha=0.5,
#     label="935",
# )
# plt.legend()
# plt.grid(True)
# plt.show()


# # Plot r vs z with the color of the points representing the output of neuron 935
# plt.scatter(input_df["z"], input_df["r"], c=neuron_935_output)
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar()
# plt.savefig(f"neuron_935_output_event{event:09d}_r_z.png")
# plt.show()

# # Plot isocurves of the output of neuron 935 with a lot of points
# plt.tricontourf(input_df["z"], input_df["r"], neuron_935_output, levels=20)
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar()
# plt.savefig(f"neuron_935_output_event{event:09d}_r_z_isocurves.png")
# plt.show()

# # Plot isocurves of the output of neuron 935 with a lot of points
# plt.tricontourf(input_df["z"], input_df["r"], neuron_935_output, levels=20)
# plt.scatter(input_df["z"], input_df["r"], c=neuron_935_output)
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar()
# plt.show()

# Plot the output of neuron 935 vs the output of neuron 86
# plot_output_correlation(neuron_86_output, neuron_935_output, "Neuron 86", "Neuron 935")

# Plot the output of neuron 935 vs the output of neuron 44
# plot_output_correlation(neuron_44_output, neuron_935_output, "Neuron 44", "Neuron 935")

# Plot neuron 935 output vs x, y, z, r, phi, theta, rho
# plot_neuron_output_vs_features(
#     neuron_935_output, input_df, ["x", "y", "z", "r", "phi", "theta", "rho", "eta"]
# )


# For each neuron of the last layer
# last_layer = len(neuron_activations)
# for neuron_idx in range(neuron_activations[keys[last_layer - 1]].shape[1]):
#     activations = neuron_activations[keys[last_layer - 1]][neuron_idx]
#     print(activations.shape)
#     # Clustering
#     for feature in input_df.columns:
#         plt.scatter(activations, input_df[feature], c=input_df["particle_id_mapped"])
#         plt.ylabel(feature)
#         plt.xlabel(f"Neuron {neuron_idx}")
#         plt.title(f"Neuron {neuron_idx} vs {feature}")
#         plt.show()

# exit()

# Clustering
# for feature in input_df.columns:
#     plt.scatter(neuron_935_output, input_df[feature], c=input_df["particle_id_mapped"])
#     plt.ylabel(feature)
#     plt.xlabel("Neuron 935")
#     plt.show()


# plot_scatter_with_color(input_df, "eta", "pt", neuron_935_output, "eta", "pt")
# exit()

# plot_scatter_with_color(input_df, "x", "y", neuron_935_output, "x", "y")
# plot_scatter_with_color(input_df, "r", "phi", neuron_935_output, "r", "phi")
# plot_scatter_with_color(input_df, "theta", "rho", neuron_935_output, "theta", "rho")
# plot_scatter_with_color(input_df, "eta", "phi", neuron_935_output, "eta", "phi")
# plot_scatter_with_color(input_df, "eta", "r", neuron_935_output, "eta", "r")


# # Plot correlation between neuron 935 output and r
# plot_output_correlation(neuron_935_output, input_df["r"], "Neuron 935", "r")
# plot_output_correlation(neuron_935_output, input_df["eta"], "Neuron 935", "eta")
# plot_output_correlation(neuron_935_output, input_df["rho"], "Neuron 935", "rho")
# plot_output_correlation(neuron_935_output, input_df["theta"], "Neuron 935", "theta")
# plot_output_correlation(neuron_935_output, input_df["phi"], "Neuron 935", "phi")

# # Same for neuron 895 in layer 4
# neuron_895_output = activations_4[895]

# # Plot the output of neuron 895
# plt.hist(
#     neuron_895_output,
#     bins=100,
#     alpha=0.5,
#     label="895",
# )
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot r vs z with the color of the points representing the output of neuron 895
# plt.scatter(input_df["z"], input_df["r"], c=neuron_895_output)
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar()
# plt.savefig(f"neuron_895_output_event{event:09d}_r_z.png")
# plt.show()

# # Plot isocurves of the output of neuron 895 with a lot of points
# plt.tricontourf(input_df["z"], input_df["r"], neuron_895_output, levels=20)
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar()
# plt.savefig(f"neuron_895_output_event{event:09d}_r_z_isocurves.png")
# plt.show()

# # Plot the output of neuron 895 vs the output of neuron 935
# plot_output_correlation(
#     neuron_935_output, neuron_895_output, "Neuron 935", "Neuron 895"
# )

# # Plot the output of neuron 895 vs the output of neuron 86
# plot_output_correlation(neuron_86_output, neuron_895_output, "Neuron 86", "Neuron 895")

# # Plot the output of neuron 895 vs the output of neuron 44
# plot_output_correlation(neuron_44_output, neuron_895_output, "Neuron 44", "Neuron 895")


# Plot isocurves of the output of each neuron of layer 4 one by one
layer = 4

layer_name = keys[layer - 1]
# isocurves_dir_layer = Path(f"isocurves_layer_{layer}")
# isocurves_dir_layer.mkdir(exist_ok=True)

# for i in range(len(neuron_weights_4)):
#     neuron_weights = neuron_weights_4[i]
#     neuron_biases = neuron_biases_4[i]

#     # Do input*weights + biases
#     neuron_output = activations_3.T @ neuron_weights.T + neuron_biases

#     # Plot isocurves of the output of neuron i with a lot of points
#     plt.tricontourf(input_df["z"], input_df["r"], neuron_output, levels=20)
#     plt.xlabel("z")
#     plt.ylabel("r")
#     plt.colorbar()
#     plt.title(f"Layer {layer} Neuron {i} Output Isocurves")
#     plt.savefig(
#         isocurves_dir_layer / f"neuron_{i}_output_event{event:09d}_r_z_isocurves.png"
#     )
#     plt.close()

# # Do the same but with subplots for z-r, r-phi and z-phi
# isocurves_dir_layer_subplots = Path(f"isocurves_layer_{layer}_subplots")
# isocurves_dir_layer_subplots.mkdir(exist_ok=True)

# for i in range(len(neuron_weights_4)):
#     neuron_weights = neuron_weights_4[i]
#     neuron_biases = neuron_biases_4[i]

#     # Do input*weights + biases
#     neuron_output = activations_3.T @ neuron_weights.T + neuron_biases

#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     # Plot isocurves of the output of neuron i with a lot of points
#     axs[0].tricontourf(input_df["z"], input_df["r"], neuron_output, levels=20)
#     axs[0].set_xlabel("z")
#     axs[0].set_ylabel("r")
#     axs[0].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

#     axs[1].tricontourf(input_df["r"], input_df["phi"], neuron_output, levels=20)
#     axs[1].set_xlabel("r")
#     axs[1].set_ylabel("phi")
#     axs[1].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

#     axs[2].tricontourf(input_df["z"], input_df["phi"], neuron_output, levels=20)
#     axs[2].set_xlabel("z")
#     axs[2].set_ylabel("phi")
#     axs[2].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

#     plt.savefig(
#         isocurves_dir_layer_subplots
#         / f"neuron_{i}_output_event{event:09d}_r_z_phi_isocurves.png"
#     )
#     plt.close()

# # Do the same without the isocurves
# no_iso_dir_layer_subplots = Path(f"no_iso_layer_{layer}_subplots")
# no_iso_dir_layer_subplots.mkdir(exist_ok=True)

# for i in range(len(neuron_weights_4)):
#     neuron_weights = neuron_weights_4[i]
#     neuron_biases = neuron_biases_4[i]

#     # Do input*weights + biases
#     neuron_output = activations_3.T @ neuron_weights.T + neuron_biases

#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     # Plot isocurves of the output of neuron i with a lot of points
#     axs[0].scatter(input_df["z"], input_df["r"], c=neuron_output)
#     axs[0].set_xlabel("z")
#     axs[0].set_ylabel("r")
#     axs[0].set_title(f"Layer {layer} Neuron {i} Output")

#     axs[1].scatter(input_df["r"], input_df["phi"], c=neuron_output)
#     axs[1].set_xlabel("r")
#     axs[1].set_ylabel("phi")
#     axs[1].set_title(f"Layer {layer} Neuron {i} Output")

#     axs[2].scatter(input_df["z"], input_df["phi"], c=neuron_output)
#     axs[2].set_xlabel("z")
#     axs[2].set_ylabel("phi")
#     axs[2].set_title(f"Layer {layer} Neuron {i} Output")

#     plt.savefig(
#         no_iso_dir_layer_subplots / f"neuron_{i}_output_event{event:09d}_r_z_phi.png"
#     )
#     plt.close()

# Do the same for layer 13
layer = 9

layer_name = keys[layer - 1]
# layer_name_activations = keys[layer - 2]


def get_layer_parameters(state_dict, state_dict_keys, layer):
    weights_key = state_dict_keys[(layer // 3) * 2]
    biases_key = state_dict_keys[(layer // 3) * 2 + 1]

    assert weights_key.split(".")[1] == str(
        layer - 1
    ), f"weights_key: {weights_key} layer: {layer}"
    assert biases_key.split(".")[1] == str(
        layer - 1
    ), f"biases_key: {biases_key} layer: {layer}"

    neurons_weights = state_dict[weights_key].numpy()
    neurons_biases = state_dict[biases_key].numpy()
    return neurons_weights, neurons_biases


# neurons_weights, neurons_biases = get_layer_parameters(
#     state_dict, state_dict_keys, layer
# )

# activations = neuron_activations[layer_name_activations].numpy()
activations = neuron_activations[layer_name].numpy()

# Do the same without the isocurves
no_iso_dir_layer_subplots = Path(f"no_iso_layer_{layer}_subplots")
no_iso_dir_layer_subplots.mkdir(exist_ok=True)

for i in range(len(activations)):
    # neuron_weights = neurons_weights[i]
    # neuron_biases = neurons_biases[i]

    # Do input*weights + biases
    neuron_output = activations[i]

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    # Plot isocurves of the output of neuron i with a lot of points
    axs[0].scatter(input_df["z"], input_df["r"], c=neuron_output)
    axs[0].set_xlabel("z")
    axs[0].set_ylabel("r")
    axs[0].set_title(f"Layer {layer} Neuron {i} Output")

    axs[1].scatter(input_df["x"], input_df["y"], c=neuron_output)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title(f"Layer {layer} Neuron {i} Output")
    # Make the aspect ratio equal
    axs[1].set_aspect("equal")

    axs[2].scatter(input_df["r"], input_df["phi"], c=neuron_output)
    axs[2].set_xlabel("r")
    axs[2].set_ylabel("phi")
    axs[2].set_title(f"Layer {layer} Neuron {i} Output")

    axs[3].scatter(input_df["z"], input_df["phi"], c=neuron_output)
    axs[3].set_xlabel("z")
    axs[3].set_ylabel("phi")
    axs[3].set_title(f"Layer {layer} Neuron {i} Output")

    plt.savefig(
        no_iso_dir_layer_subplots / f"neuron_{i}_output_event{event:09d}_r_z_phi.png"
    )
    plt.close()

# Do the same with isocurves
isocurves_dir_layer_subplots = Path(f"isocurves_layer_{layer}_subplots")
isocurves_dir_layer_subplots.mkdir(exist_ok=True)

for i in range(len(activations)):
    # neuron_weights = neurons_weights[i]
    # neuron_biases = neurons_biases[i]

    # Do input*weights + biases
    neuron_output = activations[i]

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    # Plot isocurves of the output of neuron i with a lot of points
    axs[0].tricontourf(input_df["z"], input_df["r"], neuron_output, levels=20)
    axs[0].set_xlabel("z")
    axs[0].set_ylabel("r")
    axs[0].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    axs[1].tricontourf(input_df["x"], input_df["y"], neuron_output, levels=20)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    axs[2].tricontourf(input_df["r"], input_df["phi"], neuron_output, levels=20)
    axs[2].set_xlabel("r")
    axs[2].set_ylabel("phi")
    axs[2].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    axs[3].tricontourf(input_df["z"], input_df["phi"], neuron_output, levels=20)
    axs[3].set_xlabel("z")
    axs[3].set_ylabel("phi")
    axs[3].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    plt.savefig(
        isocurves_dir_layer_subplots
        / f"neuron_{i}_output_event{event:09d}_r_z_phi_isocurves.png"
    )
    plt.close()

isocurves_dir_layer_subplots = Path(f"isocurves_layer_{layer}_subplots_eta_rho")
isocurves_dir_layer_subplots.mkdir(exist_ok=True)

for i in range(len(activations)):
    # neuron_weights = neurons_weights[i]
    # neuron_biases = neurons_biases[i]

    # Do input*weights + biases
    neuron_output = activations[i]

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    # Plot isocurves of the output of neuron i with a lot of points
    axs[0].tricontourf(input_df["z"], input_df["r"], neuron_output, levels=20)
    axs[0].set_xlabel("z")
    axs[0].set_ylabel("r")
    axs[0].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    axs[1].tricontourf(input_df["x"], input_df["y"], neuron_output, levels=20)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    axs[2].tricontourf(input_df["eta"], input_df["rho"], neuron_output, levels=20)
    axs[2].set_xlabel("eta")
    axs[2].set_ylabel("rho")
    axs[2].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    axs[3].tricontourf(input_df["eta"], input_df["z"], neuron_output, levels=20)
    axs[3].set_xlabel("eta")
    axs[3].set_ylabel("z")
    axs[3].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    plt.savefig(
        isocurves_dir_layer_subplots
        / f"neuron_{i}_output_event{event:09d}_r_z_phi_isocurves.png"
    )
    plt.close()
