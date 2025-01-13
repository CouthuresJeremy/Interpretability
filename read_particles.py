import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from pathlib import Path
from sklearn.neighbors import KernelDensity
from sklearn.feature_selection import mutual_info_regression
from load_data import (
    match_input_data,
    load_csv_data,
    load_event_data,
    load_event_activations,
    load_model,
)

event = 101


def get_layer_parameters(state_dict, layer):
    state_dict_keys = list(state_dict.keys())
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


def drop_wrong_activation_assignement(
    input_df, duplicated_activations_1, verbose=False
):
    from load_data import scale_data

    # Load the weights and biases for the first layer
    model_state_dict = load_model()

    layer = 1

    neurons_weights, neurons_biases = get_layer_parameters(model_state_dict, layer)

    # Calculate the neuron's output for each hit
    hit_coordinates = input_df[["r", "phi", "z"]]

    # Scale the hit coordinates
    hit_coordinates = scale_data(hit_coordinates, scales=[1 / 1000, 1 / 3.14, 1 / 1000])
    # Change data type to float32
    hit_coordinates["r"] = hit_coordinates["r"].astype("float32")
    hit_coordinates["phi"] = hit_coordinates["phi"].astype("float32")
    hit_coordinates["z"] = hit_coordinates["z"].astype("float32")

    # Calculate the neuron outputs
    neuron_outputs = calculate_neuron_output(
        hit_coordinates, neurons_weights, neurons_biases
    )
    dropped_indices = []
    print(f"{duplicated_activations_1.shape = }")
    print(f"{neuron_outputs.shape = }")
    # Compare the neuron outputs with the activations
    for i in range(neuron_outputs.shape[1]):
        if verbose:
            print(f"Neuron {i}")
        # # Check if the activation value can be found in the neuron outputs
        # for j in range(neuron_outputs.shape[0] - 1, -1, -1):
        #     if j in dropped_indices:
        #         continue
        #     same_index = np.where(
        #         neuron_outputs[j, i] == duplicated_activations_1[i, :]
        #     )
        #     if len(same_index[0]) == 0 or not j in same_index[0]:
        #         # Remove the row from the duplicated_activations_1
        #         print(f"{duplicated_activations_1.shape = } {j = }")
        #         duplicated_activations_1 = np.delete(
        #             duplicated_activations_1, j, axis=1
        #         )
        #         # Remove the row from the input_df
        #         input_df = input_df.drop(j)
        #         dropped_indices.append(j)

        # Get all the indices where the neuron output is not the same as the activation
        # same_index = np.where(neuron_outputs[:, i] == duplicated_activations_1[i, :])
        dropping_indices = np.where(
            neuron_outputs[:, i] != duplicated_activations_1[i, :]
        )
        if verbose:
            print(f"{dropping_indices = }")
        dropping_indices_clean = []
        for dropping_index in list(dropping_indices[0]):
            if dropping_index not in dropped_indices:
                dropping_indices_clean.append(dropping_index)
                if verbose:
                    print(f"Dropping {dropping_index}")

        dropped_indices.extend(dropping_indices_clean)
        if dropping_indices_clean:
            # Remove the row from the duplicated_activations_1
            duplicated_activations_1 = np.delete(
                duplicated_activations_1, dropping_indices_clean, axis=1
            )
            # Remove the row from the input_df
            input_df = input_df.drop(dropping_indices_clean)

            # Remove the row from the neuron_outputs
            neuron_outputs = np.delete(neuron_outputs, dropping_indices_clean, axis=0)
        # print(f"{duplicated_activations_1.shape = } {i = }")
        # print(f"{input_df.shape = } {i = }")

    return input_df, duplicated_activations_1, dropped_indices


def check_2(event_id):
    # Reload to make sure the data is saved correctly
    activations = torch.load(
        f"activations/activations_event{event_id:09d}.pt",
        map_location=torch.device("cpu"),
    )

    # # Compute manually the activations for the first layer and compare with the saved activations
    # first_layer = self.network[0]
    # first_layer_name = f"layer_0_{first_layer.__class__.__name__}"
    # Load data
    input_df = load_csv_data(
        file_name=f"input_data_event{event_id:09d}.csv", directory="input_data"
    )
    # Change to correct device
    input_tensor = torch.tensor(input_df.values).float()
    # first_layer_activations = first_layer(input_tensor)
    # # Compare exact values
    # print(
    #     f"Comparing first layer activations: {torch.allclose(first_layer_activations, activations[first_layer_name], atol=0) = }"
    # )
    # assert torch.allclose(
    #     first_layer_activations, activations[first_layer_name], atol=0
    # )
    first_layer_name = list(activations)[0]
    print(f"{first_layer_name = }")

    # Compare with cpu computation
    cpu_first_layer = activations[first_layer_name].cpu()
    # Get the weights and biases of the model
    model_path = "model/best--f1=0.313180-epoch=89.ckpt"
    model = torch.load(model_path, map_location=torch.device("cpu"))

    model_keys = list(model["state_dict"])
    print(model_keys)
    weights, biases = (
        model["state_dict"][model_keys[0]],
        model["state_dict"][model_keys[1]],
    )
    print(f"{weights.shape = } {biases.shape = }")
    # Calculate the neuron outputs
    input_tensor = input_tensor.cpu()
    weights = weights.cpu()
    biases = biases.cpu()
    activations_computed = torch.matmul(input_tensor, weights.T) + biases
    # Compare the activations
    print(
        f"Comparing first layer activations: {torch.allclose(activations_computed, cpu_first_layer, atol=0) = }"
    )
    assert torch.allclose(activations_computed, cpu_first_layer, atol=0)

    # Compare with numpy computation
    cpu_first_layer_np = cpu_first_layer.numpy()
    input_tensor_np = input_tensor.numpy()
    weights_np = weights.numpy()
    biases_np = biases.numpy()
    import numpy as np

    activations_computed_np = np.matmul(input_tensor_np, weights_np.T) + biases_np
    print(
        f"Comparing first layer activations: {np.allclose(activations_computed_np, cpu_first_layer_np, atol=0) = }"
    )
    assert np.allclose(activations_computed_np, cpu_first_layer_np, atol=0)
    assert (activations_computed_np == cpu_first_layer_np).all()


def verify_activation_assignement(input_df, duplicated_activations_1, verbose=False):
    from load_data import scale_data

    # Load the weights and biases for the first layer
    model_state_dict = load_model()

    layer = 1

    neurons_weights, neurons_biases = get_layer_parameters(model_state_dict, layer)

    # Calculate the neuron's output for each hit
    hit_coordinates = input_df[["r", "phi", "z"]]

    # Scale the hit coordinates
    hit_coordinates = scale_data(hit_coordinates, scales=[1 / 1000, 1 / 3.14, 1 / 1000])
    # Change data type to float32
    hit_coordinates["r"] = hit_coordinates["r"].astype("float32")
    hit_coordinates["phi"] = hit_coordinates["phi"].astype("float32")
    hit_coordinates["z"] = hit_coordinates["z"].astype("float32")

    neurons_weights = neurons_weights.astype("float32")
    neurons_biases = neurons_biases.astype("float32")

    # Calculate the neuron outputs
    neuron_outputs = calculate_neuron_output(
        hit_coordinates, neurons_weights, neurons_biases
    )
    # Compare the neuron outputs with the activations
    for i in range(neuron_outputs.shape[1]):
        if verbose:
            print(f"Neuron {i}")
            # Check if the activation value can be found in the neuron outputs
            for j in range(neuron_outputs.shape[0]):
                same_index = np.where(
                    neuron_outputs[j, i] == duplicated_activations_1[i, :]
                )
                assert (
                    len(same_index[0]) >= 1
                ), f"{same_index = } {np.abs(neuron_outputs[j, i] - duplicated_activations_1[i, :]).min() = }"
                assert j in same_index[0], f"{same_index = } {j = }"

        assert (
            neuron_outputs[:, i] == duplicated_activations_1[i, :]
        ).all(), f"Neuron {i} is wrong {(neuron_outputs[:, i] - duplicated_activations_1[i, :]).max() = }"


def handle_shared_hits(input_df, neuron_activations):
    # For each duplicate [r, phi, z] in feature, assign the same neuron output
    # Hit j is assigned to activation_1[:, j]
    # Create a unique key for each unique row
    # Convert each row to a tuple for factorization
    input_df_tuples = list(
        input_df[["r", "phi", "z"]].itertuples(index=False, name=None)
    )
    input_df_keys, unique_indices = pd.factorize(input_df_tuples, sort=False)

    keys = list(neuron_activations)

    activations_1_df = pd.DataFrame(neuron_activations[keys[0]].T)
    duplicated_activations_1 = (
        activations_1_df.iloc[input_df_keys].reset_index(drop=True).T
    ).to_numpy()

    dropped_indices = []

    try:
        # Verify that the assignment is correct
        verify_activation_assignement(input_df, duplicated_activations_1)
    except AssertionError as e:
        input_df, duplicated_activations_1, dropped_indices = (
            drop_wrong_activation_assignement(input_df, duplicated_activations_1)
        )

        print(f"Reverifying assignement")
        # Verify that the assignment is correct
        # verify_activation_assignement(input_df, duplicated_activations_1, verbose=True)
        verify_activation_assignement(input_df, duplicated_activations_1, verbose=False)

        print(f"{input_df_keys.shape = }")
        print(f"{input_df_keys = }")

    for layer in neuron_activations:
        # Add the duplicated activations to the neuron activations
        neuron_activations_layer_df = pd.DataFrame(neuron_activations[layer].T)
        duplicated_activations = (
            neuron_activations_layer_df.iloc[input_df_keys].reset_index(drop=True).T
        ).to_numpy()

        # Keep only the rows that are in the input_df
        duplicated_activations = np.delete(
            duplicated_activations, dropped_indices, axis=1
        )

        # print(f"{duplicated_activations.shape = }")

        neuron_activations[layer] = torch.tensor(duplicated_activations)
        # print(neuron_activations[layer].shape)

    if dropped_indices:
        print(f"{len(dropped_indices)} hits were dropped")

        activations_1_df = pd.DataFrame(neuron_activations[keys[0]].T)
        duplicated_activations_1 = activations_1_df.reset_index(drop=True).T.to_numpy()

        # Verify that the assignment is correct
        verify_activation_assignement(input_df, duplicated_activations_1)

    return neuron_activations, input_df


def entropy_1D(df_continuous, feature):
    # Check if the feature is discrete integer
    y_is_discrete = (
        df_continuous[feature].abs() == df_continuous[feature].abs().astype(int)
    ).all()

    from conditional_entropy import entropy_discrete

    if y_is_discrete:
        entropy_feature = entropy_discrete(df_continuous[feature].to_numpy())
        print(f"Feature: {feature}; Entropy({feature}): {entropy_feature}")
        # Compute the mutual information between the feature and itself
        entropy_feature_mutual = mutual_info_regression(
            df_continuous[feature].to_numpy().reshape(-1, 1),
            df_continuous[feature].to_numpy(),
            random_state=42,
            discrete_features=True,
        )
        print(f"Entropy({feature}) mutual: {entropy_feature_mutual}")
    else:
        # Compute the mutual information between the feature and itself
        entropy_feature = mutual_info_regression(
            df_continuous[feature].to_numpy().reshape(-1, 1),
            df_continuous[feature].to_numpy(),
            random_state=42,
        )

        # Mutual_info_regression is scale invariant!

        # Check the scale dependance of mutual_info_regression (compare with 2* scale)
        entropy_feature_scaled = mutual_info_regression(
            5000 * (df_continuous[feature].to_numpy().reshape(-1, 1)),
            5000 * (df_continuous[feature].to_numpy()),
            random_state=42,
        )
        print(f"Entropy({feature}) (mutual): {entropy_feature}")
        print(f"Entropy({feature}) (mutual) (scaled): {entropy_feature_scaled}")
    return entropy_feature


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
    # Cyllindrical coordinates to cartesian
    df["x"] = df["r"] * np.cos(df["phi"])
    df["y"] = df["r"] * np.sin(df["phi"])

    # Cyllindrical coordinates to spherical
    df["theta"] = np.arctan2(df["r"], df["z"])
    df["rho"] = np.sqrt(df["r"] ** 2 + df["z"] ** 2)

    # Calculate eta
    df["eta"] = -np.log(np.tan(df["theta"] / 2))
    return df


# Multi-feature Plot Against Neuron Output
def plot_neuron_output_vs_features(neuron_output, df, features):
    for feature in features:
        plt.plot(neuron_output, df[feature], "o", label=feature)
    plt.legend()
    plt.grid(True)
    plt.show()


# # Load the particles
# particles = load_csv_data(file_name="event000000101-particles.csv", directory="data")
# # df.columns = ["r", "phi", "z"]
# print(particles.head())

# # Load the truth
# truth = load_csv_data(file_name="event000000101-truth.csv", directory="data")
# print(truth.head())

# # Get the unique particle id in truth (particle_id_1 + particle_id_2 + particle_id)
# # print(truth["particle_id_1"])
# # print(truth["particle_id_2"])
# # print(truth["particle_id"])
# print(truth["particle_id_1"].unique())
# print(truth["particle_id_2"].unique())
# print(truth["particle_id"].unique())
# # Compare the unique particle id
# print(truth["particle_id_1"].unique().shape)
# print(truth["particle_id_2"].unique().shape)
# print(truth["particle_id"].unique().shape)

# # Count when particle_id_1 == particle_id_2 vs particle_id_1 == particle_id_2 == particle_id
# print(truth[truth["particle_id_1"] == truth["particle_id_2"]].shape)
# print(
#     truth[
#         (truth["particle_id_1"] == truth["particle_id_2"])
#         & (truth["particle_id_1"] == truth["particle_id"])
#     ].shape
# )
# print(truth.shape)

# # Make sure that particle_id is either particle_id_1 or particle_id_2
# print(
#     truth[
#         (truth["particle_id"] != truth["particle_id_1"])
#         & (truth["particle_id"] != truth["particle_id_2"])
#     ].shape
# )

# # Get shape of truth where particle_id corresponding line in particles 'pass' column is 'YES'
# # Ensure that the particle_id in truth exists in particles before accessing the 'pass' column
# passed_particle_ids = particles[particles["pass"] == "YES"]["particle_id"]
# passed_particle_ids = particles[particles["pt"] > 1000]["particle_id"]
# # valid_particle_ids = truth["particle_id"].isin(particles["particle_id"])
# valid_particle_ids = (
#     truth["particle_id"].isin(passed_particle_ids)
#     & truth["particle_id_1"].isin(passed_particle_ids)
#     & truth["particle_id_2"].isin(passed_particle_ids)
# )
# # passed_valid_particle_ids = truth["particle_id"].isin(passed_particle_ids)
# passed_valid_particle_ids = valid_particle_ids
# # print(valid_particle_ids)
# print(truth[passed_valid_particle_ids].shape)

# # Save truth[passed_valid_particle_ids] into new csv file
# truth[passed_valid_particle_ids].to_csv(
#     "data/event000000101-truth-passed.csv", index=False
# )

# print()


def plot_feature_distribution(particles, truth_particles):
    for i, feature_name in enumerate(particles.columns):
        # Determine if the data is constant or categorical or boolean or integer or continuous
        # Constant is when all the values are the same
        # Categorical is when there is values that are not numbers
        # Boolean is when there is only True and False or 1 and 0
        # Integer is when there is only integers
        # Continuous is when there is only floats
        type_of_data = "continuous"
        if particles[feature_name].nunique() == 1:
            type_of_data = "constant"
        elif particles[feature_name].dtype == "object":
            type_of_data = "categorical"
        # handle boolean by checking if there is only True and False or 1 and 0
        elif (
            particles[feature_name].isin([True, False]).all()
            or particles[feature_name].isin([1, 0]).all()
        ):
            type_of_data = "boolean"
        # handle positive and negative integers by using absolute value to check if it is an integer
        elif (
            particles[feature_name].abs() == particles[feature_name].abs().astype(int)
        ).all():
            type_of_data = "integer"

        print(f"Feature {feature_name} is {type_of_data}")

        # Determine bins based on the type of data
        bins = 100
        if type_of_data == "categorical":
            bins = particles[feature_name].nunique()
            bins = min(bins, 100)
        elif type_of_data == "boolean":
            bins = 2
        elif type_of_data == "integer":
            if particles[feature_name].max() - particles[feature_name].min() < 100:
                bins = np.arange(
                    particles[feature_name].min() - 0.5,
                    particles[feature_name].max() + 1,
                )

        plt.hist(particles[feature_name], bins=bins, alpha=0.5, label="Before cut")
        plt.hist(
            truth_particles[feature_name],
            bins=bins,
            alpha=0.5,
            label="After cut",
        )

        # Do y log scale if there is more than 1000 counts in one bin
        try:
            if (
                max(np.histogram(particles[feature_name], bins=100)[0]) > 1000
                or max(np.histogram(truth_particles[feature_name], bins=100)[0]) > 1000
            ):
                plt.yscale("log")
        except:
            pass

        plt.xlabel(feature_name)
        plt.legend()
        plt.title(f"Particles {feature_name} Distribution")
        plt.grid(True)
        plt.ylabel("Count")
        plt.show()


# # Load the particles
# particles = load_csv_data(
#     file_name="event000000101-hard-cut-particles.csv", directory="csv"
# )


# Plot the distribution of the particles
# plot_feature_distributions(particles)
# plot_feature_distribution(particles, truth_particles)

# print(input_df[["eta", "phi"]].to_numpy().reshape(n_hits, 2).shape)
# print(input_df[["eta", "phi"]].to_numpy().reshape(n_hits, 2))

# print(mutual_information(input_df, "eta", "eta"))
# print(calculate_entropy(input_df, "eta", "eta"))
# print(conditional_entropy(input_df["eta"].to_numpy(), input_df["eta"].to_numpy()))
# print(
#     mutual_info_regression(
#         input_df["eta"].to_numpy().reshape(-1, 1),
#         input_df["eta"].to_numpy(),
#         random_state=42,
#     )
# )
# print(conditional_entropy(input_df["phi"].to_numpy(), input_df["eta"].to_numpy()))
# print(
#     mutual_info_regression(
#         input_df["phi"].to_numpy().reshape(-1, 1),
#         input_df["eta"].to_numpy(),
#         random_state=42,
#     )
# )
# print(
#     mutual_info_regression(
#         input_df[["eta", "phi"]].to_numpy().reshape(n_hits, 2),
#         input_df["eta"].to_numpy(),
#         random_state=42,
#     )
# )
# exit()

# Plot the scaled data for feature "r" vs "z"
# plt.scatter(df_scaled["z"], df_scaled["r"])
# plt.xlabel("z")
# plt.ylabel("r")
# plt.show()


def remove_invalid_match(input_df, neuron_activations):
    # Make sure the nan rows are the ones with '_merge' column as 'left_only'
    # print(input_df.loc[input_df["_merge"] == "left_only"])
    # print(input_df.loc[input_df["_merge"] == "left_only"]["particle_id"].isna())
    # print(input_df.loc[input_df["_merge"] == "left_only"])
    assert (
        input_df.loc[input_df["_merge"] == "left_only"]["particle_id"].isna().all()
        == True
    )
    n_hits = input_df.shape[0]

    # Bad indices
    bad_indices = input_df.loc[input_df["_merge"] == "left_only"].index
    print(f"{bad_indices = }")

    def th_delete(tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask = mask.reshape(tensor.shape)
        mask[:, indices] = False
        return tensor[mask].reshape(tensor.shape[0], -1)

    # def th_delete(tensor, indices):
    #     mask = torch.ones(tensor.numel(), dtype=torch.bool)
    #     mask = mask.reshape(tensor.shape)
    #     mask[:, indices] = False
    #     return tensor[mask]

    # Remove the bad indices
    # print(f"{bad_indices.shape = }")
    # print(f"{input_df.shape = }")
    # input_df.dropna(subset=["particle_id"], inplace=True)
    # print(f"{input_df.shape = }")
    input_df.drop(bad_indices, inplace=True)
    assert input_df.shape[0] == n_hits - bad_indices.shape[0], f"{input_df.shape = }"

    # Make sure the bad indices are not in the df
    assert (
        input_df.loc[input_df["_merge"] == "left_only"].shape[0] == 0
    ), f"{input_df.loc[input_df['_merge'] == 'left_only'].shape = }"

    # Reverse sort the bad indices
    # bad_indices = np.sort(bad_indices)[::-1]
    # bad_indices = np.sort(bad_indices)[::-1]
    for key in neuron_activations:
        neuron_activations[key] = neuron_activations[key][:, input_df.index.to_numpy()]
        # print(f"{bad_indices = }")
        # neuron_activations[key] = th_delete(neuron_activations[key], bad_indices)
        # for bad_index in bad_indices:
        #     neuron_activations[key] = torch.cat(
        #         (
        #             neuron_activations[key][:, :bad_index],
        #             neuron_activations[key][:, bad_index + 1 :],
        #         ),
        #         dim=1,
        #     )

    # # Remove the bad indices
    # print(f"{bad_indices.shape = }")
    # print(f"{input_df.shape = }")
    # input_df.dropna(subset=["particle_id"], inplace=True)
    # print(f"{input_df.shape = }")
    # assert (
    #     input_df.shape[0] == n_hits - bad_indices.shape[0]
    # ), f"{input_df.shape = }"

    # Make sure the bad indices are not in the df
    assert (
        input_df.loc[input_df["_merge"] == "left_only"].shape[0] == 0
    ), f"{input_df.loc[input_df['_merge'] == 'left_only'].shape = }"

    input_df.reset_index(drop=True, inplace=True)

    # print(f"{input_df.shape = }")

    return input_df, neuron_activations


event_list = [101, 103, 104, 105, 106, 107, 109]
# event_list = [101, 104, 107]
# event_list = [101, 107]
# Dataframe to store the truth particles for all events
all_input_df = pd.DataFrame()
all_neuron_activations = {}
for event in event_list:
    print(f"Event {event}")
    check_2(event_id=event)
    truth_particles = load_event_data(event_id=event, verbose=True)

    # Compute r, phi, z for the truth particles
    truth_particles["r"] = np.sqrt(
        truth_particles["x"] ** 2 + truth_particles["y"] ** 2
    )
    truth_particles["phi"] = np.arctan2(truth_particles["y"], truth_particles["x"])
    truth_particles["z"] = truth_particles["z"]

    input_df = match_input_data(truth_particles, event_id=event, load_data=False)

    # Add various coordinate transformations
    input_df = add_coordinate_transformations(input_df)

    # Read activations
    neuron_activations = load_event_activations(event_id=event)

    # neuron_activations, input_df = handle_shared_hits(input_df, neuron_activations)
    verify_activation_assignement(
        input_df, neuron_activations[list(neuron_activations)[0]].numpy()
    )

    all_input_df = pd.concat(
        [all_input_df, input_df], ignore_index=True
    )  # index issues when dropping if they are not ignored here
    for key in neuron_activations:
        if key not in all_neuron_activations:
            all_neuron_activations[key] = neuron_activations[key]
        else:
            all_neuron_activations[key] = torch.cat(
                (all_neuron_activations[key], neuron_activations[key]), dim=1
            )

n_hits = all_input_df.shape[0]


all_input_df, all_neuron_activations = remove_invalid_match(
    all_input_df, all_neuron_activations
)

# Save the input_df to a CSV file
# all_input_df.to_csv(f"input_data_event_all.csv", index=False)

# Save computed activation
layer = 1
model_state_dict = load_model()
neurons_weights, neurons_biases = get_layer_parameters(model_state_dict, layer)

# Calculate the neuron's output for each hit
hit_coordinates = all_input_df[["r", "phi", "z"]]

print(f"{hit_coordinates.dtypes = }")

from load_data import scale_data

# Scale the hit coordinates
hit_coordinates = scale_data(hit_coordinates, scales=[1 / 1000, 1 / 3.14, 1 / 1000])

# Calculate the neuron outputs
neuron_outputs = calculate_neuron_output(
    hit_coordinates, neurons_weights, neurons_biases
)

# # Save the neuron outputs for first layer to a CSV file
# pd.DataFrame(neuron_outputs).to_csv(f"neuron_outputs_event_all.csv", index=False)

# # Save the neuron activations for first layer to a CSV file
# pd.DataFrame(all_neuron_activations[list(all_neuron_activations)[0]].T.numpy()).to_csv(
#     f"neuron_activations_event_all.csv", index=False
# )

verify_activation_assignement(
    all_input_df, all_neuron_activations[list(all_neuron_activations)[0]].numpy()
)
# exit()

# # Drop rows with 'nan' values
# all_input_df = all_input_df.dropna()
# # Remove corresponding rows in neuron_activations
# for key in all_neuron_activations:
#     all_neuron_activations[key] = all_neuron_activations[key][
#         :, all_input_df.index.to_numpy()
#     ]

neuron_activations = all_neuron_activations
input_df = all_input_df

# Map particle_id to [0..N]
particle_id_map = {
    particle_id: i for i, particle_id in enumerate(input_df["particle_id"].unique())
}
input_df["particle_id_mapped"] = input_df["particle_id"].map(particle_id_map)

keys = list(neuron_activations)
# Remove duplicate hits
print(all_input_df.shape)
# all_input_df = all_input_df.drop_duplicates(subset=["r", "phi", "z"], ignore_index=True)
print(all_input_df.shape)
print(all_neuron_activations[keys[0]].shape)
verify_activation_assignement(all_input_df, all_neuron_activations[keys[0]].numpy())
activations_1 = neuron_activations[keys[0]].numpy()
activations_3 = neuron_activations[keys[2]].numpy()
activations_4 = neuron_activations[keys[3]].numpy()
event = 0

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

# Plot r vs z with the color of the points representing the pt
plt.scatter(input_df["z"], input_df["r"], c=np.log10(input_df["pt"]))
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
# plt.savefig(f"neuron_86_output_event{event:09d}_r_z.png")
plt.show()

plt.scatter(input_df["z"], input_df["r"], c=input_df["particle_id_mapped"])
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.show()


# Remove non-continuous features
df_continuous = input_df.select_dtypes(include=[np.number])

# Add 3 random variables for each hit
# uniform, normal, poisson
df_continuous["uniform_hit"] = np.random.uniform(0, 1, size=(df_continuous.shape[0], 1))
df_continuous["normal_hit"] = np.random.normal(0, 1, size=(df_continuous.shape[0], 1))
df_continuous["poisson_hit"] = np.random.poisson(1, size=(df_continuous.shape[0], 1))


def entropy_discrete(y):
    """
    Compute entropy for a discrete variable Y.
    :param y: Numpy array of shape (n_samples,), the discrete variable Y
    :return: Estimated entropy
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log(probabilities))

def compute_information_coverage(event, df_continuous):
    # Load the data
    csv_dir = Path("./conditional_entropy")
    csv_files = list(csv_dir.glob(f"mutual_information_event{event:09d}_layer*.csv"))

    # Read all the csv files and concatenate them
    df_mutual = pd.concat((pd.read_csv(csv_file) for csv_file in csv_files))
    print(df_mutual)

    # Compute the information coverage dataframe
    df_information_coverage = pd.DataFrame()
    df_information_coverage["layer"] = df_mutual["layer"]
    df_information_coverage["neuron"] = df_mutual["neuron"]
    for feature in df_mutual.columns[2:]:
        entropy_feature = entropy_1D(df_continuous, feature)

        df_information_coverage[feature] = df_mutual[feature] / entropy_feature

    # Save the information coverage to a CSV file
    df_information_coverage.to_csv(
        csv_dir / f"information_coverage_event{event:09d}.csv", index=False
    )
    print(df_information_coverage.max(axis=0))

    return df_information_coverage


df_information_coverage = compute_information_coverage(event, df_continuous)

print(df_information_coverage)

# Plot the data for each layer (Y axis: "layer", X axis: "mutual_information")
def plot_information_coverage(event, df_information_coverage):
    output_dir = Path("./conditional_entropy")
    for feature in df_information_coverage.columns:
        if feature in ["layer", "neuron"]:
            continue

        # Do violin plot for each layer
        plt.violinplot(
            [
                df_information_coverage[df_information_coverage["layer"] == layer][
                    feature
                ]
                * 100
                for layer in df_information_coverage["layer"].unique()
            ],
            vert=False,
            # Set y axis values
            positions=df_information_coverage["layer"].unique(),
            widths=1,
        )
        marker_size = 5
        marker = "o"
        plt.scatter(
            df_information_coverage[feature] * 100,
            df_information_coverage["layer"],
            s=marker_size,
            marker=marker,
        )
        # plt.xlabel("Information coverage [%]")
        plt.xlabel("Proficiency [%]")
        plt.ylabel("Layer")
        # plt.title(f"Information coverage for {feature}")
        plt.title(f"Single neuron proficiency for {feature}")
        plt.grid(True)
        xlim = (-4, 105)
        assert df_information_coverage[feature].max() < xlim[1] / 100
        assert (
            not np.isfinite(df_information_coverage[feature].min())
            or df_information_coverage[feature].min() > xlim[0] / 100
        )
        plt.xlim(*xlim)
        # plt.savefig(f"information_coverage_{feature}_event{event:09d}.png")
        plt.savefig(
            output_dir / f"proficiency_single_neuron_{feature}_event{event:09d}.png"
        )
        plt.show()


plot_information_coverage(event, df_information_coverage)

# Compute mutual information for all neuron outputs in a layer and save them into a CSV file
def compute_mutual_information(event, neuron_activations, df_continuous):
    """
    Compute mutual information for all neuron outputs in a layer and save them into a CSV file.

    Parameters:
    event (int): The event number.
    neuron_activations (dict): Dictionary containing neuron activations for each layer.
    df_continuous (pd.DataFrame): DataFrame containing continuous features.

    Returns:
        None
    """
    mutual_information_df = pd.DataFrame()
    # Start from the last layer
    reversed_neuron_activations = list(neuron_activations.keys())[::-1]
    for layer_index, layer in enumerate(reversed_neuron_activations):
        start_time = time.time()
        layer_index = len(neuron_activations) - layer_index

        # Get all neuron outputs of the current layer as a 2D array (shape: num_samples x num_neurons)
        layer_outputs = np.array(
            neuron_activations[layer]
        ).T  # Transpose to match mutual_info_regression input format

        print(f"Computing mutual information for Layer {layer_index}")

        output_dir = Path("conditional_entropy")
        output_dir.mkdir(exist_ok=True)

        # File path for the current layer's CSV
        layer_file_path = f"mutual_information_event{event:09d}_layer{layer_index}.csv"

        layer_file_path = output_dir / layer_file_path

        # Initialize or load existing data for the layer
        if os.path.exists(layer_file_path):
            mutual_information_df_layer = pd.read_csv(layer_file_path)
            processed_neurons = mutual_information_df_layer["neuron"].tolist()
            print(
                f"Found existing file for Layer {layer_index}. Resuming from Neuron {len(processed_neurons)}"
            )
        else:
            mutual_information_df_layer = pd.DataFrame()
            processed_neurons = []

        for neuron_idx in range(layer_outputs.shape[1]):
            if neuron_idx in processed_neurons:
                continue  # Skip already processed neurons

            print(f"Computing mutual information for Neuron {neuron_idx}")

            mutual_information_values = mutual_info_regression(
                df_continuous.to_numpy(), layer_outputs[:, neuron_idx], random_state=42
            )

            # # Use Adjusted Mutual Information
            # from sklearn.metrics import adjusted_mutual_info_score

            # adjusted_mutual_information_values = []
            # for i, feature in enumerate(df_continuous.columns):
            #     adjusted_mutual_information_values.append(
            #         adjusted_mutual_info_score(
            #             df_continuous[feature].to_numpy(), layer_outputs[:, neuron_idx]
            #         )
            #     )

            # Create a new row for the current neuron
            new_row = {
                "layer": layer_index,
                "neuron": neuron_idx,
                **{
                    feature: mutual_information_values[i]
                    for i, feature in enumerate(df_continuous.columns)
                },
            }

            # Append the new row to the existing DataFrame for the layer
            mutual_information_df_layer = pd.concat(
                [mutual_information_df_layer, pd.DataFrame([new_row])],
                ignore_index=True,
            )

            # Save the updated layer data to the CSV file
            mutual_information_df_layer.to_csv(layer_file_path, index=False)

        # Append the processed layer to the global DataFrame
        mutual_information_df = pd.concat(
            [mutual_information_df, mutual_information_df_layer], ignore_index=True
        )

        # Save the combined mutual information to a global CSV file
        mutual_information_df.to_csv(
            f"mutual_information_event{event:09d}.csv", index=False
        )

        end_time = time.time()
        print(f"Time taken for Layer {layer_index}: {end_time - start_time} seconds")

    # Final save of the mutual information to a CSV file
    mutual_information_df.to_csv(
        f"mutual_information_event{event:09d}.csv", index=False
    )


compute_mutual_information(event, neuron_activations, df_continuous)

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

# Plot r vs z with the color of the points representing the output of neuron 86
plt.scatter(df_scaled["z"], df_scaled["r"], c=neuron_86_output)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.savefig(f"neuron_86_output_event{event:09d}_r_z.png")
plt.show()

# Plot isocurves of the output of neuron 86 with a lot of points
plt.tricontourf(df_scaled["z"], df_scaled["r"], neuron_86_output, levels=20)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.savefig(f"neuron_86_output_event{event:09d}_r_z_isocurves.png")
plt.show()

# Plot the output of neuron 44
plt.hist(
    neuron_44_output,
    bins=100,
    alpha=0.5,
    label="44",
)
plt.legend()
plt.grid(True)
plt.show()

# Plot r vs z with the color of the points representing the output of neuron 44
plt.scatter(df_scaled["z"], df_scaled["r"], c=neuron_44_output)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.savefig(f"neuron_44_output_event{event:09d}_r_z.png")
plt.show()

# Plot isocurves of the output of neuron 44 with a lot of points
plt.tricontourf(df_scaled["z"], df_scaled["r"], neuron_44_output, levels=20)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.savefig(f"neuron_44_output_event{event:09d}_r_z_isocurves.png")
plt.show()

# Plot the output of neuron 44 vs the output of neuron 86
plt.scatter(neuron_86_output, neuron_44_output)
plt.xlabel("Neuron 86")
plt.ylabel("Neuron 44")
plt.savefig("neuron_44_output_vs_neuron_86_output.png")
plt.show()

# Plot the output of neuron 935
plt.hist(
    neuron_935_output,
    bins=100,
    alpha=0.5,
    label="935",
)
plt.legend()
plt.grid(True)
plt.show()


# Plot r vs z with the color of the points representing the output of neuron 935
plt.scatter(df_scaled["z"], df_scaled["r"], c=neuron_935_output)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.savefig(f"neuron_935_output_event{event:09d}_r_z.png")
plt.show()

# Plot isocurves of the output of neuron 935 with a lot of points
plt.tricontourf(df_scaled["z"], df_scaled["r"], neuron_935_output, levels=20)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.savefig(f"neuron_935_output_event{event:09d}_r_z_isocurves.png")
plt.show()

# # Plot isocurves of the output of neuron 935 with a lot of points
# plt.tricontourf(df_scaled["z"], df_scaled["r"], neuron_935_output, levels=20)
# plt.scatter(df_scaled["z"], df_scaled["r"], c=neuron_935_output)
# plt.xlabel("z")
# plt.ylabel("r")
# plt.colorbar()
# plt.show()

# Plot the output of neuron 935 vs the output of neuron 86
plot_output_correlation(neuron_86_output, neuron_935_output, "Neuron 86", "Neuron 935")

# Plot the output of neuron 935 vs the output of neuron 44
plot_output_correlation(neuron_44_output, neuron_935_output, "Neuron 44", "Neuron 935")

# Plot neuron 935 output vs x, y, z, r, phi, theta, rho
# plot_neuron_output_vs_features(
#     neuron_935_output, input_df, ["x", "y", "z", "r", "phi", "theta", "rho", "eta"]
# )

exit()

# Clustering
for feature in input_df.columns:
    plt.scatter(neuron_935_output, input_df[feature], c=input_df["particle_id_mapped"])
    plt.ylabel(feature)
    plt.xlabel("Neuron 935")
    plt.show()


plot_scatter_with_color(input_df, "eta", "pt", neuron_935_output, "eta", "pt")
exit()

plot_scatter_with_color(input_df, "x", "y", neuron_935_output, "x", "y")
plot_scatter_with_color(input_df, "r", "phi", neuron_935_output, "r", "phi")
plot_scatter_with_color(input_df, "theta", "rho", neuron_935_output, "theta", "rho")
plot_scatter_with_color(input_df, "eta", "phi", neuron_935_output, "eta", "phi")
plot_scatter_with_color(input_df, "eta", "r", neuron_935_output, "eta", "r")


# Plot correlation between neuron 935 output and r
plot_output_correlation(neuron_935_output, input_df["r"], "Neuron 935", "r")
plot_output_correlation(neuron_935_output, input_df["eta"], "Neuron 935", "eta")
plot_output_correlation(neuron_935_output, input_df["rho"], "Neuron 935", "rho")
plot_output_correlation(neuron_935_output, input_df["theta"], "Neuron 935", "theta")
plot_output_correlation(neuron_935_output, input_df["phi"], "Neuron 935", "phi")

# Same for neuron 895 in layer 4
neuron_895_output = activations_4[895]

# Plot the output of neuron 895
plt.hist(
    neuron_895_output,
    bins=100,
    alpha=0.5,
    label="895",
)
plt.legend()
plt.grid(True)
plt.show()

# Plot r vs z with the color of the points representing the output of neuron 895
plt.scatter(df_scaled["z"], df_scaled["r"], c=neuron_895_output)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.savefig(f"neuron_895_output_event{event:09d}_r_z.png")
plt.show()

# Plot isocurves of the output of neuron 895 with a lot of points
plt.tricontourf(df_scaled["z"], df_scaled["r"], neuron_895_output, levels=20)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.savefig(f"neuron_895_output_event{event:09d}_r_z_isocurves.png")
plt.show()

# Plot the output of neuron 895 vs the output of neuron 935
plot_output_correlation(
    neuron_935_output, neuron_895_output, "Neuron 935", "Neuron 895"
)

# Plot the output of neuron 895 vs the output of neuron 86
plot_output_correlation(neuron_86_output, neuron_895_output, "Neuron 86", "Neuron 895")

# Plot the output of neuron 895 vs the output of neuron 44
plot_output_correlation(neuron_44_output, neuron_895_output, "Neuron 44", "Neuron 895")


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
#     plt.tricontourf(df_scaled["z"], df_scaled["r"], neuron_output, levels=20)
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
#     axs[0].tricontourf(df_scaled["z"], df_scaled["r"], neuron_output, levels=20)
#     axs[0].set_xlabel("z")
#     axs[0].set_ylabel("r")
#     axs[0].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

#     axs[1].tricontourf(df_scaled["r"], df_scaled["phi"], neuron_output, levels=20)
#     axs[1].set_xlabel("r")
#     axs[1].set_ylabel("phi")
#     axs[1].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

#     axs[2].tricontourf(df_scaled["z"], df_scaled["phi"], neuron_output, levels=20)
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
#     axs[0].scatter(df_scaled["z"], df_scaled["r"], c=neuron_output)
#     axs[0].set_xlabel("z")
#     axs[0].set_ylabel("r")
#     axs[0].set_title(f"Layer {layer} Neuron {i} Output")

#     axs[1].scatter(df_scaled["r"], df_scaled["phi"], c=neuron_output)
#     axs[1].set_xlabel("r")
#     axs[1].set_ylabel("phi")
#     axs[1].set_title(f"Layer {layer} Neuron {i} Output")

#     axs[2].scatter(df_scaled["z"], df_scaled["phi"], c=neuron_output)
#     axs[2].set_xlabel("z")
#     axs[2].set_ylabel("phi")
#     axs[2].set_title(f"Layer {layer} Neuron {i} Output")

#     plt.savefig(
#         no_iso_dir_layer_subplots / f"neuron_{i}_output_event{event:09d}_r_z_phi.png"
#     )
#     plt.close()

# Do the same for layer 13
layer = 12

layer_name = keys[layer - 1]
# layer_name_activations = keys[layer - 2]


# neurons_weights, neurons_biases = get_layer_parameters(
#     state_dict, layer
# )

# activations = neuron_activations[layer_name_activations].numpy()
activations = neuron_activations[layer_name].numpy()

# Do the same without the isocurves
# no_iso_dir_layer_subplots = Path(f"no_iso_layer_{layer}_subplots")
# no_iso_dir_layer_subplots.mkdir(exist_ok=True)

# for i in range(len(activations)):
#     # neuron_weights = neurons_weights[i]
#     # neuron_biases = neurons_biases[i]

#     # Do input*weights + biases
#     neuron_output = activations[i]

#     fig, axs = plt.subplots(1, 4, figsize=(22, 5))

#     # Plot isocurves of the output of neuron i with a lot of points
#     axs[0].scatter(df_scaled["z"], df_scaled["r"], c=neuron_output)
#     axs[0].set_xlabel("z")
#     axs[0].set_ylabel("r")
#     axs[0].set_title(f"Layer {layer} Neuron {i} Output")

#     axs[1].scatter(df_scaled["x"], df_scaled["y"], c=neuron_output)
#     axs[1].set_xlabel("x")
#     axs[1].set_ylabel("y")
#     axs[1].set_title(f"Layer {layer} Neuron {i} Output")
#     # Make the aspect ratio equal
#     axs[1].set_aspect("equal")

#     axs[2].scatter(df_scaled["r"], df_scaled["phi"], c=neuron_output)
#     axs[2].set_xlabel("r")
#     axs[2].set_ylabel("phi")
#     axs[2].set_title(f"Layer {layer} Neuron {i} Output")

#     axs[3].scatter(df_scaled["z"], df_scaled["phi"], c=neuron_output)
#     axs[3].set_xlabel("z")
#     axs[3].set_ylabel("phi")
#     axs[3].set_title(f"Layer {layer} Neuron {i} Output")

#     plt.savefig(
#         no_iso_dir_layer_subplots / f"neuron_{i}_output_event{event:09d}_r_z_phi.png"
#     )
#     plt.close()

# # Do the same with isocurves
# isocurves_dir_layer_subplots = Path(f"isocurves_layer_{layer}_subplots")
# isocurves_dir_layer_subplots.mkdir(exist_ok=True)

# for i in range(len(activations)):
#     # neuron_weights = neurons_weights[i]
#     # neuron_biases = neurons_biases[i]

#     # Do input*weights + biases
#     neuron_output = activations[i]

#     fig, axs = plt.subplots(1, 4, figsize=(22, 5))

#     # Plot isocurves of the output of neuron i with a lot of points
#     axs[0].tricontourf(df_scaled["z"], df_scaled["r"], neuron_output, levels=20)
#     axs[0].set_xlabel("z")
#     axs[0].set_ylabel("r")
#     axs[0].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

#     axs[1].tricontourf(df_scaled["x"], df_scaled["y"], neuron_output, levels=20)
#     axs[1].set_xlabel("x")
#     axs[1].set_ylabel("y")
#     axs[1].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

#     axs[2].tricontourf(df_scaled["r"], df_scaled["phi"], neuron_output, levels=20)
#     axs[2].set_xlabel("r")
#     axs[2].set_ylabel("phi")
#     axs[2].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

#     axs[3].tricontourf(df_scaled["z"], df_scaled["phi"], neuron_output, levels=20)
#     axs[3].set_xlabel("z")
#     axs[3].set_ylabel("phi")
#     axs[3].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

#     plt.savefig(
#         isocurves_dir_layer_subplots
#         / f"neuron_{i}_output_event{event:09d}_r_z_phi_isocurves.png"
#     )
#     plt.close()

isocurves_dir_layer_subplots = Path(f"isocurves_layer_{layer}_subplots_eta_rho")
isocurves_dir_layer_subplots.mkdir(exist_ok=True)

for i in range(len(activations)):
    # neuron_weights = neurons_weights[i]
    # neuron_biases = neurons_biases[i]

    # Do input*weights + biases
    neuron_output = activations[i]

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    # Plot isocurves of the output of neuron i with a lot of points
    axs[0].tricontourf(df_scaled["z"], df_scaled["r"], neuron_output, levels=20)
    axs[0].set_xlabel("z")
    axs[0].set_ylabel("r")
    axs[0].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    axs[1].tricontourf(df_scaled["x"], df_scaled["y"], neuron_output, levels=20)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    axs[2].tricontourf(df_scaled["eta"], df_scaled["rho"], neuron_output, levels=20)
    axs[2].set_xlabel("eta")
    axs[2].set_ylabel("rho")
    axs[2].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    axs[3].tricontourf(df_scaled["eta"], df_scaled["z"], neuron_output, levels=20)
    axs[3].set_xlabel("eta")
    axs[3].set_ylabel("z")
    axs[3].set_title(f"Layer {layer} Neuron {i} Output Isocurves")

    plt.savefig(
        isocurves_dir_layer_subplots
        / f"neuron_{i}_output_event{event:09d}_r_z_phi_isocurves.png"
    )
    plt.close()
