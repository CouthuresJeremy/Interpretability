import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

event = 101


# Load Data Function
def load_csv_data(file_name="input_data_event000000101.csv", directory="csv"):
    csv_dir = Path(directory)
    csv_file = csv_dir / file_name
    df = pd.read_csv(csv_file)
    return df


# Scaling Data Function
def scale_data(df, scales=[1000, 3.14, 1000]):
    df_scaled = df.copy()
    for i, feature in enumerate(df.columns):
        df_scaled[feature] *= scales[i]
    return df_scaled


# Plotting Distribution Function
def plot_feature_distributions(df, scales=[1000, 3.14, 1000]):
    for i, feature_name in enumerate(df.columns):
        plt.hist(df[feature_name] * scales[i], bins=100, alpha=0.5)
        plt.xlabel(feature_name)
        plt.show()


# Load Model Function
def load_model(model_path="model/best--f1=0.313180-epoch=89.ckpt"):
    model = torch.load(model_path, map_location=torch.device("cpu"))
    return model["state_dict"]


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


# Load the input data
df = load_csv_data(file_name="input_data_event000000101.csv", directory="csv")
df.columns = ["r", "phi", "z"]
print(df.head())


# node_features: [r,    phi,  z]
# node_scales:   [1000, 3.14, 1000]
df.columns = ["r", "phi", "z"]
feature_scale = [1000, 3.14, 1000]

# Plot distribution of input data for each feature
feature_names = df.columns
# print(feature_names)
# for feature_name in feature_names:
#     plt.hist(
#         df[feature_name] * feature_scale[feature_names.get_loc(feature_name)],
#         bins=100,
#         alpha=0.5,
#     )
#     plt.xlabel(feature_name)
#     plt.show()
#     plt.close()

# Scale the data
df_scaled = df.copy()
for feature_name in feature_names:
    df_scaled[feature_name] = (
        df[feature_name] * feature_scale[feature_names.get_loc(feature_name)]
    )

# Rename the columns
df_scaled.columns = ["r", "phi", "z"]

# Add various coordinate transformations
df_scaled = add_coordinate_transformations(df_scaled)

# Plot the scaled data for feature "r" vs "z"
# plt.scatter(df_scaled["z"], df_scaled["r"])
# plt.xlabel("z")
# plt.ylabel("r")
# plt.show()

# Read model

model = torch.load(
    "model/best--f1=0.313180-epoch=89.ckpt", map_location=torch.device("cpu")
)

state_dict = model["state_dict"]
state_dict_keys = list(state_dict.keys())
print(state_dict_keys)

# Get weights and biases for neuron 86 in layer 1
neuron_weights = state_dict[state_dict_keys[0]].numpy()
neuron_biases = state_dict[state_dict_keys[1]].numpy()

neuron_86_weights = neuron_weights[86]
neuron_86_biases = neuron_biases[86]

# Do input*weights + biases
neuron_86_output = df.to_numpy() @ neuron_86_weights.T + neuron_86_biases

# Plot the output of neuron 86
plt.hist(
    neuron_86_output,
    bins=100,
    alpha=0.5,
    label="86",
)
plt.legend()
plt.grid(True)
plt.show()

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


# Same for neuron 44 in layer 1
neuron_44_weights = neuron_weights[44]
neuron_44_biases = neuron_biases[44]

# Do input*weights + biases
neuron_44_output = df.to_numpy() @ neuron_44_weights.T + neuron_44_biases

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

# Same for neuron 935 in layer 4
neuron_weights_4 = state_dict[state_dict_keys[2]].numpy()
neuron_biases_4 = state_dict[state_dict_keys[3]].numpy()

neuron_935_weights = neuron_weights_4[935]
neuron_935_biases = neuron_biases_4[935]

# Read activations
event = 100
event = 101
activations = torch.load(
    f"activations/activations_event{event:09d}.pt", map_location=torch.device("cpu")
)
print(activations)

for key in activations:
    print(key, activations[key].shape)

keys = list(activations)
print(keys)
layer_1_name = keys[0]
layer_3_name = keys[2]

# Transpose the activation to get the neuron distributions
neuron_activations = {
    layer_name: layer_activations.T
    for layer_name, layer_activations in activations.items()
}

activations_3 = neuron_activations[layer_3_name].numpy()
print(activations_3.shape)
print(neuron_935_weights.T.shape)

# Do input*weights + biases
neuron_935_output = activations_3.T @ neuron_935_weights.T + neuron_935_biases

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

# Cyllindrical coordinates to cartesian
df["x"] = df["r"] * np.cos(df["phi"])
df["y"] = df["r"] * np.sin(df["phi"])

# Cyllindrical coordinates to spherical
df["theta"] = np.arctan2(df["r"], df["z"])
df["rho"] = np.sqrt(df["r"] ** 2 + df["z"] ** 2)

# Calculate eta
df["eta"] = -np.log(np.tan(df["theta"] / 2))

# Plot neuron 935 output vs x, y, z, r, phi, theta, rho
# plot_neuron_output_vs_features(
#     neuron_935_output, df, ["x", "y", "z", "r", "phi", "theta", "rho", "eta"]
# )


plot_scatter_with_color(df, "x", "y", neuron_935_output, "x", "y")
plot_scatter_with_color(df, "r", "phi", neuron_935_output, "r", "phi")
plot_scatter_with_color(df, "theta", "rho", neuron_935_output, "theta", "rho")
plot_scatter_with_color(df, "eta", "phi", neuron_935_output, "eta", "phi")
plot_scatter_with_color(df, "eta", "r", neuron_935_output, "eta", "r")


# Plot correlation between neuron 935 output and r
plot_output_correlation(neuron_935_output, df["r"], "Neuron 935", "r")
plot_output_correlation(neuron_935_output, df["eta"], "Neuron 935", "eta")
plot_output_correlation(neuron_935_output, df["rho"], "Neuron 935", "rho")
plot_output_correlation(neuron_935_output, df["theta"], "Neuron 935", "theta")
plot_output_correlation(neuron_935_output, df["phi"], "Neuron 935", "phi")

# Same for neuron 895 in layer 4
neuron_895_weights = neuron_weights_4[895]
neuron_895_biases = neuron_biases_4[895]

# Do input*weights + biases
neuron_895_output = activations_3.T @ neuron_895_weights.T + neuron_895_biases

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
