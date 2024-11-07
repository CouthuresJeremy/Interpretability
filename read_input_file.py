import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
csv_dir = Path("csv")
csv_files = list(csv_dir.glob("*.csv"))

csv_file = csv_files[1]
csv_file = csv_dir / "input_data_event000000101.csv"
df = pd.read_csv(csv_file)
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

# Plot the scaled data for feature "r" vs "z"
# plt.scatter(df_scaled["z"], df_scaled["r"])
# plt.xlabel("z")
# plt.ylabel("r")
# plt.show()

# Read model
import torch

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
plt.show()

# Plot isocurves of the output of neuron 86 with a lot of points
plt.tricontourf(df_scaled["z"], df_scaled["r"], neuron_86_output, levels=20)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
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
plt.show()

# Plot isocurves of the output of neuron 44 with a lot of points
plt.tricontourf(df_scaled["z"], df_scaled["r"], neuron_44_output, levels=20)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.show()

# Plot the output of neuron 44 vs the output of neuron 86
plt.scatter(neuron_86_output, neuron_44_output)
plt.xlabel("Neuron 86")
plt.ylabel("Neuron 44")
plt.show()

# Same for neuron 935 in layer 4
neuron_weights_4 = state_dict[state_dict_keys[2]].numpy()
neuron_biases_4 = state_dict[state_dict_keys[3]].numpy()

neuron_935_weights = neuron_weights_4[935]
neuron_935_biases = neuron_biases_4[935]

# Read activations
import torch
from pathlib import Path

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
plt.show()

# Plot isocurves of the output of neuron 935 with a lot of points
plt.tricontourf(df_scaled["z"], df_scaled["r"], neuron_935_output, levels=20)
plt.xlabel("z")
plt.ylabel("r")
plt.colorbar()
plt.show()

# Plot the output of neuron 935 vs the output of neuron 86
plt.scatter(neuron_86_output, neuron_935_output)
plt.xlabel("Neuron 86")
plt.ylabel("Neuron 935")
plt.show()

# Plot the output of neuron 935 vs the output of neuron 44
plt.scatter(neuron_44_output, neuron_935_output)
plt.xlabel("Neuron 44")
plt.ylabel("Neuron 935")
plt.show()

# Cyllindrical coordinates to cartesian
df["x"] = df["r"] * np.cos(df["phi"])
df["y"] = df["r"] * np.sin(df["phi"])

# Cyllindrical coordinates to spherical
df["theta"] = np.arctan2(df["r"], df["z"])
df["rho"] = np.sqrt(df["r"] ** 2 + df["z"] ** 2)

# Calculate eta
df["eta"] = -np.log(np.tan(df["theta"] / 2))

# Plot neuron 935 output vs x, y, z, r, phi, theta, rho
plt.plot(neuron_935_output, df["x"], "o", label="x")
plt.plot(neuron_935_output, df["y"], "o", label="y")
plt.plot(neuron_935_output, df["z"], "o", label="z")
plt.plot(neuron_935_output, df["r"], "o", label="r")
plt.plot(neuron_935_output, df["phi"], "o", label="phi")
plt.plot(neuron_935_output, df["theta"], "o", label="theta")
plt.plot(neuron_935_output, df["rho"], "o", label="rho")
plt.plot(neuron_935_output, df["eta"], "o", label="eta")
plt.legend()
plt.grid(True)
plt.show()
