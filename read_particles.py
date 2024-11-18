import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

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


# Calculate entropy between two variables
def calculate_entropy(df, feature1, feature2):
    # Calculate the joint probability distribution
    joint_prob = df.groupby([feature1, feature2]).size() / len(df)
    # Calculate the entropy
    entropy = -joint_prob * np.log(joint_prob)
    return entropy.sum()


from sklearn.metrics import mutual_info_score

from sklearn.neighbors import KernelDensity


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

# Load the particles
particles = load_csv_data(
    file_name="event000000101-hard-cut-particles.csv", directory="csv"
)
# df.columns = ["r", "phi", "z"]
print(particles.head())

# Load the truth
truth = load_csv_data(file_name="event000000101-hard-cut-truth.csv", directory="csv")
print(truth.head())


# Get particles corresponding to the truth
truth_particle_ids = truth["particle_id"].unique()
truth_particles = particles[
    particles["particle_id"].isin(truth_particle_ids)
    | particles["particle_id"].isin(truth["particle_id_1"])
    | particles["particle_id"].isin(truth["particle_id_2"])
]
print(particles.shape)
print(truth_particles.shape)

print(truth_particles.head())

# Plot the distribution of the particles
# plot_feature_distributions(particles)


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


# plot_feature_distribution(particles, truth_particles)

# Assign particle information to the truth
truth_particles = truth.merge(
    particles, left_on="particle_id", right_on="particle_id", suffixes=("", "_particle")
)
# truth_particles = truth_particles.merge(
#     particles,
#     left_on="particle_id_1",
#     right_on="particle_id",
#     suffixes=("", "_particle_1"),
# )
# truth_particles = truth_particles.merge(
#     particles,
#     left_on="particle_id_2",
#     right_on="particle_id",
#     suffixes=("", "_particle_2"),
# )
print(truth_particles.head())
print(truth_particles.columns)
print(truth_particles.shape)

# Compute r, phi, z for the truth particles
truth_particles["r"] = np.sqrt(truth_particles["x"] ** 2 + truth_particles["y"] ** 2)
truth_particles["phi"] = np.arctan2(truth_particles["y"], truth_particles["x"])
truth_particles["z"] = truth_particles["z"]

print(truth_particles.head())


# Load the input data
df = load_csv_data(file_name="input_data_event000000101.csv", directory="csv")
df.columns = ["r", "phi", "z"]
print(df.head())
print(df.shape)

# Scale the input data
df_scaled = scale_data(df, scales=[1000, 3.14, 1000])
print(df_scaled.head())

# Check for (r, phi, z) duplicates in the input data
print(df_scaled.duplicated(subset=["r", "phi", "z"]).sum())
# Check for (r, phi, z) duplicates in the truth
print(truth_particles.duplicated(subset=["x", "y", "z"]).sum())

# Try to match (r, phi, z) from the truth to the input data
# df_scaled = df_scaled.merge(
#     truth_particles,
#     on=["r", "phi", "z"],
#     suffixes=("", "_truth"),
#     validate="one_to_one",
# )
# Do the match manually
# df_scaled = df_scaled.merge(
#     truth_particles,
#     on=["r", "phi", "z"],
#     suffixes=("", "_truth"),
#     how="left",
# )
# Compare equality of first line of the truth and the input data
print(truth_particles.iloc[0][["r", "phi", "z"]])
print(df_scaled.iloc[0])
print(truth_particles.iloc[0][["r", "phi", "z"]] == df_scaled.iloc[0])
# print dtype of the columns
print(truth_particles[["r", "phi", "z"]].dtypes)
print(df_scaled[["r", "phi", "z"]].dtypes)
# print the difference between the two dataframes
print(truth_particles[["r", "phi", "z"]] - df_scaled[["r", "phi", "z"]])
# Consider the difference to be zero if it is less than 1e-6
print((truth_particles[["r", "phi", "z"]] - df_scaled[["r", "phi", "z"]]).abs() < 1e-6)

# Load the file if it exists
mathed_file = Path("input_data_event000000101_matched.csv")
if mathed_file.exists():
    df_scaled = pd.read_csv("input_data_event000000101_matched.csv")
else:
    # Consider matching if all the differences are less than 1e-6
    for hit in df_scaled.index:
        # Find the matching truth hit for the hit with the same (r, phi, z)
        matching_truth_hit = truth_particles[
            ((truth_particles["r"] - df_scaled.loc[hit, "r"]).abs() < 1e-3)
            & ((truth_particles["phi"] - df_scaled.loc[hit, "phi"]).abs() < 1e-6)
            & ((truth_particles["z"] - df_scaled.loc[hit, "z"]).abs() < 1e-5)
        ]
        # If there is a matching truth hit, assign the truth hit to the hit
        if not matching_truth_hit.empty:
            # Add particle information to the hit
            df_scaled.loc[hit, "particle_id"] = matching_truth_hit[
                "particle_id"
            ].values[0]
            for feature in truth_particles.columns:
                if feature not in df.columns:
                    df_scaled.loc[hit, feature] = matching_truth_hit[feature].values[0]
        # If there is no matching truth hit, assign -1 to the hit
        assert (
            len(matching_truth_hit) >= 1
        ), f"len(matching_truth_hit): {len(matching_truth_hit)}, hit: {hit}, matching_truth_hit: {matching_truth_hit}"

    # Save the matched input data
    df_scaled.to_csv(mathed_file, index=False)

print(df_scaled.head())

# Add various coordinate transformations
df_scaled = add_coordinate_transformations(df_scaled)

df = df_scaled

from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

n_hits = df.shape[0]
# print(df[["eta", "phi"]].to_numpy().reshape(n_hits, 2).shape)
# print(df[["eta", "phi"]].to_numpy().reshape(n_hits, 2))

# print(mutual_information(df, "eta", "eta"))
# print(calculate_entropy(df, "eta", "eta"))
# print(conditional_entropy(df["eta"].to_numpy(), df["eta"].to_numpy()))
# print(
#     mutual_info_regression(
#         df["eta"].to_numpy().reshape(-1, 1),
#         df["eta"].to_numpy(),
#         random_state=42,
#     )
# )
# print(conditional_entropy(df["phi"].to_numpy(), df["eta"].to_numpy()))
# print(
#     mutual_info_regression(
#         df["phi"].to_numpy().reshape(-1, 1),
#         df["eta"].to_numpy(),
#         random_state=42,
#     )
# )
# print(
#     mutual_info_regression(
#         df[["eta", "phi"]].to_numpy().reshape(n_hits, 2),
#         df["eta"].to_numpy(),
#         random_state=42,
#     )
# )
# exit()

# Plot the scaled data for feature "r" vs "z"
# plt.scatter(df_scaled["z"], df_scaled["r"])
# plt.xlabel("z")
# plt.ylabel("r")
# plt.show()

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
layer_4_name = keys[3]

# Transpose the activation to get the neuron distributions
neuron_activations = {
    layer_name: layer_activations.T
    for layer_name, layer_activations in activations.items()
}

activations_1 = neuron_activations[layer_1_name].numpy()
activations_3 = neuron_activations[layer_3_name].numpy()
activations_4 = neuron_activations[layer_4_name].numpy()
print(activations_3.shape)
# print(neuron_935_weights.T.shape)

# Do input*weights + biases
# neuron_935_output = activations_3.T @ neuron_935_weights.T + neuron_935_biases

neuron_86_output = activations_1[86]
neuron_44_output = activations_1[44]
neuron_935_output = activations_4[935]

for feature in df.columns:
    feature_values = df[feature].to_numpy()
    # If the feature is not continuous, skip
    if not np.issubdtype(feature_values.dtype, np.number):
        print(f"Feature {feature} is not continuous")
        continue
    print(f"Feature: {feature}")
    print(
        f"Conditional Entropy: {conditional_entropy(neuron_935_output, feature_values)}"
    )
print(
    mutual_info_regression(
        # df.to_numpy().reshape(-1, 1),
        df.to_numpy(),
        neuron_935_output,
        random_state=42,
    )
)
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

# Map particle_id to [0..N]
particle_id_map = {
    particle_id: i for i, particle_id in enumerate(df["particle_id"].unique())
}
df["particle_id_mapped"] = df["particle_id"].map(particle_id_map)

exit()

# Clustering
for feature in df.columns:
    plt.scatter(neuron_935_output, df[feature], c=df["particle_id_mapped"])
    plt.ylabel(feature)
    plt.xlabel("Neuron 935")
    plt.show()


plot_scatter_with_color(df, "eta", "pt", neuron_935_output, "eta", "pt")
exit()

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
