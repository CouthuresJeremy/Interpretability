import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from load_data import (
    match_input_data,
    load_csv_data,
    load_event_data,
    load_event_activations,
    load_model,
)

# Command-line arguments
import argparse

# Create the parser
parser = argparse.ArgumentParser(
    description="Compute mutual information for each layer"
)

# Command-line arguments: layer, start_neuron, end_neuron
parser.add_argument(
    "--layer",
    type=int,
    default=1,
    help="Layer index for which to compute mutual information",
)
parser.add_argument(
    "--start_neuron",
    type=int,
    default=0,
    help="Start neuron index for which to compute mutual information",
)
parser.add_argument(
    "--end_neuron",
    type=int,
    default=None,
    help="End neuron index for which to compute mutual information",
)

# Parse the arguments
args = parser.parse_args()

print(f"{args = }")

# Layer index
layer_index = args.layer

# Start and end neuron indices
start_neuron = args.start_neuron
end_neuron = args.end_neuron


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

    # Verify that the assignment is correct
    verify_activation_assignement(input_df, duplicated_activations_1)

    for layer in neuron_activations:
        # Add the duplicated activations to the neuron activations
        neuron_activations_layer_df = pd.DataFrame(neuron_activations[layer].T)
        duplicated_activations = (
            neuron_activations_layer_df.iloc[input_df_keys].reset_index(drop=True).T
        ).to_numpy()

        # print(f"{duplicated_activations.shape = }")

        neuron_activations[layer] = torch.tensor(duplicated_activations)
        # print(neuron_activations[layer].shape)

    return neuron_activations, input_df


# Neuron Output Calculation Function
def calculate_neuron_output(df, weights, biases):
    return df.to_numpy() @ weights.T + biases


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

    input_df, neuron_activations = remove_invalid_match(input_df, neuron_activations)

    input_df = match_input_data(
        truth_particles, event_id=event, load_data=False, all_data=True
    )

    neuron_activations, input_df = handle_shared_hits(input_df, neuron_activations)
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

# Save computed activation
layer = 1
model_state_dict = load_model()
neurons_weights, neurons_biases = get_layer_parameters(model_state_dict, layer)

# Calculate the neuron's output for each hit
hit_coordinates = all_input_df[["r", "phi", "z"]]

from load_data import scale_data

# Scale the hit coordinates
hit_coordinates = scale_data(hit_coordinates, scales=[1 / 1000, 1 / 3.14, 1 / 1000])

# Calculate the neuron outputs
neuron_outputs = calculate_neuron_output(
    hit_coordinates, neurons_weights, neurons_biases
)

verify_activation_assignement(
    all_input_df, all_neuron_activations[list(all_neuron_activations)[0]].numpy()
)

neuron_activations = all_neuron_activations
input_df = all_input_df

# Map particle_id to [0..N]
particle_id_map = {
    particle_id: i for i, particle_id in enumerate(input_df["particle_id"].unique())
}
input_df["particle_id_mapped"] = input_df["particle_id"].map(particle_id_map)

keys = list(neuron_activations)
verify_activation_assignement(all_input_df, all_neuron_activations[keys[0]].numpy())
event = 0

# Remove non-continuous features
df_continuous = input_df.select_dtypes(include=[np.number])

# Add 3 random variables for each hit
# uniform, normal, poisson
df_continuous["uniform_hit"] = np.random.uniform(0, 1, size=(df_continuous.shape[0], 1))
df_continuous["normal_hit"] = np.random.normal(0, 1, size=(df_continuous.shape[0], 1))
df_continuous["poisson_hit"] = np.random.poisson(1, size=(df_continuous.shape[0], 1))


# Compute mutual information for all neuron outputs in a layer and save them into a CSV file
def compute_layer_single_mutual_information(
    event,
    neuron_activations,
    df_continuous,
    layer_index,
    layer,
    start_neuron=0,
    end_neuron=None,
):
    """
    Compute mutual information of single neurons in a layer and save them into a CSV file.
    """

    # Get all neuron outputs of the current layer as a 2D array (shape: num_samples x num_neurons)
    layer_outputs = np.array(
        neuron_activations[layer]
    ).T  # Transpose to match mutual_info_regression input format

    print(f"Computing mutual information for Layer {layer_index}")

    if end_neuron is None:
        end_neuron = layer_outputs.shape[1] - 1

    assert end_neuron < layer_outputs.shape[1], f"{end_neuron = }"

    output_dir = Path("conditional_entropy")
    output_dir.mkdir(exist_ok=True)

    # File path for the current layer's CSV
    layer_file_path = f"mutual_information_event{event:09d}_layer{layer_index}.csv"

    if end_neuron - start_neuron != layer_outputs.shape[1] - 1:
        layer_file_path = layer_file_path.replace(
            ".csv", f"_range{start_neuron}-{end_neuron}.csv"
        )

    layer_file_path = output_dir / layer_file_path

    # Initialize or load existing data for the layer
    mutual_information_df_layer, processed_neurons = load_processed_neurons(
        layer_index, layer_file_path
    )

    assert all(
        start_neuron <= processed_neurons[i] <= end_neuron
        for i in range(len(processed_neurons))
    ), f"Processed neurons are not in range {start_neuron}-{end_neuron}: {processed_neurons = }"

    for neuron_idx in range(start_neuron, end_neuron + 1):
        if neuron_idx in processed_neurons:
            continue  # Skip already processed neurons

        print(f"Computing mutual information for Neuron {neuron_idx}")

        discrete_features_bool = np.array(
            [
                (
                    df_continuous[feature].abs()
                    == df_continuous[feature].abs().astype(int)
                ).all()
                for feature in df_continuous.columns
            ]
        )
        discrete_features_indices = np.where(discrete_features_bool)[0]

        mutual_information_values = mutual_info_regression(
            df_continuous.to_numpy(),
            layer_outputs[:, neuron_idx],
            random_state=42,
            discrete_features=discrete_features_indices,
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
    return mutual_information_df_layer


def load_processed_neurons(layer_index, layer_file_path):
    if os.path.exists(layer_file_path):
        mutual_information_df_layer = pd.read_csv(layer_file_path)
        processed_neurons = mutual_information_df_layer["neuron"].tolist()
        print(
            f"Found existing file for Layer {layer_index}. Resuming from Neuron {len(processed_neurons)}"
        )
    else:
        mutual_information_df_layer = pd.DataFrame()
        processed_neurons = []
    return mutual_information_df_layer, processed_neurons


del df_continuous["status"]
del df_continuous["hit_id"]

layer_names = list(neuron_activations.keys())
layer = layer_names[layer_index - 1]
print(f"Layer: {layer}")
compute_layer_single_mutual_information(
    event,
    neuron_activations,
    df_continuous,
    layer_index,
    layer,
    start_neuron=start_neuron,
    end_neuron=end_neuron,
)
