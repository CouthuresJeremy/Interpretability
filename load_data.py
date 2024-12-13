import pandas as pd
from pathlib import Path
import torch


# Load Model Function
def load_model(model_path="model/best--f1=0.313180-epoch=89.ckpt"):
    model = torch.load(model_path, map_location=torch.device("cpu"))
    return model["state_dict"]


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


def load_event_data(event_id=101, verbose=False, input_data_dir="truth_input_data"):
    # Load the particles
    particles = load_csv_data(
        file_name=f"event{event_id:09d}-hard-cut-particles.csv",
        directory=input_data_dir,
    )

    import numpy as np

    # Add 3 random variables for each particles
    # uniform, normal, poisson
    particles["uniform_particle"] = np.random.uniform(
        0, 1, size=(particles.shape[0], 1)
    )
    particles["normal_particle"] = np.random.normal(0, 1, size=(particles.shape[0], 1))
    particles["poisson_particle"] = np.random.poisson(1, size=(particles.shape[0], 1))

    # Load the truth
    truth = load_csv_data(
        file_name=f"event{event_id:09d}-hard-cut-truth.csv", directory=input_data_dir
    )

    # Count the number of particle_id_1 != particle_id_2
    if verbose:
        print(truth.shape)
        print(truth[truth["particle_id_1"] != truth["particle_id_2"]].shape)

    # Get particles corresponding to the truth
    truth_particle_ids = truth["particle_id"].unique()
    truth_particles = particles[
        particles["particle_id"].isin(truth_particle_ids)
        | particles["particle_id"].isin(truth["particle_id_1"])
        | particles["particle_id"].isin(truth["particle_id_2"])
    ]

    # Assign particle information to the truth
    truth_particles = truth.merge(
        particles,
        left_on="particle_id",
        right_on="particle_id",
        suffixes=("", "_particle"),
    )
    return truth_particles


def load_event_activations(event_id=101, verbose=False):
    activations = torch.load(
        f"activations/activations_event{event_id:09d}.pt",
        map_location=torch.device("cpu"),
    )
    if verbose:
        print(activations)

        for key in activations:
            print(key, activations[key].shape)

        keys = list(activations)
        print(keys)

    # Transpose the activation to get the neuron distributions
    neuron_activations = {
        layer_name: layer_activations.T
        for layer_name, layer_activations in activations.items()
    }
    return neuron_activations


# Convert r, phi and z to float32 for the matching
def match_input_data(truth_particles, load_data=True, event_id=101):
    # Load the file if it exists
    mathed_file = Path(f"input_data_event{event_id:09d}_matched.csv")
    if load_data and mathed_file.exists():
        df_scaled = pd.read_csv(mathed_file)
        return df_scaled

    truth_particles_unscaled = truth_particles.copy()
    truth_particles_unscaled["r"] = truth_particles_unscaled["r"] / 1000
    truth_particles_unscaled["phi"] = truth_particles_unscaled["phi"] / 3.14
    truth_particles_unscaled["z"] = truth_particles_unscaled["z"] / 1000
    # Change data type to float32
    truth_particles_unscaled["r"] = truth_particles_unscaled["r"].astype("float32")
    truth_particles_unscaled["phi"] = truth_particles_unscaled["phi"].astype("float32")
    truth_particles_unscaled["z"] = truth_particles_unscaled["z"].astype("float32")

    # Save the unscaled truth particles to csv
    truth_particles_unscaled.to_csv(
        f"data/event{event_id:09d}-truth-particles-unscaled.csv", index=False
    )

    # Reload the unscaled truth particles
    truth_particles_unscaled = pd.read_csv(
        f"data/event{event_id:09d}-truth-particles-unscaled.csv"
    )

    # Rescale the truth particles
    truth_particles_unscaled["r"] = truth_particles_unscaled["r"] * 1000
    truth_particles_unscaled["phi"] = truth_particles_unscaled["phi"] * 3.14
    truth_particles_unscaled["z"] = truth_particles_unscaled["z"] * 1000

    # Load the input data
    df = load_csv_data(
        file_name=f"input_data_event{event_id:09d}.csv", directory="input_data"
    )
    df.columns = ["r", "phi", "z"]
    # print(df.head())
    print(df.shape)

    # Scale the input data
    df_scaled = scale_data(df, scales=[1000, 3.14, 1000])

    # Check for (r, phi, z) duplicates in the input data
    print(df_scaled.duplicated(subset=["r", "phi", "z"]).sum())
    # Check for (r, phi, z) duplicates in the truth
    print(truth_particles.duplicated(subset=["x", "y", "z"]).sum())

    assert (
        df_scaled.duplicated(subset=["r", "phi", "z"]).sum() == 0
    ), f"{df_scaled.duplicated(subset=['r', 'phi', 'z']).sum() = }"

    print(f"{df_scaled.shape = } {truth_particles_unscaled.shape = }")

    truth_particles_unscaled.drop_duplicates(subset=["r", "phi", "z"], inplace=True)

    print(f"{df_scaled.shape = } {truth_particles_unscaled.shape = }")

    # Try to match (r, phi, z) from the truth to the input data
    df_scaled = df_scaled.merge(
        truth_particles_unscaled,
        how="left",
        on=["r", "phi", "z"],
        suffixes=("", "_truth"),
        # validate="one_to_many",
        indicator=True,
        validate="one_to_one",
    )

    # Check the indicator
    print(df_scaled["_merge"].value_counts())

    # Count duplicated [r, phi, z] in the input data
    print(df_scaled.duplicated(subset=["r", "phi", "z"]).sum())

    # Print hit id in truth that is not in the input data
    print(
        truth_particles_unscaled[
            ~truth_particles_unscaled["hit_id"].isin(df_scaled["hit_id"])
        ].shape
    )

    print(
        truth_particles_unscaled[
            ~truth_particles_unscaled["hit_id"].isin(df_scaled["hit_id"])
        ]
    )

    # Save the matched input data
    df_scaled.to_csv(mathed_file, index=False)

    # Should be equal
    assert (
        df_scaled.shape[0] <= truth_particles.shape[0]
    ), f"{df_scaled.shape = } {truth_particles.shape = }"

    return df_scaled
