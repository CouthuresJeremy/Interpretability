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


# Convert r, phi and z to float32 for the matching
def match_input_data(truth_particles):
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
        "data/event000000101-truth-particles-unscaled.csv", index=False
    )

    # Reload the unscaled truth particles
    truth_particles_unscaled = pd.read_csv(
        "data/event000000101-truth-particles-unscaled.csv"
    )

    # Rescale the truth particles
    truth_particles_unscaled["r"] = truth_particles_unscaled["r"] * 1000
    truth_particles_unscaled["phi"] = truth_particles_unscaled["phi"] * 3.14
    truth_particles_unscaled["z"] = truth_particles_unscaled["z"] * 1000

    # Load the input data
    df = load_csv_data(file_name="input_data_event000000101.csv", directory="csv")
    df.columns = ["r", "phi", "z"]
    # print(df.head())
    print(df.shape)

    # Scale the input data
    df_scaled = scale_data(df, scales=[1000, 3.14, 1000])

    # Check for (r, phi, z) duplicates in the input data
    print(df_scaled.duplicated(subset=["r", "phi", "z"]).sum())
    # Check for (r, phi, z) duplicates in the truth
    print(truth_particles.duplicated(subset=["x", "y", "z"]).sum())

    # Try to match (r, phi, z) from the truth to the input data
    df_scaled = df_scaled.merge(
        truth_particles_unscaled,
        on=["r", "phi", "z"],
        suffixes=("", "_truth"),
        validate="one_to_many",
    )

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

    # Load the file if it exists
    mathed_file = Path("input_data_event000000101_matched.csv")
    if mathed_file.exists():
        df_scaled = pd.read_csv(mathed_file)
    else:
        # Save the matched input data
        df_scaled.to_csv(mathed_file, index=False)

    # Should be equal
    assert (
        df_scaled.shape[0] <= truth_particles.shape[0]
    ), f"{df_scaled.shape = } {truth_particles.shape = }"

    return df_scaled
