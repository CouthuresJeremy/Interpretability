import pandas as pd
from pathlib import Path

# Change to the directory containing the CSV files
data_dir = Path("./results")

# Specify the output directory
output_dir = Path("./conditional_entropy")

# Define the layers you want to process
# layers = [1, 2, 3, 4, 5, 6]
# layers = [4, 5, 6]
layers = [7]

# Process each layer individually
for layer in layers:
    # Pattern to match all CSV files for the given layer
    pattern = f"mutual_information_event000000000_layer{layer}_range*.csv"
    files = data_dir.glob(pattern)

    # List to hold dataframes for the current layer
    dataframes = []

    for file in files:
        # Read each CSV file into a DataFrame
        df = pd.read_csv(file)
        dataframes.append(df)

    if dataframes:
        # Concatenate all DataFrames into one
        combined_df = pd.concat(dataframes, ignore_index=True)
        # Sort the DataFrame by the 'neuron' column
        combined_df_sorted = combined_df.sort_values(by="neuron")

        # Save the combined and sorted DataFrame to a new CSV file
        output_filename = (
            output_dir / f"combined_mutual_information_event000000000_layer{layer}.csv"
        )
        combined_df_sorted.to_csv(output_filename, index=False)
        print(f"Layer {layer}: Combined CSV saved as {output_filename}")
        print(f"Number of neurons: {combined_df_sorted['neuron'].nunique()}")
        # Assert no duplicate neurons
        assert combined_df_sorted["neuron"].nunique() == len(combined_df_sorted)
    else:
        print(f"No files found for layer {layer}.")
