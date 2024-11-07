import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
csv_dir = Path("csv")
csv_files = list(csv_dir.glob("*.csv"))

csv_file = csv_files[1]
csv_file = csv_dir / "neurons_perm_event000000101.csv"
df = pd.read_csv(csv_file)
print(df.head())
exit()

# Remove reference neuron filter
reference_neuron_filter = df["neuron"] == -1

# # Plot "loss" vs "f1"
# plt.scatter(df["loss"], df["f1"])
# plt.xlabel("loss")
# plt.ylabel("f1")
# plt.show()
# plt.close()

# # Plot "loss" vs "signal_eff"
# plt.scatter(df["loss"], df["signal_eff"])
# plt.xlabel("loss")
# plt.ylabel("signal_eff")
# plt.show()
# plt.close()

# # Plot the data for each layer
# layer_names = df["layer"].unique()

# for layer_name in layer_names:
#     layer_df = df[df["layer"] == layer_name]

#     # Drop the "layer" column
#     layer_df = layer_df.drop(columns=["layer"])
#     # print(layer_df.head())
#     # Get back the reference as neuron numbered "-1"
#     metric = "f1"
#     metric = "loss"
#     reference_perf = layer_df[layer_df["neuron"] == -1][metric].values[0]
#     layer_df = layer_df[layer_df["neuron"] != -1]

#     # Get the mean and std with respect to "permutation" column
#     layer_df = layer_df.groupby("neuron").mean().reset_index()
#     # layer_df = layer_df.groupby("neuron").agg({metric: ["mean", "std"]})
#     print(layer_df.head())

#     layer_df[metric] = layer_df[metric] - reference_perf

#     plt.hist(
#         layer_df[metric].values,
#         bins=np.arange(
#             min(layer_df[metric].values),
#             max(layer_df[metric].values),
#             0.0001,
#         ),
#         alpha=0.5,
#     )
#     plt.title(f"Layer: {layer_name}")
#     plt.show()
#     plt.close()
#     # break

# Assume df is already defined
mean_df = df.groupby(["neuron", "layer"]).mean().reset_index()
print(mean_df.head())
reference_perf = mean_df.loc[mean_df["neuron"] == -1, "loss"].iloc[0]
# Remove reference neuron
mean_df = mean_df[mean_df["neuron"] != -1]
print(mean_df.head())

# Plot the full "loss" histogram with reference line
plt.hist(mean_df["loss"], bins=100, alpha=0.5)
plt.axvline(reference_perf, color="r", linestyle="--")
plt.yscale("log")
plt.legend(["Reference", "Neurons"])
plt.xlabel("Loss")
plt.ylabel("Count")
plt.title("Permutation loss Histogram for All Neurons")
plt.grid(True)
plt.show()


plot_dir = Path("plot")
plot_dir.mkdir(exist_ok=True)


# Assume df is already defined
mean_df = df.groupby(["neuron", "layer"]).mean().reset_index()
metrics = [
    col for col in mean_df.columns if col not in ["neuron", "layer", "permutation"]
]
reference_neuron = mean_df[mean_df["neuron"] == -1]
# Remove reference neuron
mean_df = mean_df[mean_df["neuron"] != -1]

# for metric in metrics:
#     reference_perf = reference_neuron[metric].iloc[0]
#     print(f"Reference {metric}: {reference_perf}")
#     # Plot the full metric histogram with reference line
#     plt.hist(mean_df[metric], bins=100, alpha=0.5)
#     plt.axvline(reference_perf, color="r", linestyle="--")
#     plt.yscale("log")
#     plt.legend(["Reference", "Neurons"])
#     plt.xlabel(metric.capitalize())
#     plt.ylabel("Count")
#     plt.title(f"Permutation {metric.capitalize()} Histogram for All Neurons")
#     plt.grid(True)
#     # Save the plot
#     plt.savefig(plot_dir / f"{metric}_histogram.pdf")
#     plt.show()

#     # Filter out reference neuron and calculate uncertainty
#     neuron_uncertainty = (
#         df[df["neuron"] != -1].groupby(["layer", "neuron"]).std().reset_index()
#     )

#     # Plot metric vs "layer" with uncertainty
#     for (layer, neuron), group in mean_df.groupby(["layer", "neuron"]):
#         uncertainty = neuron_uncertainty.query("layer == @layer and neuron == @neuron")
#         metric_uncertainty = uncertainty[metric].iloc[0] if not uncertainty.empty else 0
#         group["layer"] = group["layer"] + np.random.uniform(-0.2, 0.2, len(group))
#         plt.errorbar(group[metric], group["layer"], xerr=metric_uncertainty, fmt="o")

#     plt.axvline(reference_perf, color="r", linestyle="--")
#     plt.ylabel("Layer")
#     plt.xlabel(metric.capitalize())
#     plt.title(f"Permutation {metric.capitalize()} vs Layer (10 permutations)")
#     plt.grid(True)
#     # Legend for reference line
#     plt.legend(["Reference", "Neurons"])
#     # Save the plot
#     plt.savefig(plot_dir / f"{metric}_vs_layer.pdf")
#     plt.show()


# Function to select specific points in a certain metric and show where they are for the other metrics
def select_points_and_plot(df, selected_metric, threshold):
    selected_points = mean_df[mean_df[selected_metric] > threshold]
    # other_metrics = [m for m in metrics if m != selected_metric]
    other_metrics = [m for m in metrics]
    print(selected_points)

    for metric in other_metrics:
        plt.scatter(mean_df["neuron"], mean_df[metric], alpha=0.5, label="All Neurons")
        plt.scatter(
            selected_points["neuron"],
            selected_points[metric],
            color="r",
            label="Selected Neurons",
        )
        plt.xlabel("Neuron")
        plt.ylabel(metric.capitalize())
        plt.title(
            f'{metric.capitalize()} for Selected Points ("{selected_metric}" > {threshold})'
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(
            plot_dir / f"{metric}_selected_points_{threshold}_{selected_metric}.pdf"
        )
        plt.show()


# Example usage: select points in a certain metric and visualize
# You can change the metric and threshold value as needed
select_points_and_plot(mean_df, selected_metric="loss", threshold=0.0174)
# select_points_and_plot(mean_df, selected_metric="loss", threshold=0.0177)
