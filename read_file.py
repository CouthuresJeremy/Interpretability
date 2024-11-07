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
exit()
keys = list(activations)
layer_1_name = keys[0]

# Transpose the activation to get the neuron distributions
neuron_activations = {
    layer_name: layer_activations.T
    for layer_name, layer_activations in activations.items()
}
print(activations[keys[0]].T[0])

import matplotlib.pyplot as plt

import numpy as np

layer_1_neurons = neuron_activations[layer_1_name]

plt.hist(
    list(layer_1_neurons[44]),
    bins=np.arange(-5, 5, 0.025),
    alpha=0.5,
    label="44",
)
plt.hist(
    list(layer_1_neurons[86]),
    bins=np.arange(-5, 5, 0.025),
    alpha=0.5,
    label="86",
)

plt.hist(
    list(layer_1_neurons[0]),
    bins=np.arange(-5, 5, 0.025),
    alpha=0.5,
    label="0",
)
plt.legend()
plt.grid(True)
plt.show()

plot_dir = Path("plot")
plot_dir.mkdir(exist_ok=True)

plt.hist(
    list(activations[layer_1_name].T[44]),
    bins=np.arange(-5, 5, 0.025),
    alpha=0.5,
    label="44",
)
plt.hist(
    list(activations[layer_1_name].T[86]),
    bins=np.arange(-5, 5, 0.025),
    alpha=0.5,
    label="86",
)
# Plot all other neurons
for i in range(len(activations[layer_1_name].T)):
    if i in [44, 86]:
        continue

    # plt.hist(
    #     list(activations[layer_1_name].T[i]),
    #     bins=np.arange(-5, 5, 0.025),
    #     alpha=0.5,
    # )
    # Find the highest y among the distributions

plt.legend(["44", "86"])
plt.xlabel("Activation")
plt.ylabel("Count")
plt.title(f"{layer_1_name} Neurons Activation Histogram")
plt.grid(True)
plt.savefig(plot_dir / f"{layer_1_name}_neurons_histogram.pdf")
plt.show()

layer_4_name = keys[3]
plt.hist(
    list(activations[layer_4_name].T[935]),
    bins=np.arange(-10, 30, 0.025),
    alpha=0.5,
    label="935",
)

plt.legend(["935"])
plt.xlabel("Activation")
plt.ylabel("Count")
plt.title(f"{layer_4_name} Neurons Activation Histogram")
plt.grid(True)
plt.savefig(plot_dir / f"{layer_4_name}_neurons_histogram.pdf")
plt.show()
exit()

high_y = 0
high_neuron = 0
# Find the highest y among the distributions
for i in range(len(layer_1_neurons)):
    counts, bins, _ = plt.hist(
        list(layer_1_neurons[i]),
        bins=np.arange(-5, 5, 0.025),
        alpha=0.5,
    )
    max_y = max(counts)
    if max_y > high_y:
        high_y = max_y
        high_neuron = i
    # plt.close()

plt.legend()
plt.grid(True)
plt.show()


plt.hist(
    list(layer_1_neurons[high_neuron]),
    bins=np.arange(-5, 5, 0.025),
    alpha=0.5,
    label=f"{high_neuron}",
)
plt.legend()
plt.grid(True)
plt.show()
