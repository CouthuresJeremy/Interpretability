import torch

model = torch.load(
    "model/best--f1=0.313180-epoch=89.ckpt", map_location=torch.device("cpu")
)
# print(model)

print(model.keys())
print(model["state_dict"].keys())

state_dict = model["state_dict"]
state_dict_keys = list(state_dict.keys())

# Read activations
event = 100
event = 101
activations = torch.load(
    f"activations/activations_event{event:09d}.pt", map_location=torch.device("cpu")
)

keys = list(activations)

# Transpose the activation to get the neuron distributions
neuron_activations = {
    layer_name: layer_activations.T
    for layer_name, layer_activations in activations.items()
}

import matplotlib.pyplot as plt
import numpy as np

for i, (k, weights) in enumerate(state_dict.items()):
    # print(weights.shape)
    if len(weights.shape) != 2:
        continue
    biases = state_dict[state_dict_keys[i + 1]]

    # Compute the expected value of the neuron activations
    layer_name = int(k.split(".")[1]) - 1
    if layer_name < 0:
        # Need the inputs
        continue

    # if layer_name < 11:
    #     continue

    print(layer_name)
    neuron_activations_layer = neuron_activations[keys[layer_name]]

    # # Get the sign of the weights
    # weights_sign = weights.sign()
    # print(weights_sign.shape)
    # print(neuron_activations_layer.shape)

    # # Multiply the activations by the sign of the weights
    # neuron_activations_layer_sign = neuron_activations_layer.unsqueeze(
    #     0
    # ) * weights_sign.unsqueeze(2)

    # # Keep only the maximum activations
    # maximum_signed_activations = neuron_activations_layer_sign.max(dim=1).values
    # minimum_signed_activations = neuron_activations_layer_sign.min(dim=1).values
    # print(maximum_signed_activations.shape)
    # print(minimum_signed_activations.shape)
    # exit()

    # Reduce the number of samples
    if weights.shape[0] > 1000:
        neuron_activations_layer = neuron_activations_layer[:, :1000]

    # Compute the information pass from the connection
    # Information pass is of shape (n_neurons_layer, n_neurons_layer-1, n_samples)
    information_pass = neuron_activations_layer.unsqueeze(0) * weights.unsqueeze(2)

    # Compute the expected information pass
    # expected_information_pass = information_pass.mean(dim=2)
    min_information_pass = information_pass.min(dim=2).values

    # Get the indices of the maximum information passed in absolute value
    max_information_pass_indices = information_pass.abs().argmax(dim=2)
    # Get the maximum signed information passed
    max_information_pass = torch.gather(
        information_pass, 2, max_information_pass_indices.unsqueeze(2)
    ).squeeze(2)

    # Make sure the correct values are selected
    assert torch.all(
        max_information_pass.abs() == information_pass.abs().max(dim=2).values
    )

    # Normalize the information pass
    # expected_information_pass_normed = (
    #     expected_information_pass / expected_information_pass.abs().sum()
    # )

    # Determine the maximum relative impact of the weights to the neuron sum
    # Consider the case where all other neurons give the minimum activation
    # and the neuron of interest give its absolute maximum activation
    minimum_information_sum = min_information_pass.sum(dim=1) + biases
    # Replace the minimum information of the neuron of interest with the maximum
    # activation in the sum
    replaced_min_information_pass = minimum_information_sum.unsqueeze(1) + (
        max_information_pass - min_information_pass
    )
    max_information_pass_min_normed = (
        max_information_pass / replaced_min_information_pass
    )

    # Plot the information pass
    # for neuron_index in range(weights.shape[0]):
    #     plt.scatter(
    #         expected_information_pass[neuron_index],
    #         expected_information_pass_normed[neuron_index],
    #     )
    # plt.show()

    # Plot the max information pass
    for neuron_index in range(weights.shape[0]):
        plt.scatter(
            max_information_pass[neuron_index],
            max_information_pass_min_normed[neuron_index].abs() * 100,
        )
    plt.ylabel("Maximum absolute relative weight impact [%]")
    plt.xlabel("Maximum absolute information pass")
    plt.show()

    if layer_name == 2:
        # Plot the max information pass distribution for neuron 935 and 0
        max_info_neuron_935 = max_information_pass_min_normed[935].abs().numpy()
        plt.hist(
            max_info_neuron_935,
            bins=np.arange(
                max_info_neuron_935.min(), max_info_neuron_935.max(), 0.0001
            ),
            alpha=0.5,
            label="935",
            color="k",
        )
        # Plot other neuron
        # for i in range(len(max_information_pass_min_normed)):
        #     if i % 20 == 0:
        #         print(f"Neuron {i}")
        #     if i == 935:
        #         continue
        #     max_info_neuron = max_information_pass_min_normed[i].abs().numpy()
        #     plt.hist(
        #         max_info_neuron,
        #         bins=np.arange(max_info_neuron.min(), max_info_neuron.max(), 0.0001),
        #         alpha=0.5,
        #     )
        # plt.legend(["935", "others"])
        plt.show()

        # Plot the distribution of the sum of the information pass for each neuron
        information_sum = max_information_pass_min_normed.abs().sum(dim=1)
        plt.hist(
            information_sum,
            bins=1000,
            alpha=0.5,
        )
        # Show the sum of the information pass for neuron 935
        plt.axvline(information_sum[935], color="k")
        plt.show()

        exit()
