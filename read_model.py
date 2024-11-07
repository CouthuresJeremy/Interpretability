import torch

model = torch.load(
    "model/best--f1=0.313180-epoch=89.ckpt", map_location=torch.device("cpu")
)
# print(model)

print(model.keys())
print(model["state_dict"].keys())

state_dict = model["state_dict"]
state_dict_keys = list(state_dict.keys())

print(state_dict[state_dict_keys[0]])
print(state_dict[state_dict_keys[0]].shape)
print(state_dict[state_dict_keys[2]])
print(state_dict[state_dict_keys[2]].shape)
print(state_dict[state_dict_keys[-2]].shape)

print(model["hyper_parameters"])

# Get weights and biases for neuron 86 in layer 1
print(state_dict[state_dict_keys[0]][86])
print(state_dict[state_dict_keys[1]][86])

# Same for neuron 44 in layer 1
print(state_dict[state_dict_keys[0]][44])
print(state_dict[state_dict_keys[1]][44])

# for key in activations:
#     print(key, activations[key].shape)

# keys = list(activations)
# print(activations[keys[0]].T[0])

# import matplotlib.pyplot as plt

# import numpy as np

# plt.hist(list(activations[keys[0]].T[0]), bins=np.arange(-1.3, 1.5, 0.025), alpha=0.5)
# plt.hist(list(activations[keys[0]].T[10]), bins=np.arange(-1.3, 3.5, 0.025), alpha=0.5)
# plt.show()
