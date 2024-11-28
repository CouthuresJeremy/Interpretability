# Plot random variables: x, y = x^2

import numpy as np
import matplotlib.pyplot as plt

# Generate random variables
x = np.random.rand(1000)
# Convert x range from [0, 1] to [-1, 1]
x = 2 * x - 1
y = x**2

# Plot random variables
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Random Variables: x, y = x^2")
plt.savefig("random_variables.png")
plt.show()

# Same plot but with black vertical lines every 0.1 with high width
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Random Variables: x, y = x^2")
for step in np.arange(-1, 1 + 0.05, 0.1):
    plt.axvline(x=step, color="black", linewidth=2)
plt.savefig("random_variables_black_lines.png")
# Get the axis limits
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
plt.show()

# Plot corresonding y for 5 different x values, one figure per x value
for i, x_value in enumerate([-0.5, -0.25, 0, 0.25, 0.5]):
    plt.scatter(x_value, x_value**2, color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Conditional distribution: y = x^2, x = {x_value}")
    # Use same scales for x and y axis than the previous plot
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(f"random_variables_x_{i}.png")
    plt.show()
