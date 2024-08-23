# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:47:41 2024

@author: Nathan
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the scaled tanh function
def scaled_tanh(x, k):
    return np.tanh(k * x)

# Generate some data
x = np.linspace(-5, 5, 100)

# Define different values for k
k_values = [0.5, 1, 2, 5]  # Example scaling factors

# Plot the results
plt.figure(figsize=(10, 6))

for k in k_values:
    y = scaled_tanh(x, k)
    plt.plot(x, y, label=f'k={k}')

plt.title('Scaled Hyperbolic Tangent Function for Various k Values')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the tanh function and its scaled versions
def tanh_standard(x):
    return np.tanh(x)

def tanh_scaled_0_1(x):
    return (np.tanh(x) + 1) / 2

def tanh_scaled_2_5(x):
    return 1.5 * np.tanh(x) + 3.5

# Generate data
x = np.linspace(-5, 5, 100)

# Calculate the function outputs for different ranges
y_standard = tanh_standard(x)        # Output range: [-1, 1]
y_scaled_0_1 = tanh_scaled_0_1(x)    # Output range: [0, 1]
y_scaled_2_5 = tanh_scaled_2_5(x)    # Output range: [2, 5]

# Plot the results
plt.figure(figsize=(12, 8))

plt.plot(x, y_standard, label='Output Range [-1, 1]', color='blue')
plt.plot(x, y_scaled_0_1, label='Output Range [0, 1]', color='green')
plt.plot(x, y_scaled_2_5, label='Output Range [2, 5]', color='red')

plt.title('Tanh Function with Different Output Ranges')
plt.xlabel('Input (x)')
plt.ylabel('Output (f(x))')
plt.legend()
plt.grid(True)
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt

# Define the scaled tanh function with the option to adjust the output range
def scaled_tanh(x, k, a=-1, b=1):
    """
    Scales the tanh function by a factor of k and adjusts the output range.

    Parameters:
    x (numpy array): Input values.
    k (float): Scaling factor for the steepness of the tanh function.
    a (float): Minimum value of the desired output range.
    b (float): Maximum value of the desired output range.

    Returns:
    numpy array: Scaled and shifted tanh values.
    """
    return (b - a) / 2 * np.tanh(k * x) + (b + a) / 2

# Generate some data
x = np.linspace(-5, 5, 100)

# Define different values for k and output ranges
k_values = [0.5, 1, 2, 5]  # Example scaling factors
output_ranges = [(-1, 1), (0, 1), (2, 5)]  # Different output ranges to demonstrate

# Plot the results for each k with different output ranges
plt.figure(figsize=(12, 8))

for k in k_values:
    # for (a, b) in (2,5):
        a = 0
        b = 1
        y = scaled_tanh(x, k, a, b)
        plt.plot(x, y, label=f'k={k}, range=({a}, {b})')

plt.title('Scaled Hyperbolic Tangent Function with Various k Values and Output Ranges')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()


# %%
import numpy as np
import time

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the tanh function (using NumPy's built-in tanh)
def tanh(x):
    return np.tanh(x)

# Generate a large array of random data
x = np.random.randn(100000)

# Measure the time for tanh function
start_time_tanh = time.time()
for _ in range(100000):
    tanh(x)
end_time_tanh = time.time()

# Measure the time for sigmoid function
start_time_sigmoid = time.time()
for _ in range(100000):
    sigmoid(x)
end_time_sigmoid = time.time()

# Calculate the elapsed time
elapsed_time_tanh = end_time_tanh - start_time_tanh
elapsed_time_sigmoid = end_time_sigmoid - start_time_sigmoid

# Print the results
print(f'Time taken for 10000 computations of tanh: {elapsed_time_tanh:.6f} seconds')
print(f'Time taken for 10000 computations of sigmoid: {elapsed_time_sigmoid:.6f} seconds')

# Compare and print which one is faster
if elapsed_time_tanh < elapsed_time_sigmoid:
    print("Tanh is faster")
else:
    print("Sigmoid is faster")
    
    
      

