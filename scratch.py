import numpy as np

# Define the sampling points
sampling_points = np.array([0, 0.25, 0.5, 0.75])


# Define the function
def f(x):
    return 4 + np.sin(6 * np.pi * x) - 2 * np.cos(2 * np.pi * x)


# Calculate the sample values
sample_values = f(sampling_points)

# Number of samples
N = len(sampling_points)

# Compute the DFT
F_k = np.fft.fft(sample_values) / float(N)

# Results
print(sample_values)
print(F_k)
