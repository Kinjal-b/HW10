# This implements an input normalization algorithm that normalizes input data to have zero mean and unit variance.
# The output shows the first 5 samples of both the original and the normalized data, demonstrating how the values have been adjusted according to the normalization process. ​

def normalize_input(data):
    """
    Normalizes input data to have zero mean and unit variance.

    Parameters:
    data (numpy array): The input data to be normalized.

    Returns:
    numpy array: The normalized data.
    """
    # Calculate the mean and standard deviation of the data
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    
    # Ensure standard deviation is not zero to avoid division by zero
    std[std == 0] = 1
    
    # Normalize the data
    normalized_data = (data - mean) / std
    
    return normalized_data

# Example usage
import numpy as np

# Generate some random data
data = np.random.rand(100, 5)  # 100 samples, 5 features

# Normalize the data
normalized_data = normalize_input(data)

print("Original Data:\n", data[:5])  # Display first 5 samples of original data
print("\nNormalized Data:\n", normalized_data[:5])  # Display first 5 samples of normalized data
