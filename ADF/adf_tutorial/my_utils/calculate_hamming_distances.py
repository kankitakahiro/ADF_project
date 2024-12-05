import numpy as np

def hamming_distance_sum(input_list, max_len=1000):
    """
    Computes the sum of Hamming distances for the given list of vectors.
    
    :param input_list: List of binary vectors
    :param max_len: Maximum number of vectors to consider
    :return: Sum of Hamming distances and updated list of vectors
    """
    # Trim the list to the maximum allowed length
    if len(input_list) > max_len:
        input_list = input_list[-max_len:]

    # Convert to numpy array for calculations
    data = np.array(input_list)
    n = data.shape[0]
    distances = np.zeros((n, n))

    # Calculate pairwise Hamming distances
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = np.sum(data[i] != data[j])
    total_distance = np.sum(distances) / 2

    return total_distance
