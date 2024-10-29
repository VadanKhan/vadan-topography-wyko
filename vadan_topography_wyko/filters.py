import numpy as np
import matplotlib.pyplot as plt

def crown_delta_filter(data, threshold):
    """
    Filters data based on the threshold for the change in y-value for crown profiles.
    
    Parameters:
    data (numpy.ndarray): The input data array.
    threshold (float): The maximum allowed change in y-value.
    
    Returns:
    numpy.ndarray: The filtered data array.
    """
    if data.size == 0 or data.shape[0] == 1:
        return data

    # Initial filtering based on the threshold for the change in y-value
    initial_filtered_data = data[np.abs(np.diff(data[:, 1], prepend=data[0, 1])) <= threshold]

    # Remove first points if they exceed the threshold compared to the filtered data
    while initial_filtered_data.size > 1 and np.abs(initial_filtered_data[0, 1] - initial_filtered_data[1, 1]) > threshold:
        initial_filtered_data = initial_filtered_data[1:]
        
    # Remove last points if they exceed the threshold compared to the filtered data
    while initial_filtered_data.size > 1 and np.abs(initial_filtered_data[-1, 1] - initial_filtered_data[-2, 1]) > threshold:
        initial_filtered_data = initial_filtered_data[:-1]
        
    # # Debug: Plot the filtered profiles
    # plt.figure()
    # plt.plot(initial_filtered_data[:, 0], initial_filtered_data[:, 1], '.-')
    # plt.title('Delta Filtered Profile')
    # plt.xlabel('"across profile" (um)')
    # plt.ylabel('Z (nm)')
    # plt.tight_layout()

    return initial_filtered_data

import numpy as np

def crown_average_filter(data, window_size=60, threshold=1):
    """
    Filters out points in a crown profile that differ from an average of the surrounding points.
    
    Parameters:
    data (numpy.ndarray): The input data array.
    window_size (int): The size of the window for averaging.
    threshold (float): The maximum allowed difference from the average.
    
    Returns:
    numpy.ndarray: The advanced filtered data array.
    """
    if data.shape[0] <= window_size:
        return data

    filtered_data = []
    half_window = window_size // 2

    for i in range(data.shape[0]):
        start_idx = max(0, i - half_window)
        end_idx = min(data.shape[0], i + half_window + 1)
        surrounding_points = np.concatenate((data[start_idx:i], data[i+1:end_idx]))

        if surrounding_points.size > 0:
            surrounding_avg = np.mean(surrounding_points[:, 1])
            if np.abs(data[i, 1] - surrounding_avg) <= threshold:
                filtered_data.append(data[i])
                
    filtered_data_arr = np.array(filtered_data)
    
    # # DEBUG: Plot the filtered profiles
    # plt.figure()
    # plt.plot(filtered_data_arr[:, 0], filtered_data_arr[:, 1], '.-')
    # plt.title('Average Filtered Profile')
    # plt.xlabel('"across profile" (um)')
    # plt.ylabel('Z (nm)')
    # plt.tight_layout()

    return filtered_data_arr
