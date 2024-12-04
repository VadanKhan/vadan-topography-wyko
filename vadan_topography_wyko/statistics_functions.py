import numpy as np


# Function to calculate R² value
def calculate_r_squared(x, y):
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0, 1]
    return correlation_xy**2


# Function to remove anomalies and identify outliers
def IQR_filtering(data):
    Q1 = np.nanpercentile(data, 25)
    Q3 = np.nanpercentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
    return filtered_data, outliers


# Function to filter both crown heights and angles
def common_data_filter(crown_column, angles):
    filtered_crown_column, crown_outliers = IQR_filtering(crown_column)
    filtered_angles = []
    angle_outliers = []

    for i in range(angles.shape[1]):
        filtered_angle, angle_outlier = IQR_filtering(angles[:, i])
        filtered_angles.append(filtered_angle)
        angle_outliers.append(angle_outlier)

    # Find common indices that are not outliers in both crown heights and angles
    common_indices = set(range(len(crown_column))) - set(crown_outliers)
    for outlier in angle_outliers:
        common_indices -= set(outlier)

    common_indices = list(common_indices)
    filtered_crown_column = crown_column[common_indices]
    filtered_angles = angles[common_indices, :]

    # Find indices that are outliers in either crown heights or angles
    all_outliers = set(range(len(crown_column))) - set(common_indices)

    return filtered_crown_column, filtered_angles, list(all_outliers)


def calculate_statistics_single_column(crown_column, angles):
    # Initialize a dictionary to store statistics
    stats = {
        "mean": [],
        "std_dev": [],
        "max": [],
        "min": [],
        "range": [],
        "outlier_lasers": [],
        "mean_theta_x": [],
        "mean_theta_y": [],
        "mean_yaw": [],
        "std_theta_x": [],
        "std_theta_y": [],
        "std_yaw": [],
        "range_theta_x": [],
        "range_theta_y": [],
        "range_yaw": [],
        "max_min_theta_x": [],
        "max_min_theta_y": [],
        "max_min_yaw": [],
        "r_squared_theta_x": [],
        "r_squared_theta_y": [],
        "r_squared_yaw": [],
    }

    # Filter data
    filtered_crown_column, filtered_angles, all_outliers = filter_data(crown_column, angles)

    # Calculate statistics for the crown column
    stats["mean"] = np.nanmean(filtered_crown_column)
    stats["std_dev"] = np.nanstd(filtered_crown_column)
    stats["max"] = np.nanmax(filtered_crown_column)
    stats["min"] = np.nanmin(filtered_crown_column)
    stats["range"] = np.nanmax(filtered_crown_column) - np.nanmin(filtered_crown_column)
    stats["outlier_lasers"] = ",".join(map(str, all_outliers))

    # Calculate statistics for angles
    stats["mean_theta_x"] = np.nanmean(filtered_angles[:, 1])
    stats["mean_theta_y"] = np.nanmean(filtered_angles[:, 2])
    stats["mean_yaw"] = np.nanmean(filtered_angles[:, 3])
    stats["std_theta_x"] = np.nanstd(filtered_angles[:, 1])
    stats["std_theta_y"] = np.nanstd(filtered_angles[:, 2])
    stats["std_yaw"] = np.nanstd(filtered_angles[:, 3])
    stats["range_theta_x"] = np.nanmax(filtered_angles[:, 1]) - np.nanmin(filtered_angles[:, 1])
    stats["range_theta_y"] = np.nanmax(filtered_angles[:, 2]) - np.nanmin(filtered_angles[:, 2])
    stats["range_yaw"] = np.nanmax(filtered_angles[:, 3]) - np.nanmin(filtered_angles[:, 3])
    stats["max_min_theta_x"] = (
        f"{np.nanmax(filtered_angles[:, 1])}, {np.nanmin(filtered_angles[:, 1])}"
    )
    stats["max_min_theta_y"] = (
        f"{np.nanmax(filtered_angles[:, 2])}, {np.nanmin(filtered_angles[:, 2])}"
    )
    stats["max_min_yaw"] = f"{np.nanmax(filtered_angles[:, 3])}, {np.nanmin(filtered_angles[:, 3])}"

    # Calculate R² values
    stats["r_squared_theta_x"] = calculate_r_squared(filtered_crown_column, filtered_angles[:, 1])
    stats["r_squared_theta_y"] = calculate_r_squared(filtered_crown_column, filtered_angles[:, 2])
    stats["r_squared_yaw"] = calculate_r_squared(filtered_crown_column, filtered_angles[:, 3])

    return stats
