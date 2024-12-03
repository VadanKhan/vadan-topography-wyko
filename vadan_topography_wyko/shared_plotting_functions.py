import numpy as np
import seaborn as sns


# Function to remove the offset (mean) from image_raw
def remove_offset(image_raw):
    mean_value = np.nanmean(image_raw)
    image_raw_no_offset = image_raw - mean_value
    return image_raw_no_offset


# Function to generate unique colors using Seaborn's color palette
def generate_unique_colors(num_colors, colour_set: str = "rainbow"):
    colors = sns.color_palette(colour_set, num_colors)
    return colors


def generate_laser_plotting_order(
    laserIDranges, row_dynamic, col_dynamic, rows, cols, plot_by_column, dynamic_arrays
):
    laser_plotting_order = []
    for dataind, laserIDrange in enumerate(laserIDranges):
        order = []
        if dynamic_arrays:
            current_rows = row_dynamic[dataind]
            current_cols = col_dynamic[dataind]
        else:
            current_rows = rows
            current_cols = cols

        if plot_by_column:
            for row in range(current_rows):
                for col in range(current_cols):
                    laserID = col * current_rows + row
                    if laserID < len(laserIDrange):
                        order.append(laserID)
        else:
            order = list(range(len(laserIDrange)))

        laser_plotting_order.append(order)  # Append the order list to laser_plotting_order
    return laser_plotting_order
