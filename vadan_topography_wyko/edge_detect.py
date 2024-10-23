import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

def edge_detection(image_raw, edgedetect):
    # Level the image data to positive values
    image_raw_positive = image_raw - np.nanmin(image_raw) # need to use nanmin to detect minimum of the non-nan values
    
    # # Check the number of pixels and non-NaN values
    # total_pixels = image_raw_positive.size
    # non_nan_pixels = np.count_nonzero(~np.isnan(image_raw_positive))

    # print(f"Total pixels: {total_pixels}")
    # print(f"Non-NaN pixels: {non_nan_pixels}")

    # # Plot the leveled image data
    # plt.figure()
    # plt.imshow(image_raw_positive, cmap='gray')
    # plt.title('Leveled Image Data')
    # plt.colorbar()
    # plt.show()

    # Thresholding to create a binary image
    image_grey = image_raw_positive > 0

    # # Plot the binary image
    # plt.figure()
    # plt.imshow(image_grey, cmap='gray')
    # plt.title('Binary Image')
    # plt.colorbar()
    # plt.show()

    # Edge detection using Canny algorithm
    laser_edge = feature.canny(image_grey, sigma=edgedetect)

    # # Debug plotting to track the detected edge
    # plt.figure()
    # plt.imshow(laser_edge, cmap='gray')
    # plt.title('Detected Edges')
    # plt.colorbar()
    # plt.show()

    return laser_edge, image_raw_positive