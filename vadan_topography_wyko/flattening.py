import numpy as np
import matplotlib.pyplot as plt

def flatten(image_raw_positive, Resolution, center_CS, leftedge_angle):
    # Find non-NaN and non-zero points in the image
    laser_x, laser_y = np.where(~np.isnan(image_raw_positive) & (image_raw_positive != 0))
    
    # Initialize the cord_laser array
    cord_laser = np.zeros((len(laser_x), 9))
    
    # Store the coordinates and heights
    cord_laser[:, 0] = laser_x
    cord_laser[:, 1] = laser_y
    cord_laser[:, 2] = image_raw_positive[laser_x, laser_y]
    
    # Convert pixel coordinates to real-world measurements
    Ysize, Xsize = image_raw_positive.shape
    cord_laser[:, 3] = Ysize * Resolution - (cord_laser[:, 0] - 1) * Resolution
    cord_laser[:, 4] = (cord_laser[:, 1] - 1) * Resolution
    
    # Fit plane of all data points
    Const = np.ones(len(cord_laser[:, 2]))
    Coefficients = np.linalg.lstsq(np.vstack([cord_laser[:, 4], cord_laser[:, 3], Const]).T, cord_laser[:, 2], rcond=None)[0]
    
    # Generate new set of coefficients for z-axis calculation with adjusted heights
    adjusted_heights = cord_laser[:, 2] / 1000  # Scale heights (in nm initially) to be in same units as x and y coods (um)
    Coefficients_adjusted = np.linalg.lstsq(np.vstack([cord_laser[:, 4], cord_laser[:, 3], Const]).T, adjusted_heights, rcond=None)[0]
    
    # Calculate real angles
    theta_z_real = np.degrees(np.arccos(1 / np.sqrt(Coefficients_adjusted[0]**2 + Coefficients_adjusted[1]**2 + 1)))
    theta_x_real = 90 - np.degrees(np.arccos(Coefficients_adjusted[0] / np.sqrt(Coefficients_adjusted[0]**2 + Coefficients_adjusted[1]**2 + 1)))
    theta_y_real = 90 - np.degrees(np.arccos(Coefficients_adjusted[1] / np.sqrt(Coefficients_adjusted[0]**2 + Coefficients_adjusted[1]**2 + 1)))
    
    # Leveling Laser to fit plane
    Z_fit_all = Coefficients[0] * cord_laser[:, 4] + Coefficients[1] * cord_laser[:, 3] + Coefficients[2]
    cord_laser[:, 5] = cord_laser[:, 2] - Z_fit_all
    cord_laser[:, 6] = cord_laser[:, 5] - np.min(cord_laser[:, 5])
    
    # Move Origin to center of hole
    cord_laser[:, 7] = cord_laser[:, 3] - center_CS[1]
    cord_laser[:, 8] = cord_laser[:, 4] - center_CS[0]
    
    # Use Rotation Matrix to orient laser
    rotate_angle = -leftedge_angle
    rotationMatrix = np.array([[np.cos(np.radians(rotate_angle)), -np.sin(np.radians(rotate_angle))],
                          [np.sin(np.radians(rotate_angle)), np.cos(np.radians(rotate_angle))]])
    
    # Initialize laserdata array with 6 columns
    laserdata = np.zeros((len(cord_laser), 6))
    
    # Store x, y and then x coordinates in "laserdata"
    laserdata[:, :2] = np.dot(rotationMatrix, cord_laser[:, [8, 7]].T).T
    laserdata[:, 2] = cord_laser[:, 6]
    
    # Delete tether residuals and contamination
    dellist = np.where((laserdata[:, 0] < -19) | (laserdata[:, 0] > 19))[0]
    laserdata = np.delete(laserdata, dellist, axis=0)
    
    # Fit plane again to processed laser data
    Const2 = np.ones(len(laserdata[:, 2]))
    Coefficients2 = np.linalg.lstsq(np.vstack([laserdata[:, 1], laserdata[:, 0], Const2]).T, laserdata[:, 2], rcond=None)[0]
    Z_fit_all2 = Coefficients2[0] * laserdata[:, 1] + Coefficients2[1] * laserdata[:, 0] + Coefficients2[2]
    laserdata[:, 3] = laserdata[:, 2] - Z_fit_all2
    laserdata[:, 4] = laserdata[:, 3] - np.min(laserdata[:, 3])
    
    # Subtract the mean of the z heights from all data points
    mean_z_height = np.mean(laserdata[:, 4])
    laserdata[:, 5] = laserdata[:, 4] - mean_z_height
    
    # Prepare data_processed_laser with columns [x, y, z]
    data_processed_laser = laserdata[:, [0, 1, 5]]
    
    # Flip the sign of the second column (y)
    data_processed_laser[:, 1] *= -1
    
    # Convert data_processed_laser to a numpy array
    data_processed_laser = np.array(data_processed_laser)
    
    # # DEBUG: Plot the leveled laser data using a scatter plot (mimicking MATLAB plot)
    # fig = plt.figure(figsize=(11, 6.5))
    # ax = fig.add_subplot(111, projection='3d')
    
    # scatter_plot = ax.scatter(data_processed_laser[:, 0], data_processed_laser[:, 1], data_processed_laser[:, 2], c=data_processed_laser[:, 2], cmap='jet', marker='.')
    
    # ax.set_xlabel('X (μm)', fontsize=12)
    # ax.set_ylabel('Y (μm)', fontsize=12)
    # ax.set_zlabel('Z (nm)', fontsize=12)
    
    # ax.set_title('Leveled Laser', fontsize=13, color='b')
    
    # fig.colorbar(scatter_plot, ax=ax, label='Z (nm)')
    
    # ax.view_init(elev=90., azim=0)
    
    # # Set the aspect ratio to be equal
    # ax.set_box_aspect([np.ptp(data_processed_laser[:, 0]), np.ptp(data_processed_laser[:, 1]), np.ptp(data_processed_laser[:, 2])])  # Aspect ratio is 1:1:1
    
    
    return data_processed_laser, theta_z_real, theta_x_real, theta_y_real

# Example usage
# laserdata, theta_z_real, theta_x_real, theta_y_real = level(image_raw_positive, Resolution, center_CS, leftedge_angle)

