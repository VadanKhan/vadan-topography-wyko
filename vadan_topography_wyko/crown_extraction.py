import numpy as np
import matplotlib.pyplot as plt

def extract_crown_profiles(data_processed, Resolution):
    # Initialize crown_profile and xcrown_profile arrays
    crown_profile = []
    xcrown_profile = []
    
    # Extract crown_profile profile (longitudinal)
    for i in range(len(data_processed)):
        if (-0.5 * Resolution <= data_processed[i, 0] <= 0.5 * Resolution):
            if not np.isnan(data_processed[i, 1]) and not np.isnan(data_processed[i, 2]):
                crown_profile.append([data_processed[i, 1], data_processed[i, 2]])
    
    # Extract xcrown_profile profile (transverse)
    for i in range(len(data_processed)):
        if (-1 * Resolution <= data_processed[i, 1] <= Resolution):
            if not np.isnan(data_processed[i, 0]) and not np.isnan(data_processed[i, 2]):
                xcrown_profile.append([data_processed[i, 0], data_processed[i, 2]])
    
    # Convert to numpy arrays for further processing
    crown_profile = np.array(crown_profile)
    xcrown_profile = np.array(xcrown_profile)
    
    # Sort rows by the first column
    crown_profile = crown_profile[crown_profile[:, 0].argsort()]
    xcrown_profile = xcrown_profile[xcrown_profile[:, 0].argsort()]
    
    # Filter out NaNs from crown_profile and xcrown_profile
    crown_profile = crown_profile[~np.isnan(crown_profile).any(axis=1)]
    xcrown_profile = xcrown_profile[~np.isnan(xcrown_profile).any(axis=1)]
    
    # Shift crown_profile to mid = 0
    if len(crown_profile) > 0:
        midind = round(len(crown_profile) / 2)
        crown_profile[:, 1] -= crown_profile[midind, 1]
    
    # Shift xcrown_profile to mid = 0
    if len(xcrown_profile) > 0:
        midindx = round(len(xcrown_profile) / 2)
        xcrown_profile[:, 1] -= xcrown_profile[midindx, 1]
        
    # # DEBUG: plot extracted profiles
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # axs[0].plot(crown_profile[:, 0], crown_profile[:, 1], '.-')
    # axs[0].set_title('Raw YCrown Profile')
    # axs[0].set_xlabel('Y (um)')
    # axs[0].set_ylabel('Z (nm)')
    
    # axs[1].plot(xcrown_profile[:, 0], xcrown_profile[:, 1], '.-')
    # axs[1].set_title('Raw XCrown Profile')
    # axs[1].set_xlabel('X (um)')
    # axs[1].set_ylabel('Z (nm)')
    
    # plt.tight_layout()
    
    return crown_profile, xcrown_profile
# Example usage
# data_processed = ... (your processed data here)
# Resolution = ... (your resolution value here)
# crown_profile, xcrown_profile = extract_crown_profiles(data_processed, Resolution)
