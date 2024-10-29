"""
This code is for processing Wyko data files with extension '.fc.opd'.
The bottom surface of the laser array on uTP Stamp was measured by Wyko in
PSI & FA mode with the recipe 'LASER_20x1x_Cheng_PSI_FA_STAMP.ini'.
This code is capable of processing multiple Stamp samples at once,
generating & saving plots and statistics tables fully automatically.
In this version, the crown profiles are at the center lines in both
longitudinal and transverse directions.

This code will:
    1. read all opd files.
    2. process raw data to extract crown profiles and bowing.
    3. calculate statistics per stamp sample.
    4. auto plot analysis results.
    5. auto save plots and statistics summary table in excel.

User input required:
    1. input and output path.
    2. Samples information.
    3. Plot colors.
    4. specify parameters for edge detection, only change it when needed.

Output files by this code:
    1. f1: raw data plots per stamp sample.
    2. f2: processed data plots per stamp sample.
    3. f3: Crown and XCrown profiles per stamp sample.
    4. f4: bowing values of Crown, XCrown_P, XCrown_N.
    5. f5: boxplots of bowing values of Crown, XCrown_P, XCrown_N.
    6. f6: overlayed all Crown/XCrown profiles of all samples.
    7. f7: Stamp samples comparison by mean of bow in XCrown&Crown plane.
    8. table in excel: Laser bowing data statistics, including mean, sigma,
       max, min, range for all stamp samples.
    9. mat: all selected important data saved in mat file for record or
       further process.

Note: the vector length of waferIDs, cubeIDs, designinfos, machinenames,
colors should be the same, matching how many stamp samples you want to
process together. And the sequence of information in vectors should match
real samples. This info will be auto-applied to plots/table/file names.

Multiple Stamp samples with the same array size can be processed at once.
Each stamp sample has its own folder under 'inputPath', with the name format
of 'waferID_CUBE_cubeID', for example, 'NVIIH_CUBE_161'. Inside the folder,
individual opd files are autosaved by Wyko, with the naming format of
'Row_#_Col_#_.fc.opd' by STST Wyko 582, or 'row#column#.fc.opd' by NRM
Wyko 45.

1st released version: 'WykoDataProcessor_Laser_Stamps_FastAuto_PSI_v1'
on July 25, 2024.
Developed by Cheng Chen, Mech CAG.

1st upgraded version: 'WykoDataProcessor_Laser_Stamps_VK'
on 24/09/2024.
Developed by Vadan Khan, R&D Physicist at STST. 
"""
# -------------------------- Package import section -------------------------- #
#region IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from skimage import feature
from scipy.stats import norm
import scipy.io
import tkinter as tk
from tkinter import messagebox
import struct
from surfalize import Surface

from opdread_package import read_wyko_opd
from edge_detect import edge_detection
from laser_orientation import estimate_rotation_and_cs
from flattening import flatten
from crown_extraction import extract_crown_profiles
from filters import crown_delta_filter
from filters import crown_average_filter

#endregion

# -------------------- USER INPUT / CONFIGURATION SECTION -------------------- #
#region USER INPUTS
# Close all plots, clear data in workspace and records in Command Window.
plt.close('all')
# Clear all variables (not directly applicable in Python, but resetting relevant variables)
waferIDs = []
cubeIDs = []

waferIDs = ['NEHZX']  # Few Files Check

cubeIDs = ['161']  # Few Files Check

# User Inputs to specify Sample information.

# input folder path of Wyko data files.
inputPath = 'C:\\Users\\762093\\Documents\\WYKO_DATA'

# Output folder path of data analysis results.
output_path = 'C:\\Users\\762093\\Documents\\WYKO_DATA\\OUTPUTS\\output_debug'

design_infos = ['S1.7g',]  # Few Files Check

machinenames = ['Wyko45']  # Few Files Check

# Specify Colors for plots.
# colors = [
#     [1, 0, 0],       # Red
#     [0, 1, 0],       # Green
#     [0, 0, 1],       # Blue
#     [1, 0.5, 0],     # Orange
#     [0.5, 0, 0.5],   # Purple
#     [0, 1, 1],       # Cyan
#     [1, 0, 1],       # Magenta
#     [0.5, 0.5, 0],   # Olive
#     [0, 0.5, 0.5]    # Teal
# ]

colors = [[1, 0.5, 0], [0.5, 0, 0.5], [1, 0.5, 0], [0, 0, 1], [1, 0, 1]]  # Few Files Check

colours_design_organised = [
    [0, 1, 1],
    [0.5, 0, 0.5],
    [1, 0.5, 0],
    [0.5, 0.5, 0],
    [1, 0, 1]
]

# input the number of rows in measured laser array.
rows = 3

# input the number of columns in measured laser array.
cols = 7

# Saved Images Quality (300 for decent runtime, 1000 for images that can be presented)
imgqual = 300

# Contour Plot Z limit (range = 2*zlim, measured in nm)
zlim = 400

# Option to group and color label plots based on 'design infos'
group_by_design_info = False  # Set to true to group by design infos, false to group by waferID and cubeID


# The "delta_threshold" will give the maximum allowed difference between adjacent 
# heights (in nm). If above this value, the code will filter out this data point
# as unphysical
# 
# The "anomaly_threshold" gives a similar maximum for a separate filter. This filter
# runs an average of 60 points surrounding each point, and if the target is 
# different above this set threshold, then it is filtered out as unphysical.
# This should be set higher than the "delta_threshold".
delta_threshold = 3  # Adjust this value as needed
anomaly_threshold = 25  # Adjust this value as needed
window_size_input = 20 # Adjust this value as needed


# Image Detection Parameters for Edge Detection
edgedetect = 3 # parameter for edge detect function. only change when needed.
RctangleCS_leftedge = [[200, 450], [220, 260]] # window for left edge detect, specify X,Y ranges.
#endregion

# ---------------------------------------------------------------------------- #
#                    Preprocessing Steps and Initialisation                    #
# ---------------------------------------------------------------------------- #
#region Pre-Processing.

# Check if input file name info have the same length.
if len(waferIDs) != len(cubeIDs):
    raise ValueError('Number of wafers not equal to Number of cubes! Please make sure length match!')
elif len(waferIDs) != len(design_infos):
    raise ValueError('Number of wafers not equal to Number of designs! Please make sure length match!')
elif len(waferIDs) != len(machinenames):
    raise ValueError('Number of wafers not equal to Number of machinenames! Please make sure length match!')

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

rowrange = range(1, rows + 1)
colrange = range(1, cols + 1)
laserIDrange = range(1, len(rowrange) * len(colrange) + 1)

# Create variables for data storage.
data_raw = [[None for _ in range(len(cubeIDs))] for _ in range(len(laserIDrange))]  # store all raw data from opd files.
data_processed = [[None for _ in range(len(cubeIDs))] for _ in range(len(laserIDrange))]  # store all processed data.
data_crownprofiles = [[None for _ in range(len(cubeIDs))] for _ in range(len(laserIDrange))]  # store all longitudinal profiles.
data_xcrownprofiles = [[None for _ in range(len(cubeIDs))] for _ in range(len(laserIDrange))]  # store all transverse profiles.
data_crowns = np.nan * np.zeros((len(laserIDrange) * len(cubeIDs), 3))  # 1 is crown, 2,3 is xcrown.
corwndataindex = 1
#endregion




# ---------------------------------------------------------------------------- #
#                       Iterate Loop over all input Data                       #
# ---------------------------------------------------------------------------- #
#region Iterative Loop
# Loop to read and process all opd files
for cubeind in range(len(cubeIDs)):
    waferID = waferIDs[cubeind]
    cubeID = cubeIDs[cubeind]
    machinename = machinenames[cubeind]
    print(f"{waferID} CUBE {cubeID} measured by {machinename} is in processing.")

    # Define opd file name format based on Wyko machine.
    if machinename == 'Wyko45':
        opdfilenameformat = 'row{0}column{1}'
    elif machinename == 'Wyko582':
        opdfilenameformat = 'Row_{0}_Col_{1}_'
        
    # debug specific .opd file
    rowID_debug = 1
    colID_debug = 1
    opdfilename_debug = opdfilenameformat.format(rowID_debug, colID_debug)
    filename_debug = filename = os.path.join(inputPath, f"{waferID}_CUBE_{cubeID}", f"{opdfilename_debug}.fc.opd")
        
    # surface = Surface.load(filename_debug)
    # surface.show()
    
    # for rowIDind in range(len(rowrange)):
    #     rowID = rowrange[rowIDind]
    #     for colIDind in range(len(colrange)):
    #         colID = colrange[colIDind]
    #         opdfilename = opdfilenameformat.format(rowID, colID)
    #         filename = os.path.join(inputPath, f"{waferID}_CUBE_{cubeID}", f"{opdfilename}.fc.opd")
#endregion Iterative Loop
    
# ---------------------------------------------------------------------------- #
#                              Reading .opd Files                              #
# ---------------------------------------------------------------------------- #
    #region File Parsing
    blocks, params, image_raw = read_wyko_opd(filename_debug)  # Read the .opd file
    
    image_raw = np.transpose(image_raw)
    
    # print(params)  # Display all metadata
    # print(len(image_raw))        
    
    # Calculate Resolution
    Resolution = float(params['Pixel_size']) * 1000  # um

    # # Plot the raw data for debugging
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(image_raw, cmap='jet', aspect='equal')
    # plt.colorbar(label='Z$(\\mu m)$')
    # plt.xlabel('Column Pixel')
    # plt.ylabel('Row Pixel')
    # plt.title('Raw Data', fontsize=13, color='b')
    # plt.show()
    
    #endregion File Parsing
        
# ---------------------------------------------------------------------------- #
#                               Image Processing                               #
# ---------------------------------------------------------------------------- #
    #region Image Processing
        
    # ------------------------------ Edge Detection ------------------------------ #
    laser_edge, image_raw_positive = edge_detection(image_raw, edgedetect)
    
    
    # ----------------------------- Laser Orientation ---------------------------- #    
    leftedge_angle, center_CS = estimate_rotation_and_cs(laser_edge, Resolution, RctangleCS_leftedge, image_raw)
    # print(f"Left Edge Angle: {leftedge_angle}")
    # print(f"Center Coordinate System: {center_CS}")
    
    
    # ------------------------------- Plane Fitting ------------------------------ #
    data_processed, theta_z_real, theta_x_real, theta_y_real = flatten(image_raw_positive, Resolution, center_CS, leftedge_angle)
    # print(f"Theta_x (roll): {theta_x_real}")
    # print(f"Theta_y (pitch): {theta_y_real}")
    #endregion Image Processing
    
    
# ---------------------------------------------------------------------------- #
#                           Crown Profile Extraction                           #
# ---------------------------------------------------------------------------- #
    #region Profile Extraction
    crown_profile, xcrown_profile = extract_crown_profiles(data_processed, Resolution)
    
    # Apply filters to the crown and xcrown profiles  
    partiallyfiltered_crown_profile = crown_delta_filter(crown_profile, delta_threshold)
    partiallyfiltered_xcrown_profile = crown_delta_filter(xcrown_profile, delta_threshold)
    filtered_crown_profile = crown_average_filter(partiallyfiltered_crown_profile, window_size=window_size_input, threshold=anomaly_threshold)
    filtered_xcrown_profile = crown_average_filter(partiallyfiltered_xcrown_profile, window_size=window_size_input, threshold=anomaly_threshold)

    # Calculate crown_values and xcrown values
    edgedistance = 0  # pixel numbers
    crown_value = 0 - 0.5 * (filtered_crown_profile[edgedistance, 1] + filtered_crown_profile[-edgedistance - 1, 1])
    
    xcrownP_value = 0 - filtered_xcrown_profile[edgedistance, 1]
    xcrownN_value = 0 - filtered_xcrown_profile[-edgedistance - 1, 1]
    
    print(f"YCrown: {crown_value}")
    print(f"XCrownP: {xcrownP_value}")
    print(f"XCrownN: {xcrownN_value}")
    
    plt.show()
    
    #endregion Crown Profile Extraction





