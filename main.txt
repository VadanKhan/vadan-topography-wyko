# ---------------------------------------------------------------------------- #
#                            Package Import Section                            #
# ---------------------------------------------------------------------------- #
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
# Close all plots, clear data in workspace and records in Command Window.
plt.close('all')
# Clear all variables (not directly applicable in Python, but resetting relevant variables)
DATASETS = []


# ---------------------------------------------------------------------------- #
#                      USER INPUT / CONFIGURATION SECTION                      #
# ---------------------------------------------------------------------------- #
#region USER INPUTS

# ---------------- Input and Output Paths + Analysis Plot Names -------------- #
INPUTPATH = 'C:\\Users\\762093\\Documents\\WYKO_DATA'
OUTPUTPATH = 'C:\\Users\\762093\\Documents\\WYKO_DATA\\OUTPUTS\\output_debug'

CAMPAIGN_NAME = 'nxhpp_comparisons'


# ---------------------------- DATASETS to Analyse --------------------------- #
DATASETS = [
    'NEHZX_CUBE_167',
    'NEHZX_CUBE_161',
    # Add more DATASETS as needed
]
NUMDATA = len(DATASETS)


# ------- input the number of ROWS and columns in measured laser array. ------- #
ROWS = 1
COLS = 1
ROWRANGE = range(1, ROWS + 1)
COLRANGE = range(1, COLS + 1)


# -------------------------- Colours of Each Dataset ------------------------- #
COLORS = [[1, 0.5, 0], [0.5, 0, 0.5], [1, 0.5, 0], [0, 0, 1], [1, 0, 1]]


# Saved Images Quality (300 for decent runtime)
IMGQUAL = 300


# Contour Plot Z limit (range = 2*zlim, measured in nm)
ZLIM = 400


# Define Maps for edge and centre cubes
EDGE_POINTS = ['166', '167', '147', '148', '185', '128', '177', '157', '158', '138', '139', '120', '205', '195', '90', '81']
CENTRE_POINTS = ['161', '162', '163', '142', '143', '144', '123', '124', '136']

# ------------------------------ Filter Settings ----------------------------- #
# The "DELTA_THRESHOLD" will give the maximum allowed difference between adjacent 
# heights (in nm). If above this value, the code will filter out this data point
# as unphysical
# 
# The "ANOMALY_THRESHOLD" gives a similar maximum for a separate filter. This filter
# runs an average of 60 points surrounding each point, and if the target is 
# different above this set threshold, then it is filtered out as unphysical.
# This should be set higher than the "DELTA_THRESHOLD".
APPLY_DELTA_FILTER = True
APPLY_AVERAGE_FILTER = True

DELTA_THRESHOLD = 3  # Adjust this value as needed
ANOMALY_THRESHOLD = 25  # Adjust this value as needed
WINDOW_SIZE_INPUT = 20 # Adjust this value as needed


# --------------- Image Detection Parameters for Edge Detection -------------- #
EDGEDETECT = 3 # parameter for edge detect function. only change when needed.
LEFTEDGEWINDOW = [[200, 450], [220, 260]] # window for left edge detect, specify X,Y ranges.
#endregion


# ------- Option to group and color label plots based on 'design infos' ------ #
GROUP_BY_DESIGN_INFO = False  # Set to true to group by design infos, false to group by waferID and cubeID

DESIGN_INFOS = ['S1.7g',]  # Few Files Check

COLOURS_DESIGN_ORGANISED = [[0, 1, 1], [0.5, 0, 0.5], [1, 0.5, 0], [0.5, 0.5, 0], [1, 0, 1]]


# --------------------------- Different Array Sizes -------------------------- #
#  Set this to true if you want different array sizes within an analysis
#  batch. The sizes can be set in the row_dynamic and column_dynamic vectors
# . row_dynamic and column_dynamic ARE NOT USED IF THIS IS SET TO FALSE, ALL
# THE ARRAY SIZES ARE PRESUMED TO FOLLOW THE INITIAL "ROWS" and "columns"
# setting
DYNAMIC_ARRAYS = False

ROW_DYNAMIC = []

COLUMN_DYNAMIC = []


# ------------------------- Plotting Indexing Option: ROW  ------------------------- #
# NOTE that the for loop needs to edited (for each row, for each column)
# for this to be complete
PLOT_BY_COLUMN = False

#endregion USER INPUTS


# ---------------------------------------------------------------------------- #
#                    Preprocessing Steps and Initialisation                    #
# ---------------------------------------------------------------------------- #
#region Pre-Processing.

# Create output directory if it doesn't exist
os.makedirs(OUTPUTPATH, exist_ok=True)


# Check if the lengths of datasets and design_infos match
if len(DATASETS) != len(DESIGN_INFOS):
    if len(DESIGN_INFOS) < len(DATASETS):
        DESIGN_INFOS.extend(['unspecified'] * (len(DATASETS) - len(DESIGN_INFOS)))
    elif len(DESIGN_INFOS) > len(DATASETS):
        DESIGN_INFOS = DESIGN_INFOS[:len(DATASETS)]
# Verify lengths match after adjustment
if len(DATASETS) != len(DESIGN_INFOS):
    raise ValueError('Number of datasets not equal to number of design infos! Please make sure lengths match.')


# ---------------------- Initialize location_labels list --------------------- #
location_labels = []
# Determine edge or centre for each cubeID
for dataset in DATASETS:
    _, cubeID = dataset.split('_CUBE_')
    if cubeID in EDGE_POINTS:
        location_labels.append('Edge')
    elif cubeID in CENTRE_POINTS:
        location_labels.append('Centre')
    else:
        location_labels.append('Other')  # In case the cubeID is not found in either list
        
        
# ----------------------- Initialize laserIDranges list ---------------------- #
laserIDranges = []
# Loop through each dataset to calculate and store laserIDrange
for dataset in DATASETS:
    if DYNAMIC_ARRAYS:
        index = DATASETS.index(dataset)
        rows = ROW_DYNAMIC[index]
        cols = COLUMN_DYNAMIC[index]
        rowrange = range(1, rows + 1)
        colrange = range(1, cols + 1)
        laserIDranges.append(list(range(1, len(rowrange) * len(colrange) + 1)))
    else:
        # If DYNAMIC_ARRAYS is false, use a default range (adjust as needed)
        rowrange = range(1, ROWS + 1)
        colrange = range(1, COLS + 1)
        laserIDranges.append(list(range(1, len(rowrange) * len(colrange) + 1)))


# -------------------- Preallocate lists for data storage -------------------- #
data_raw = [None] * NUMDATA
data_processed = [None] * NUMDATA
data_crownprofiles = [None] * NUMDATA
data_xcrownprofiles = [None] * NUMDATA
data_crowns = [None] * NUMDATA
angle_matrix = [None] * NUMDATA
# Loop through each dataset to preallocate inner lists (FOR INCREASED PROCESSING SPEED)
for dataind in range(NUMDATA):
    laserIDrange = laserIDranges[dataind]
    # Preallocate inner lists
    data_raw[dataind] = [None] * len(laserIDrange)
    data_processed[dataind] = [None] * len(laserIDrange)
    data_crownprofiles[dataind] = [None] * len(laserIDrange)
    data_xcrownprofiles[dataind] = [None] * len(laserIDrange)
    data_crowns[dataind] = np.full((len(laserIDrange), 3), np.nan)  # Initialize with NaNs
    angle_matrix[dataind] = np.full((len(laserIDrange), 3), np.nan)  # Initialize with NaNs

processedMessages = []
#endregion Pre-Processing


# ---------------------------------------------------------------------------- #
#                       Iterate Loop over all input Data                       #
# ---------------------------------------------------------------------------- #
#region Iterative Loop
# Loop to read and process all opd files

# ITERATE FOR EACH DATASET
for dataind, dataset in enumerate(DATASETS):
    waferID, cubeID = dataset.split('_CUBE_')
    print(f"{waferID} CUBE {cubeID} is in processing.")
    
    # Define the input path for the current cube
    cubePath = os.path.join(INPUTPATH, dataset)
    
    # Detect the format of the .opd file
    test_files = [f for f in os.listdir(cubePath) if f.endswith('.fc.opd')]
    if not test_files:
        raise FileNotFoundError(f"No .opd files found in the directory: {cubePath}")
    
    # Read the first .opd file to determine the format
    test_file_name = test_files[0]
    if 'Row_' in test_file_name and '_Col_' in test_file_name:
        opdfilenameformat = 'Row_{0}_Col_{1}_'
    elif 'row' in test_file_name and 'column' in test_file_name:
        opdfilenameformat = 'row{0}column{1}'
    else:
        raise ValueError(f"Unknown .opd file format: {test_file_name}")
    
    #endregion Iterative Loop
    
    # ITERNATE FOR ALL LASERS PER DATASET
    for laserIDind, (rowID, colID) in enumerate(zip(ROWRANGE, COLRANGE)):
        opdfilename = opdfilenameformat.format(rowID, colID)
        filename = os.path.join(cubePath, f"{opdfilename}.fc.opd")
        
        # surface = Surface.load(filename_debug)
        # surface.show()
            
        # ---------------------------------------------------------------------------- #
        #                              Reading .opd Files                              #
        # ---------------------------------------------------------------------------- #
        #region File Parsing
        blocks, params, image_raw = read_wyko_opd(filename)  # Read the .opd file
        image_raw = np.transpose(image_raw)
        # Calculate Resolution
        Resolution = float(params['Pixel_size']) * 1000  # um
        
        # print(params)  # Display all metadata
        # print(len(image_raw))        
        
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
        laser_edge, image_raw_positive = edge_detection(image_raw, EDGEDETECT)
        
        
        # ----------------------------- Laser Orientation ---------------------------- #    
        leftedge_angle, center_CS = estimate_rotation_and_cs(laser_edge, Resolution, LEFTEDGEWINDOW, image_raw)
        # print(f"Left Edge Angle: {leftedge_angle}")
        # print(f"Center Coordinate System: {center_CS}")
        
        
        # ------------------------------- Plane Fitting ------------------------------ #
        data_processed_laser, theta_z_real, theta_x_real, theta_y_real = flatten(image_raw_positive, Resolution, center_CS, leftedge_angle)
        # print(f"Theta_x (roll): {theta_x_real}")
        # print(f"Theta_y (pitch): {theta_y_real}")
        #endregion Image Processing
        
        # Store raw and processed data
        data_raw[dataind][laserIDind] = image_raw
        data_processed[dataind][laserIDind] = data_processed_laser
        angle_matrix[dataind][laserIDind] = [theta_z_real, theta_x_real, theta_y_real]
        
        
        # ---------------------------------------------------------------------------- #
        #                           Crown Profile Extraction                           #
        # ---------------------------------------------------------------------------- #
        #region Profile Extraction
        crown_profile, xcrown_profile = extract_crown_profiles(data_processed_laser, Resolution)
        
        # Apply filters to the crown and xcrown profiles
        if APPLY_DELTA_FILTER:
            crown_profile = crown_delta_filter(crown_profile, DELTA_THRESHOLD)
            xcrown_profile = crown_delta_filter(xcrown_profile, DELTA_THRESHOLD)
        if APPLY_AVERAGE_FILTER:
            crown_profile = crown_average_filter(crown_profile, window_size=WINDOW_SIZE_INPUT, threshold=ANOMALY_THRESHOLD)
            xcrown_profile = crown_average_filter(xcrown_profile, window_size=WINDOW_SIZE_INPUT, threshold=ANOMALY_THRESHOLD)

        # Calculate crown_values and xcrown values
        edgedistance = 0  # pixel numbers
        crown_value = 0 - 0.5 * (crown_profile[edgedistance, 1] + crown_profile[-edgedistance - 1, 1])
        
        xcrownP_value = 0 - xcrown_profile[edgedistance, 1]
        xcrownN_value = 0 - xcrown_profile[-edgedistance - 1, 1]
        
        # Store crown profiles and values
        data_crownprofiles[dataind][laserIDind] = crown_profile
        data_xcrownprofiles[dataind][laserIDind] = xcrown_profile
        data_crowns[dataind][laserIDind] = [crown_value, xcrownP_value, xcrownN_value]
        
        # print(f"Assigned to data_crowns[{dataind}][{laserIDind}]: YCrown = {crown_value}, XCrownP = {xcrownP_value}, XCrownN = {xcrownN_value}")
        
        plt.show()
        
        #endregion Crown Profile Extraction

# -------------------------- Print summary messages -------------------------- #
# print("-------------------------- Data Processing is Completed! --------------------------")
# print("Summary of all processed Datsets:")
# for dataset in DATASETS:
#     print(f"{dataset} has been processed.")
    
# # -------------------------- Print crown values -------------------------- #
# print("\nCrown values for each dataset:")
# # print(data_crowns)
# for dataind, dataset in enumerate(DATASETS):
#     waferID, cubeID = dataset.split('_CUBE_')
#     print(f"\n{waferID} CUBE {cubeID}:")
#     for laserIDind, crown_values in enumerate(data_crowns[dataind]):
#         ycrown, xcrownP, xcrownN = crown_values
#         print(f"  Laser {laserIDind + 1}: YCrown = {ycrown:.2f} nm, XCrownP = {xcrownP:.2f} nm, XCrownN = {xcrownN:.2f} nm")

# # -------------------------- Print crown profiles -------------------------- #
# print("\nCrown profiles for each dataset:")
# for dataind, dataset in enumerate(DATASETS):
#     waferID, cubeID = dataset.split('_CUBE_')
#     print(f"\n{waferID} CUBE {cubeID}:")
#     for laserIDind, crown_profile in enumerate(data_crownprofiles[dataind]):
#         print(f"  Laser {laserIDind + 1} Crown Profile:")
#         for point in crown_profile:
#             print(f"    {point}")


