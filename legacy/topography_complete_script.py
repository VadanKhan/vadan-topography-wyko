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

Note: the vector length of wafer_ids, cube_ids, designinfos, machinenames,
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


#endregion

# -------------------- USER INPUT / CONFIGURATION SECTION -------------------- #
#region USER INPUTS
# Close all plots, clear data in workspace and records in Command Window.


plt.close('all')
# Clear all variables (not directly applicable in Python, but resetting relevant variables)
wafer_ids = []
cube_ids = []

# # Input wafer ID for each stamp sample.
# wafer_ids = [
#     'NEHZW',
#     'NEHZX',
#     'NEHZX',
#     'NVI06',
#     'NVIIG',
#     'NVIIG',
#     'NVIIY',
#     'NVIJQ',
#     'NVIJQ'
# ]

wafer_ids = ['NVI06', 'NVIIG']  # Few Files Check

# # Input CUBE ID for each stamp sample.
# cube_ids = [
#     '167',
#     '161',
#     '167',
#     '161',
#     '161',
#     '167',
#     '161',
#     '161',
#     '167'
# ]

cube_ids = ['161', '161']  # Few Files Check

# Close all plots, clear data in workspace and records in Command Window.
plt.close('all')
# Clear all variables (not directly applicable in Python, but resetting relevant variables)
wafer_ids = []
cube_ids = []

# # Input wafer ID for each stamp sample.
# wafer_ids = [
#     'NEHZW',
#     'NEHZX',
#     'NEHZX',
#     'NVI06',
#     'NVIIG',
#     'NVIIG',
#     'NVIIY',
#     'NVIJQ',
#     'NVIJQ'
# ]

wafer_ids = ['NEHZX']  # Few Files Check

# # Input CUBE ID for each stamp sample.
# cube_ids = [
#     '167',
#     '161',
#     '167',
#     '161',
#     '161',
#     '167',
#     '161',
#     '161',
#     '167'
# ]

cube_ids = ['161']  # Few Files Check

# User Inputs to specify Sample information.

# input folder path of Wyko data files.
# inputPath = 'C:\\Users\\946859\\OneDrive - Seagate Technology\\Desktop\\Meeting\\Wyko'
# inputPath = 'L:\\wyko data\\4. STST Stamp Samples'
input_path = 'C:\\Users\\762093\\Documents\\WYKO_DATA'

# Output folder path of data analysis results.
output_path = 'C:\\Users\\762093\\Documents\\WYKO_DATA\\output_debug'

# # Generate the output folder path dynamically using wafer_ids and cube_ids.
# output_path = os.path.join(input_path, f"{wafer_ids[0]}_CUBE_{cube_ids[0]}", 'Outputs of Wyko45 and Wyko582 Data Analysis')


# # Input design info for each stamp sample.
# design_infos = [
#     'f8 tether',
#     'f8 tether',
#     'f8 tether',
#     'g6 tether',
#     'g8 tether',
#     'g8 tether',
#     'g8 tether',
#     'g6 tether',
#     'g6 tether'
# ]


design_infos = ['S1.7g']  # Few Files Check

# Input Wyko info for each stamp sample.
# # If STST Wyko, set as 'Wyko582'; if NRM Wyko, set as 'Wyko45'
# machine_names = [
#     'Wyko45',
#     'Wyko45',
#     'Wyko45',
#     'Wyko45',
#     'Wyko45',
#     'Wyko45',
#     'Wyko45',
#     'Wyko45',
#     'Wyko45'
# ]

machine_names = ['Wyko45']  # Few Files Check

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

# parameter for edge detection.
edgedetect = 0.973  # parameter for edge detect function. only change when needed.
RctangleCS_leftedge = [[200, 450], [220, 260]]  # window for left edge detect, specify X,Y ranges.

# Saved Images Quality (300 for decent runtime, 1000 for images that can be presented)
imgqual = 300

# Contour Plot Z limit (range = 2*zlim, measured in nm)
zlim = 400

# Option to group and color label plots based on 'design infos'
group_by_design_info = False  # Set to true to group by design infos, false to group by waferID and cubeID

# Define the threshold for the maximum allowed change in y-value

# The "delta_threshold" will give the maximum allowed difference between adjacent 
# heights (in nm). If above this value, the code will filter out this data point
# as unphysical
# 
# The "anomaly_threshold" gives a similar maximum for a separate filter. This filter
# runs an average of 60 points surrounding each point, and if the target is 
# different above this set threshold, then it is filtered out as unphysical.
# This should be set higher than the "delta_threshold".

delta_threshold = 1.5  # Adjust this value as needed
anomaly_threshold = 20  # Adjust this value as needed
#endregion


# Typically no need to change codes below.
# Initialize variables/settings.

# Check if input file name info have the same length.
if len(wafer_ids) != len(cube_ids):
    raise ValueError('Number of wafers not equal to Number of cubes! Please make sure length match!')
elif len(wafer_ids) != len(design_infos):
    raise ValueError('Number of wafers not equal to Number of designs! Please make sure length match!')
elif len(wafer_ids) != len(machine_names):
    raise ValueError('Number of wafers not equal to Number of machinenames! Please make sure length match!')

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

rowrange = range(1, rows + 1)
colrange = range(1, cols + 1)
laserIDrange = range(1, len(rowrange) * len(colrange) + 1)

# Create variables for data storage.
data_raw = [[None for _ in range(len(cube_ids))] for _ in range(len(laserIDrange))]  # store all raw data from opd files.
data_processed = [[None for _ in range(len(cube_ids))] for _ in range(len(laserIDrange))]  # store all processed data.
data_crownprofiles = [[None for _ in range(len(cube_ids))] for _ in range(len(laserIDrange))]  # store all longitudinal profiles.
data_xcrownprofiles = [[None for _ in range(len(cube_ids))] for _ in range(len(laserIDrange))]  # store all transverse profiles.
data_crowns = np.nan * np.zeros((len(laserIDrange) * len(cube_ids), 3))  # 1 is crown, 2,3 is xcrown.
corwndataindex = 1


# ----------------- Typically no need to change codes below. ----------------- #
# -------------------------- inital processing steps ------------------------- #
#region Initialize settings.

# Check if input file name info have the same length.
if len(wafer_ids) != len(cube_ids):
    raise ValueError('Number of wafers not equal to Number of cubes! Please make sure length match!')
elif len(wafer_ids) != len(design_infos):
    raise ValueError('Number of wafers not equal to Number of designs! Please make sure length match!')
elif len(wafer_ids) != len(machine_names):
    raise ValueError('Number of wafers not equal to Number of machinenames! Please make sure length match!')

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

rowrange = range(1, rows + 1)
colrange = range(1, cols + 1)
laserIDrange = range(1, len(rowrange) * len(colrange) + 1)

# Create variables for data storage.
data_raw = [[None for _ in range(len(cube_ids))] for _ in range(len(laserIDrange))]  # store all raw data from opd files.
data_processed = [[None for _ in range(len(cube_ids))] for _ in range(len(laserIDrange))]  # store all processed data.
data_crownprofiles = [[None for _ in range(len(cube_ids))] for _ in range(len(laserIDrange))]  # store all longitudinal profiles.
data_xcrownprofiles = [[None for _ in range(len(cube_ids))] for _ in range(len(laserIDrange))]  # store all transverse profiles.
data_crowns = np.nan * np.zeros((len(laserIDrange) * len(cube_ids), 3))  # 1 is crown, 2,3 is xcrown.
corwndataindex = 1
#endregion

# ------------------------------- processing loop ------------------------------- #
#region Loop to read and process all opd files.
for cubeind in range(len(cube_ids)):
    wafer_id = wafer_ids[cubeind]
    cube_id = cube_ids[cubeind]
    laser_id_ind = 1
    machine_name = machine_names[cubeind]
    print(f"{wafer_id} CUBE {cube_id} measured by {machine_name} is in processing.")
    
    # Define opd file name format based on Wyko machine.
    if machine_name == 'Wyko45':
        opd_filename_format = 'row{}column{}'
    elif machine_name == 'Wyko582':
        opd_filename_format = 'Row_{}_Col_{}_'
    
    for row_id_ind in range(len(rowrange)):
        row_id = rowrange[row_id_ind]
        for col_id_ind in range(len(colrange)):
            col_id = colrange[col_id_ind]
            opd_filename = opd_filename_format.format(row_id, col_id)  # specify .opd file name.
            filename = os.path.join(input_path, f"{wafer_id}_CUBE_{cube_id}", f"{opd_filename}.fc.opd")  # specify full file name.
            
            # Read opd file. DON'T CHANGE ANYTHING IN THIS SECTION!
            with open(filename, 'rb') as f:
                E = f.read()
            E2 = E.decode('ISO-8859-1').encode('ISO-8859-1')

            Directorybytes = 6002
            Block_len = 24
            ind = E2.find(b'Directory')
            BlockID = 1
            BLOCKS = []

            while ind + 21 <= Directorybytes:
                block = {
                    'name': E2[ind:ind+16].decode('ISO-8859-1').strip(),
                    'type': int.from_bytes(E2[ind+16:ind+18], 'big'),
                    'Length': int.from_bytes(E2[ind+18:ind+22], 'little'),  # Changed to 'little' endian
                    'BlockTail': int.from_bytes(E2[ind+22:ind+24], 'big')
                }
                print(f"Raw bytes for Length: {E2[ind+18:ind+22]}, Interpreted Length: {block['Length']}")
                BLOCKS.append(block)
                BlockID += 1
                ind += Block_len

            name = [block['name'] for block in BLOCKS]
            type_ = [block['type'] for block in BLOCKS]
            Length = [block['Length'] for block in BLOCKS]
            BlockTail = [block['BlockTail'] for block in BLOCKS]

            # Find block index.
            Parameters = ['Pixel_size', 'Wavelength']
            ParametersValue = []
            for param in Parameters:
                found = False
                for j, block in enumerate(BLOCKS):
                    print(f"Block {j}: Name={block['name']}, Type={block['type']}, Length={block['Length']}, BlockTail={block['BlockTail']}")
                    if block['name'].startswith(param):
                        ParameterInd = j
                        found = True
                        break
                if found:
                    ind = 2 + sum(Length[:j]) + 1
                    print(f"ind: {ind}, Length[j]: {Length[j]}, Buffer length: {len(E2)}")
                    if 0 < Length[j] <= len(E2) - ind:  # Added validation for Length
                        buffer_slice = E2[ind:ind+Length[j]]
                        if len(buffer_slice) == Length[j]:
                            ParametersValue.append(np.frombuffer(buffer_slice, dtype=np.float32)[0])
                            print(f"Extracted {param}: {ParametersValue[-1]}")
                        else:
                            print(f"Buffer slice length mismatch at index {ind} with length {Length[j]}")
                    else:
                        print(f"Skipping invalid length at index {ind} with length {Length[j]}")
                else:
                    print(f"Parameter {param} not found in BLOCKS")

            # Debugging: Print the extracted parameters
            print(f"Extracted ParametersValue: {ParametersValue}")

            # Ensure ParametersValue has the required elements
            if len(ParametersValue) < 2:
                raise ValueError("Failed to extract required parameters: Pixel_size and Wavelength")

            # RAW_DATA.
            ind = 2 + Length[0] + Length[1] + Length[2] + 1
            Xsize = int.from_bytes(E2[ind:ind+2], 'big')
            Ysize = int.from_bytes(E2[ind+2:ind+4], 'big')
            # Extract two bytes and convert to int16, then to int32
            Nbytes_data_bytes = E2[ind+4:ind+6]
            Nbytes_data = struct.unpack('<h', Nbytes_data_bytes)[0]  # '<h' for little-endian 16-bit integer
            
            Nbytes_data = 4 # VTEMPORARY FIX TO SET DATA BYTES

            # Debugging: Print raw bytes for Nbytes_data
            print(f"Raw bytes for Nbytes_data: {Nbytes_data_bytes}, Interpreted Nbytes_data: {Nbytes_data}")

            # Validate Nbytes_data
            if Nbytes_data == 0:
                raise ValueError("Nbytes_data is zero. Please check the file format and data extraction.")

            pixel_bytes = Nbytes_data
            ind += 6

            # Debugging: Print sizes and buffer length
            print(f"Xsize: {Xsize}, Ysize: {Ysize}, Nbytes_data: {Nbytes_data}, Buffer length: {len(E2[ind:ind+Xsize*Ysize*pixel_bytes])}")

            # Ensure the buffer length is sufficient
            if len(E2[ind:ind+Xsize*Ysize*pixel_bytes]) < Xsize * Ysize * pixel_bytes:
                raise ValueError("Buffer length is insufficient for the expected pixel data size.")

            pixeldata = np.frombuffer(E2[ind:ind+Xsize*Ysize*pixel_bytes], dtype=np.float32)

            # Check if pixeldata is empty
            if pixeldata.size == 0:
                raise ValueError("No data read into pixeldata. Please check the file and indices.")

            # Make a writable copy of pixeldata
            pixeldata = np.copy(pixeldata)

            idx = np.where(pixeldata >= 1e10)  # find empty pixels
            pixeldata[idx] = np.nan
            VSIWavelngth = ParametersValue[1]

            # Debugging: Print pixeldata size before reshaping
            print(f"pixeldata size: {pixeldata.size}, Expected size: {Xsize * Ysize}")

            if pixeldata.size != Xsize * Ysize:
                raise ValueError(f"Mismatch in pixeldata size. Expected {Xsize * Ysize}, got {pixeldata.size}")

            image_raw = pixeldata.reshape((Ysize, Xsize)) * VSIWavelngth - np.nanmean(pixeldata) * VSIWavelngth  # same as Vision output plot leveling to mean of data, but unit is nm.
            # Store raw data.
            data_raw[laser_id_ind-1][cubeind] = image_raw



            # Process Raw Data.
            Reslution = float(ParametersValue[0] * 1000)  # um

            # Edge detect.
            Ysize, Xsize = image_raw.shape
            columnsInImage, rowsInImage = np.meshgrid(np.arange(1, Xsize+1), np.arange(1, Ysize+1))
            columnsInImage_mm = columnsInImage * Reslution
            rowsInImage_mm = rowsInImage * Reslution
            # Leveling Z to positive.
            image_raw_positive = image_raw - np.nanmin(image_raw)
            # Find edges in data for further features identification
            image_grey = image_raw_positive > 0
            laser_edge = np.array(image_grey, dtype=np.uint8)
            # Using skimage for edge detection
            laser_edge = feature.canny(laser_edge, sigma=1)

            # Estimate rotation angle and CS of center.
            # Detect angle of Laser, for rotating laser to vertical.
            leftedgeRectangle = (rowsInImage >= RctangleCS_leftedge[0][0]) & (rowsInImage <= RctangleCS_leftedge[0][1]) & \
                                (columnsInImage >= RctangleCS_leftedge[1][0]) & (columnsInImage <= RctangleCS_leftedge[1][1])
            left_edge = laser_edge & leftedgeRectangle
            left_edge_x, left_edge_y = np.where(left_edge == 1)
            x_left = np.median(left_edge_y)
            leftedge = np.polyfit(left_edge_x, left_edge_y, 1)
            leftedge_angle = np.degrees(np.arctan(leftedge[0]))

            # Find CS of laser center, for shifting CS Origin to laser center.
            indexy, indexx = np.where(laser_edge)
            x_mid = np.mean(indexx)
            y_mid = np.mean(indexy)
            center_CS = [x_mid * Reslution, (Ysize - y_mid + 1) * Reslution]

            # Image to list.
            # Find center of hole and rotation angle.
            # Select data in raw, transform to points Data.
            laser_x, laser_y = np.where(~np.isnan(image_raw_positive) & (image_raw_positive != 0))
            cord_laser = np.zeros((len(laser_x), 9))
            cord_laser[:, 0] = laser_x
            cord_laser[:, 1] = laser_y
            for pt in range(len(cord_laser)):
                cord_laser[pt, 2] = image_raw_positive[int(cord_laser[pt, 0]), int(cord_laser[pt, 1])]
            cord_laser[:, 3] = Ysize * Reslution - (cord_laser[:, 0] - 1) * Reslution
            cord_laser[:, 4] = (cord_laser[:, 1] - 1) * Reslution

            # Fit plane of all data points.
            Const = np.ones((len(cord_laser[:, 2]), 1))  # Vector of ones for constant term.
            Coefficients = np.linalg.lstsq(np.column_stack((cord_laser[:, 4], cord_laser[:, 3], Const)), cord_laser[:, 2], rcond=None)[0]  # Find the coefficients.
            XCoeff = Coefficients[0]  # X coefficient.
            YCoeff = Coefficients[1]  # Y coefficient.
            CCoeff = Coefficients[2]  # constant term.

            # Formula of Using the above variables, z = XCoeff * x + YCoeff * y + CCoeff.
            # Leveling Laser to fit plane.
            Z_fit_all = XCoeff * cord_laser[:, 4] + YCoeff * cord_laser[:, 3] + CCoeff
            cord_laser[:, 6] = cord_laser[:, 2] - Z_fit_all
            cord_laser[:, 7] = cord_laser[:, 6] - np.nanmin(cord_laser[:, 6])

            # Move Origin to center of hole.
            cord_laser[:, 8] = cord_laser[:, 3] - center_CS[1]
            cord_laser[:, 9] = cord_laser[:, 4] - center_CS[0]

            # Rotate scan data to vertical, and leveling to positive.
            rotate_angle = -leftedge_angle
            rotationM = np.array([[np.cos(np.radians(rotate_angle)), -np.sin(np.radians(rotate_angle))],
                                  [np.sin(np.radians(rotate_angle)), np.cos(np.radians(rotate_angle))]])
            laserdata = np.dot(rotationM, cord_laser[:, [9, 8]].T).T
            laserdata = np.column_stack((laserdata, cord_laser[:, 7]))

            # Delete tether residuals and contamination.
            dellist = []
            for i in range(len(laserdata)):
                if (laserdata[i, 0] < -19) or (laserdata[i, 0] > 19):
                    dellist.append(i)
            laserdata = np.delete(laserdata, dellist, axis=0)

            # Fit plane again to processed laser data.
            Const2 = np.ones((len(laserdata[:, 2]), 1))  # Vector of ones for constant term
            Coefficients2 = np.linalg.lstsq(np.column_stack((laserdata[:, 1], laserdata[:, 0], Const2)), laserdata[:, 2], rcond=None)[0]  # Find the coefficients
            XCoeff2 = Coefficients2[0]  # X coefficient.
            YCoeff2 = Coefficients2[1]  # Y coefficient.
            CCoeff2 = Coefficients2[2]  # constant term.
            # Re-leveling laser to new fit plane.
            Z_fit_all2 = XCoeff2 * laserdata[:, 1] + YCoeff2 * laserdata[:, 0] + CCoeff2
            laserdata[:, 3] = laserdata[:, 2] - Z_fit_all2

            # Positive leveling.
            laserdata[:, 4] = laserdata[:, 3] - np.nanmin(laserdata[:, 3])

            # Store processed data.
            data_processed[laser_id_ind-1][cubeind] = laserdata[:, [0, 1, 4]]

            # Reorientation of all data.
            data_processed[laser_id_ind-1][cubeind][:, 1] = laserdata[:, 1] * (-1)

            # Find mid cross-section profile, crown is longitudinal, xcrown is transverse at the middle of laser.
            crown = []
            for i in range(len(laserdata)):
                if (-0.5 * Reslution <= data_processed[laser_id_ind-1][cubeind][i, 0] <= 0.5 * Reslution):
                    crown.append(data_processed[laser_id_ind-1][cubeind][i, [1, 2]])
            crown = np.array(crown)

            xcrown = []
            for i in range(len(laserdata)):
                if (-1 * Reslution <= data_processed[laser_id_ind-1][cubeind][i, 1] <= Reslution):
                    xcrown.append(data_processed[laser_id_ind-1][cubeind][i, [0, 2]])
            xcrown = np.array(xcrown)

            crown = crown[crown[:, 0].argsort()]
            xcrown = xcrown[xcrown[:, 0].argsort()]
            # Shift crown to mid = 0.
            midind = round(len(crown) / 2)
            crown[:, 1] = crown[:, 1] - crown[midind, 1]
            midindx = round(len(xcrown) / 2)
            xcrown[:, 1] = xcrown[:, 1] - xcrown[midindx, 1]
            # Store crown and xcrown data.
            data_crownprofiles[laser_id_ind-1][cubeind] = crown
            data_xcrownprofiles[laser_id_ind-1][cubeind] = xcrown

            # Calculate crown
            edgedistance = 10  # pixel numbers.
            data_crowns[corwndataindex-1, 0] = 0 - 0.5 * (crown[edgedistance, 1] + crown[-edgedistance-1, 1])  # positive means, middle is higher than edges.

            # Calculate xcrown
            xedgedistance = 10  # pixel numbers.
            data_crowns[corwndataindex-1, 1] = 0 - xcrown[edgedistance, 1]  # positive means, middle is higher than edges.
            data_crowns[corwndataindex-1, 2] = 0 - xcrown[-edgedistance-1, 1]  # positive means, middle is higher than edges.

            corwndataindex += 1
            laser_id_ind += 1

# endregion  

# ------------------------------ Raw data plots ------------------------------ #
#region Plot 1: raw data
fig1 = plt.figure(figsize=(15, 10))
for cubeind in range(len(cube_ids)):
    cube_id = cube_ids[cubeind]
    wafer_id = wafer_ids[cubeind]
    lasername = design_infos[cubeind]
    fig1.suptitle(f"{wafer_id} - CUBE {cube_id} - RAW", fontsize=18, color='r')
    for laser_id_ind in range(len(laserIDrange)):
        ax = fig1.add_subplot(rows, cols, laser_id_ind + 1, projection='3d')
        X, Y = np.meshgrid(np.arange(data_raw[laser_id_ind][cubeind].shape[1]), 
                           np.arange(data_raw[laser_id_ind][cubeind].shape[0]))
        Z = data_raw[laser_id_ind][cubeind]
        surf = ax.plot_surface(X, Y, Z, cmap='jet')
        ax.view_init(elev=90, azim=0)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Pixel')
        ax.set_title(str(laser_id_ind + 1), fontsize=13, color='b')
        ax.set_zlim(-zlim, zlim)
        fig1.colorbar(surf, ax=ax, label='Z$(nm)$')
plt.tight_layout()
plt.savefig(os.path.join(output_path, f"{lasername}_{wafer_id}_{cube_id}_f1_Raw_Data.png"), dpi=imgqual)
plt.close(fig1)
#endregion

# ----------------------- Processed and flattened plots ---------------------- #
#region Plot 2: processed data
fig2 = plt.figure(figsize=(15, 10))
for cubeind in range(len(cube_ids)):
    cube_id = cube_ids[cubeind]
    wafer_id = wafer_ids[cubeind]
    lasername = design_infos[cubeind]
    fig2.suptitle(f"{wafer_id} - CUBE {cube_id} - PROCESSED", fontsize=18, color='r')
    for laser_id_ind in range(len(laserIDrange)):
        ax = fig2.add_subplot(rows, cols, laser_id_ind + 1, projection='3d')
        sc = ax.scatter(data_processed[laser_id_ind][cubeind][:, 0], 
                        data_processed[laser_id_ind][cubeind][:, 1], 
                        data_processed[laser_id_ind][cubeind][:, 2], 
                        c=data_processed[laser_id_ind][cubeind][:, 2], cmap='jet', marker='.')
        ax.view_init(elev=90, azim=0)
        ax.set_xlabel('X$(um)$', fontsize=10)
        ax.set_ylabel('Y$(um)$', fontsize=10)
        ax.set_xlim(200, 350)
        ax.set_ylim(50, 600)
        ax.set_title(str(laser_id_ind + 1), fontsize=13, color='b')
        ax.set_zlim(-zlim, zlim)
        fig2.colorbar(sc, ax=ax, label='Z$(nm)$')
plt.tight_layout()
plt.savefig(os.path.join(output_path, f"{lasername}_{wafer_id}_{cube_id}_f2_Processed_Data.png"), dpi=imgqual)
plt.close(fig2)
#endregion

# --------------------------------- Filtering -------------------------------- #
#region Filtering
# Function to filter data based on the threshold
def filter_data(data, threshold):
    # Check if data is empty or has only one row
    if len(data) == 0 or len(data) == 1:
        return data
    
    # Initial filtering based on the threshold for the change in y-value
    initial_filtered_data = data[np.abs(np.diff(data[:, 1], prepend=data[0, 1])) <= threshold]
    
    # Remove first points if they exceed the threshold compared to the filtered data
    while len(initial_filtered_data) > 1 and np.abs(initial_filtered_data[0, 1] - initial_filtered_data[1, 1]) > threshold:
        initial_filtered_data = initial_filtered_data[1:]
    
    return initial_filtered_data

# Function to filter out points that differ from an average of the 60 surrounding points
def advanced_filter(data, window_size=60, threshold=1):
    if len(data) <= window_size:
        return data
    
    filtered_data = []
    half_window = window_size // 2
    
    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        surrounding_points = np.concatenate((data[start_idx:i], data[i+1:end_idx]))
        
        if len(surrounding_points) > 0:
            surrounding_avg = np.mean(surrounding_points[:, 1])
            if np.abs(data[i, 1] - surrounding_avg) <= threshold:
                filtered_data.append(data[i])
    
    return np.array(filtered_data)

# Loop through each profile and filter the data
for cubeind in range(len(cube_ids)):
    for laser_id_ind in range(len(laserIDrange)):
        # Filter crown profiles
        data_crownprofiles[laser_id_ind][cubeind] = filter_data(data_crownprofiles[laser_id_ind][cubeind], delta_threshold)
        
        # Filter xcrown profiles
        data_xcrownprofiles[laser_id_ind][cubeind] = filter_data(data_xcrownprofiles[laser_id_ind][cubeind], delta_threshold)

# Loop through each profile and filter the data using advanced_filter function
for cubeind in range(len(cube_ids)):
    for laser_id_ind in range(len(laserIDrange)):
        # Filter crown profiles using advanced_filter function
        data_crownprofiles[laser_id_ind][cubeind] = advanced_filter(data_crownprofiles[laser_id_ind][cubeind], 60, anomaly_threshold)
        
        # Filter xcrown profiles using advanced_filter function
        data_xcrownprofiles[laser_id_ind][cubeind] = advanced_filter(data_xcrownprofiles[laser_id_ind][cubeind], 60, anomaly_threshold)

print("Filtering completed.")
#endregion

# ------------------------ Seperate laser profile plots ------------------------ #
#region Plot 3: Individual Profiles
indlegend = [str(laser_id) for laser_id in laserIDrange]

for cubeind in range(len(cube_ids)):
    cube_id = cube_ids[cubeind]
    wafer_id = wafer_ids[cubeind]
    lasername = design_infos[cubeind]
    fig3, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig3.suptitle(f"{wafer_id} - CUBE {cube_id}", fontsize=18, color='r')
    
    for laser_id_ind in range(len(laserIDrange)):
        axs[0].plot(data_crownprofiles[laser_id_ind][cubeind][:, 0], data_crownprofiles[laser_id_ind][cubeind][:, 1], '.')
        axs[0].set_xlabel('Y$(um)$', fontsize=10)
        axs[0].set_ylabel('Z$(nm)$', fontsize=10)
        axs[0].set_title('Laser Crown Profile', fontsize=13, color='b')
        axs[0].grid(True)
    
    axs[0].legend(indlegend, loc='center left', bbox_to_anchor=(1, 0.5))
    
    for laser_id_ind in range(len(laserIDrange)):
        axs[1].plot(data_xcrownprofiles[laser_id_ind][cubeind][:, 0], data_xcrownprofiles[laser_id_ind][cubeind][:, 1], '.')
        axs[1].set_xlabel('X$(um)$', fontsize=10)
        axs[1].set_ylabel('Z$(nm)$', fontsize=10)
        axs[1].set_title('Laser XCrown Profile', fontsize=13, color='b')
        axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{lasername}_{wafer_id}_{cube_id}_f3_Crown_Profiles.png"), dpi=imgqual)
    plt.close(fig3)

print("Crown profiles plots have been saved successfully.")
#endregion

# ------------------------ Crown Variation Comparisons ----------------------- #
#region Plot 4: Crown Plots
# Plot crowns values
fig4 = plt.figure(figsize=(14, 7))

# Define legend labels
if group_by_design_info:
    unique_design_infos = list(set(design_infos))
    legendnames = unique_design_infos
else:
    legendnames = [f"{wafer_ids[ii]} {cube_ids[ii]}" for ii in range(len(cube_ids))]

# Subplot 1: YCrown
ax1 = fig4.add_subplot(1, 3, 1)
ax1.set_title('Laser YCrown', fontsize=13, color='b')
ax1.set_xlabel('Laser Sample Index(#)', fontsize=12, color='b')
ax1.set_ylabel('YCrown(nm)', fontsize=12, color='b')
ax1.grid(True)

if group_by_design_info:
    for design_ind, design_info in enumerate(unique_design_infos):
        xData = []
        yData = []
        for cubeind in [i for i, x in enumerate(design_infos) if x == design_info]:
            xData.extend(laserIDrange + [np.nan])
            yData.extend(data_crowns[laserIDrange[0] + len(laserIDrange) * cubeind - 1:len(laserIDrange) * cubeind, 0].tolist() + [np.nan])
        ax1.plot(xData, yData, 'd-', color=colours_design_organised[design_ind])
else:
    for ii in range(len(cube_ids)):
        ax1.plot(laserIDrange + [np.nan], data_crowns[laserIDrange[0] + len(laserIDrange) * ii - 1:len(laserIDrange) * ii, 0].tolist() + [np.nan], 'd-', color=colors[ii])

ax1.legend(legendnames, loc='best', ncol=1)

# Subplot 2: XCrown_P
ax2 = fig4.add_subplot(1, 3, 2)
ax2.set_title('Laser XCrown_P', fontsize=13, color='b')
ax2.set_xlabel('Laser Sample Index(#)', fontsize=12, color='b')
ax2.set_ylabel('XCrown_P(nm)', fontsize=12, color='b')
ax2.grid(True)

if group_by_design_info:
    for design_ind, design_info in enumerate(unique_design_infos):
        xData = []
        yData = []
        for cubeind in [i for i, x in enumerate(design_infos) if x == design_info]:
            xData.extend(laserIDrange + [np.nan])
            yData.extend(data_crowns[laserIDrange[0] + len(laserIDrange) * cubeind - 1:len(laserIDrange) * cubeind, 1].tolist() + [np.nan])
        ax2.plot(xData, yData, 'd-', color=colours_design_organised[design_ind])
else:
    for ii in range(len(cube_ids)):
        ax2.plot(laserIDrange + [np.nan], data_crowns[laserIDrange[0] + len(laserIDrange) * ii - 1:len(laserIDrange) * ii, 1].tolist() + [np.nan], 'd-', color=colors[ii])

ax2.legend(legendnames, loc='best', ncol=1)

# Subplot 3: XCrown_N
ax3 = fig4.add_subplot(1, 3, 3)
ax3.set_title('Laser XCrown_N', fontsize=13, color='b')
ax3.set_xlabel('Laser Sample Index(#)', fontsize=12, color='b')
ax3.set_ylabel('XCrown_N(nm)', fontsize=12, color='b')
ax3.grid(True)

if group_by_design_info:
    for design_ind, design_info in enumerate(unique_design_infos):
        xData = []
        yData = []
        for cubeind in [i for i, x in enumerate(design_infos) if x == design_info]:
            xData.extend(laserIDrange + [np.nan])
            yData.extend(data_crowns[laserIDrange[0] + len(laserIDrange) * cubeind - 1:len(laserIDrange) * cubeind, 2].tolist() + [np.nan])
        ax3.plot(xData, yData, 'd-', color=colours_design_organised[design_ind])
else:
    for ii in range(len(cube_ids)):
        ax3.plot(laserIDrange + [np.nan], data_crowns[laserIDrange[0] + len(laserIDrange) * ii - 1:len(laserIDrange) * ii, 2].tolist() + [np.nan], 'd-', color=colors[ii])

ax3.legend(legendnames, loc='best', ncol=1)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'AllSamples_f4_Crown_values.png'), dpi=imgqual)
plt.close(fig4)

print("Crown values plots have been saved successfully.")
#endregion

# ------------------------ Reshape data_crowns arrays ------------------------ #
# Define data_crowns_reshape
data_crowns_reshape = data_crowns[:, 0].reshape((len(laserIDrange), len(cube_ids)))
data_xcrownsP_reshape = data_crowns[:, 1].reshape((len(laserIDrange), len(cube_ids)))
data_xcrownsN_reshape = data_crowns[:, 2].reshape((len(laserIDrange), len(cube_ids)))

print("Data reshaping completed.")

# --------------------------------- Box Plots -------------------------------- #
#region Plot 5: Box Plots
# Box plot
fig5 = plt.figure(figsize=(14, 7))

# Define legend labels and colors
if group_by_design_info:
    unique_design_infos = list(set(design_infos))
    legendnames = unique_design_infos
    colors_to_use = colours_design_organised
else:
    legendnames = [f"{wafer_ids[ii]} {cube_ids[ii]}" for ii in range(len(cube_ids))]
    colors_to_use = colors

# Subplot 1: YCrown
ax1 = fig5.add_subplot(1, 3, 1)
ax1.grid(True)
ax1.set_title('Laser YCrown Distribution', fontsize=13, color='b')
ax1.set_xlabel('Laser Sample Index(#)', fontsize=12, color='b')
ax1.set_ylabel('YCrown(nm)', fontsize=12, color='b')

if group_by_design_info:
    yData = []
    group = []
    colorIndex = []
    for design_ind, design_info in enumerate(unique_design_infos):
        for cubeind in [i for i, x in enumerate(design_infos) if x == design_info]:
            yData.extend(data_crowns[laserIDrange[0] + len(laserIDrange) * cubeind - 1:len(laserIDrange) * cubeind, 0].tolist())
            group.extend([design_ind] * len(laserIDrange))
            colorIndex.extend([design_ind] * len(laserIDrange))
    ax1.boxplot(yData, positions=group, patch_artist=True)
else:
    ax1.boxplot(data_crowns_reshape, patch_artist=True)

ax1.set_xticklabels(legendnames, rotation=45)
ax1.legend(legendnames, loc='best', ncol=1)

# Subplot 2: XCrown_P
ax2 = fig5.add_subplot(1, 3, 2)
ax2.grid(True)
ax2.set_title('Laser XCrown_P Distribution', fontsize=13, color='b')
ax2.set_xlabel('Laser Sample Index(#)', fontsize=12, color='b')
ax2.set_ylabel('XCrown_P(nm)', fontsize=12, color='b')

if group_by_design_info:
    yData = []
    group = []
    colorIndex = []
    for design_ind, design_info in enumerate(unique_design_infos):
        for cubeind in [i for i, x in enumerate(design_infos) if x == design_info]:
            yData.extend(data_crowns[laserIDrange[0] + len(laserIDrange) * cubeind - 1:len(laserIDrange) * cubeind, 1].tolist())
            group.extend([design_ind] * len(laserIDrange))
            colorIndex.extend([design_ind] * len(laserIDrange))
    ax2.boxplot(yData, positions=group, patch_artist=True)
else:
    ax2.boxplot(data_xcrownsP_reshape, patch_artist=True)

ax2.set_xticklabels(legendnames, rotation=45)
ax2.legend(legendnames, loc='best', ncol=1)

# Subplot 3: XCrown_N
ax3 = fig5.add_subplot(1, 3, 3)
ax3.grid(True)
ax3.set_title('Laser XCrown_N Distribution', fontsize=13, color='b')
ax3.set_xlabel('Laser Sample Index(#)', fontsize=12, color='b')
ax3.set_ylabel('XCrown_N(nm)', fontsize=12, color='b')

if group_by_design_info:
    yData = []
    group = []
    colorIndex = []
    for design_ind, design_info in enumerate(unique_design_infos):
        for cubeind in [i for i, x in enumerate(design_infos) if x == design_info]:
            yData.extend(data_crowns[laserIDrange[0] + len(laserIDrange) * cubeind - 1:len(laserIDrange) * cubeind, 2].tolist())
            group.extend([design_ind] * len(laserIDrange))
            colorIndex.extend([design_ind] * len(laserIDrange))
    ax3.boxplot(yData, positions=group, patch_artist=True)
else:
    ax3.boxplot(data_xcrownsN_reshape, patch_artist=True)

ax3.set_xticklabels(legendnames, rotation=45)
ax3.legend(legendnames, loc='best', ncol=1)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'AllSamples_f5_Crown_values_boxplot.png'), dpi=imgqual)
plt.close(fig5)

print("Box plots have been saved successfully.")
#endregion

# ------------------------ Combined Crown Profile Plot ----------------------- #
#region Plot 6: Combined Crown Profile Plot
# Plot all crown profiles together
fig6 = plt.figure(figsize=(13, 7))

# Define legend labels
if group_by_design_info:
    unique_design_infos = list(set(design_infos))
    legendnames = unique_design_infos
else:
    legendnames = [f"{wafer_ids[ii]} {cube_ids[ii]}" for ii in range(len(cube_ids))]

# Subplot 1: YCrown Profile
ax1 = fig6.add_subplot(1, 2, 1)
ax1.set_title('Laser YCrown Profile', fontsize=13, color='b')
ax1.set_xlabel('Y$(um)$', fontsize=10)
ax1.set_ylabel('Z$(nm)$', fontsize=10)
ax1.grid(True)
ax1.set_xlim(-150, 150)

if group_by_design_info:
    for design_ind, design_info in enumerate(unique_design_infos):
        xData = []
        yData = []
        for cubeind in [i for i, x in enumerate(design_infos) if x == design_info]:
            for laser_id_ind in laserIDrange:
                xData.extend(data_crownprofiles[laser_id_ind][cubeind][:, 0])
                yData.extend(data_crownprofiles[laser_id_ind][cubeind][:, 1])
        ax1.plot(xData, yData, '.', color=colours_design_organised[design_ind])
else:
    for cubeind in range(len(cube_ids)):
        xData = []
        yData = []
        for laser_id_ind in laserIDrange:
            xData.extend(data_crownprofiles[laser_id_ind][cubeind][:, 0])
            yData.extend(data_crownprofiles[laser_id_ind][cubeind][:, 1])
        ax1.plot(xData, yData, '.', color=colors[cubeind])

ax1.legend(legendnames, loc='best', ncol=1)

# Subplot 2: XCrown Profile
ax2 = fig6.add_subplot(1, 2, 2)
ax2.set_title('Laser XCrown Profile', fontsize=13, color='b')
ax2.set_xlabel('X$(um)$', fontsize=10)
ax2.set_ylabel('Z$(nm)$', fontsize=10)
ax2.grid(True)
ax2.set_xlim(-20, 20)

if group_by_design_info:
    for design_ind, design_info in enumerate(unique_design_infos):
        xData = []
        yData = []
        for cubeind in [i for i, x in enumerate(design_infos) if x == design_info]:
            for laser_id_ind in laserIDrange:
                xData.extend(data_xcrownprofiles[laser_id_ind][cubeind][:, 0])
                yData.extend(data_xcrownprofiles[laser_id_ind][cubeind][:, 1])
        ax2.plot(xData, yData, '.', color=colours_design_organised[design_ind])
else:
    for cubeind in range(len(cube_ids)):
        xData = []
        yData = []
        for laser_id_ind in laserIDrange:
            xData.extend(data_xcrownprofiles[laser_id_ind][cubeind][:, 0])
            yData.extend(data_xcrownprofiles[laser_id_ind][cubeind][:, 1])
        ax2.plot(xData, yData, '.', color=colors[cubeind])

ax2.legend(legendnames, loc='best', ncol=1)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'AllSamples_f6_All_Crown_profiles.png'), dpi=imgqual)
plt.close(fig6)

print("All crown profiles plots have been saved successfully.")
#endregion

# --------------------------- Calculate Statistics --------------------------- #
#region Calculate Statistics
# Initialize tables
crown_distribtable = np.nan * np.zeros((5, len(cube_ids)))
xcrownP_distribtable = np.nan * np.zeros((5, len(cube_ids)))
xcrownN_distribtable = np.nan * np.zeros((5, len(cube_ids)))

# Calculate statistics for crown
for iii in range(len(cube_ids)):
    data = data_crowns_reshape[:, iii]
    pd = norm.fit(data)
    crown_distribtable[0, iii] = pd[0]  # mean
    crown_distribtable[1, iii] = pd[1]  # sigma
    crown_distribtable[2, iii] = np.max(data)
    crown_distribtable[3, iii] = np.min(data)
    crown_distribtable[4, iii] = crown_distribtable[2, iii] - crown_distribtable[3, iii]

# Calculate statistics for xcrownP
for iii in range(len(cube_ids)):
    data = data_xcrownsP_reshape[:, iii]
    pd = norm.fit(data)
    xcrownP_distribtable[0, iii] = pd[0]  # mean
    xcrownP_distribtable[1, iii] = pd[1]  # sigma
    xcrownP_distribtable[2, iii] = np.max(data)
    xcrownP_distribtable[3, iii] = np.min(data)
    xcrownP_distribtable[4, iii] = xcrownP_distribtable[2, iii] - xcrownP_distribtable[3, iii]

# Calculate statistics for xcrownN
for iii in range(len(cube_ids)):
    data = data_xcrownsN_reshape[:, iii]
    pd = norm.fit(data)
    xcrownN_distribtable[0, iii] = pd[0]  # mean
    xcrownN_distribtable[1, iii] = pd[1]  # sigma
    xcrownN_distribtable[2, iii] = np.max(data)
    xcrownN_distribtable[3, iii] = np.min(data)
    xcrownN_distribtable[4, iii] = xcrownN_distribtable[2, iii] - xcrownN_distribtable[3, iii]
#endregion

# --------------------------- Final Comparison Plot -------------------------- #
#region Plot 7: Final Comparison Plot
# Define unique design infos and corresponding colors
unique_design_infos = list(set(design_infos))

fig7, ax = plt.subplots(figsize=(11, 7))

if group_by_design_info:
    # Initialize plot handles for legend
    handles = []
    legend_labels = []

    # Plot points and annotations with colors determined by designinfo
    for design_ind, design_info in enumerate(unique_design_infos):
        for ii in range(len(cube_ids)):
            if design_infos[ii] == design_info:
                ax.plot(crown_distribtable[0, ii], xcrownP_distribtable[0, ii], '.', markersize=15, color=colours_design_organised[design_ind])
                ax.text(crown_distribtable[0, ii], xcrownP_distribtable[0, ii] - 2, f"{wafer_ids[ii]} {cube_ids[ii]}", color=colours_design_organised[design_ind])
                if design_info not in legend_labels:
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colours_design_organised[design_ind], markersize=10))
                    legend_labels.append(design_info)

    for design_ind, design_info in enumerate(unique_design_infos):
        for ii in range(len(cube_ids)):
            if design_infos[ii] == design_info:
                ax.plot(crown_distribtable[0, ii], xcrownN_distribtable[0, ii], 'd', markersize=8, markerfacecolor=colours_design_organised[design_ind], markeredgecolor=colours_design_organised[design_ind])
                ax.text(crown_distribtable[0, ii], xcrownN_distribtable[0, ii] - 2, f"{wafer_ids[ii]} {cube_ids[ii]}", color=colours_design_organised[design_ind])
                if design_info not in legend_labels:
                    handles.append(plt.Line2D([0], [0], marker='d', color='w', markerfacecolor=colours_design_organised[design_ind], markersize=10))
                    legend_labels.append(design_info)

    # Add legend to the right of the plot
    ax.legend(handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
else:
    # Plot points and annotations with original colors
    for ii in range(len(cube_ids)):
        ax.plot(crown_distribtable[0, ii], xcrownP_distribtable[0, ii], '.', markersize=15, color=colors[ii])
        ax.text(crown_distribtable[0, ii], xcrownP_distribtable[0, ii] - 2, f"{wafer_ids[ii]} {cube_ids[ii]}", color=colors[ii])

    for ii in range(len(cube_ids)):
        ax.plot(crown_distribtable[0, ii], xcrownN_distribtable[0, ii], 'd', markersize=8, markerfacecolor=colors[ii], markeredgecolor=colors[ii])
        ax.text(crown_distribtable[0, ii], xcrownN_distribtable[0, ii] - 2, f"{wafer_ids[ii]} {cube_ids[ii]}", color=colors[ii])

ax.set_xlabel(r'\textbf{Mean of YCrown distribution(nm)}', fontsize=12, color='b')
ax.set_ylabel(r'\textbf{Mean of XCrown distribution(nm)}', fontsize=12, color='b')
ax.set_title(r'\textbf{Wafer Samples Comparison by mean of Bowing}', fontsize=13, color='b')
ax.set_ylim([np.min(data_crowns[:, 1:3]) - 10, np.max(data_crowns[:, 1:3]) + 10])
ax.set_xlim([np.min(data_crowns[:, 0]) - 10, np.max(data_crowns[:, 0]) + 10])
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'AllSamples_f7_Wafer_Sample_Comparison_by_mean_of_bowing.png'), dpi=imgqual)
print("All crown profiles plots have been saved successfully.")
#endregion

# ------------------------- Form and Save Spreadsheet ------------------------ #
#region From Spreadsheet
# Initialize new rows for differences in mean crown height, XCrown_P, and XCrown_N
crown_delta = []
xcrownP_delta = []
xcrownN_delta = []
waferID_delta = []

# Loop through each unique wafer ID and calculate the differences
unique_wafer_ids = np.unique(wafer_ids)

for wafer_id in unique_wafer_ids:
    # Find the corresponding columns for edge and center cubes with the same wafer ID
    edge_col = np.where((np.array(cube_ids) == '167') & (np.array(wafer_ids) == wafer_id))[0]
    center_col = np.where((np.array(cube_ids) == '161') & (np.array(wafer_ids) == wafer_id))[0]
    
    if edge_col.size > 0 and center_col.size > 0:
        # Calculate the differences
        crown_diff = crown_distribtable[0, center_col] - crown_distribtable[0, edge_col]
        xcrownP_diff = xcrownP_distribtable[0, center_col] - xcrownP_distribtable[0, edge_col]
        xcrownN_diff = xcrownN_distribtable[0, center_col] - xcrownN_distribtable[0, edge_col]
        
        # Check if the wafer ID already exists in the waferID_delta array
        if wafer_id not in waferID_delta:
            # Store the differences and wafer ID in the new rows
            crown_delta.append(crown_diff[0])
            xcrownP_delta.append(xcrownP_diff[0])
            xcrownN_delta.append(xcrownN_diff[0])
            waferID_delta.append(wafer_id)

# Prepare data for writing to Excel
outputfilename = 'Laser_bowing_data_statistics.xlsx'
rownames = ['mean', 'sigma', 'max', 'min', 'range']
crownnames = ['Crown'] * 5
xcrownPnames = ['XCrown_P'] * 5
xcrownNnames = ['XCrown_N'] * 5

# Create a dictionary to hold the data
data = {
    'Unit(nm)': [''] + [''] * 4 + [''] * 5 + [''] * 5 + ['Wafer ID'] + ['Crown delta (Centre - Edge)'] + ['XCrown_P delta (Centre - Edge)'] + ['XCrown_N delta (Centre - Edge)'],
    '': [''] + rownames + rownames + rownames + [''] + crown_delta + xcrownP_delta + xcrownN_delta,
    'Design Info': [''] + design_infos + [''] * 4 + [''] * 5 + [''] * 5 + waferID_delta,
    'Wafer ID': [''] + wafer_ids + [''] * 4 + [''] * 5 + [''] * 5,
    'Cube ID': [''] + cube_ids + [''] * 4 + [''] * 5 + [''] * 5,
    'Crown': [''] + crownnames + [''] * 5 + [''] * 5,
    'XCrown_P': [''] + xcrownPnames + [''] * 5 + [''] * 5,
    'XCrown_N': [''] + xcrownNnames + [''] * 5 + [''] * 5,
}

# Create a DataFrame
df = pd.DataFrame(data)

# Write the DataFrame to an Excel file
with pd.ExcelWriter(os.path.join(output_path, outputfilename), engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False, header=False)

print("Excel file has been generated successfully.")
#endregion

# -------------------------------- Final steps ------------------------------- #
#region Final Steps
# Save data to a .mat file
scipy.io.savemat(os.path.join(output_path, 'AllLaserSampleData.mat'), {
    'design_infos': design_infos,
    'cube_ids': cube_ids,
    'wafer_ids': wafer_ids,
    'data_raw': data_raw,
    'data_processed': data_processed,
    'data_crownprofiles': data_crownprofiles,
    'data_xcrownprofiles': data_xcrownprofiles,
    'data_crowns': data_crowns,
    'data_crowns_reshape': data_crowns_reshape,
    'data_xcrownsP_reshape': data_xcrownsP_reshape,
    'data_xcrownsN_reshape': data_xcrownsN_reshape,
    'crown_distribtable': crown_distribtable,
    'xcrownP_distribtable': xcrownP_distribtable,
    'xcrownN_distribtable': xcrownN_distribtable,
    'Parameters': Parameters,
    'ParametersValue': ParametersValue
})

# Display a message box at completion
root = tk.Tk()
root.withdraw()  # Hide the root window
messagebox.showinfo("Data Analysis Completed", f"Data Analysis is Completed!\n\nCheck results in folder:\n{output_path}")

# Show all plots
plt.show()
#endregion