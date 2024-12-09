# Wyko Data Processor for Optical Topography Data Samples

This script processes Wyko data files with the extension `.fc.opd`. It is designed to handle multiple stamp samples simultaneously, generating and saving plots and statistics tables automatically. The main tasks include:

1. **Reading all `.opd` files**.
2. **Processing raw data** to extract crown profiles and bowing.
3. **Calculating statistics** for each stamp sample.
4. **Auto-plotting analysis results**.
5. **Auto-saving plots and statistics summary tables in Excel**.

## User Inputs Required
- **Input and output paths**.
- **Sample information**.
- **Plot colors**.
- **Parameters for edge detection** (only change if needed).

## Output Files
The script generates various output files, including:
- Raw data plots.
- Processed data plots.
- Crown and XCrown profiles.
- Bowing values.
- Boxplots of bowing values.
- Overlayed profiles of all samples.
- Comparison of stamp samples by mean bow.
- An Excel table with laser bowing data statistics.

## Key Features
- **Edge Detection**: Uses parameters to detect edges in the data.
- **Laser Orientation**: Estimates rotation and coordinate systems.
- **Plane Fitting**: Flattens the image data for analysis.
- **Profile Extraction**: Extracts crown and XCrown profiles, applying filters to remove unphysical data points.
- **Statistics Calculation**: Computes mean, sigma, max, min, and range for bowing values.

## Important Notes
- The script requires the vector lengths of `waferIDs`, `cubeIDs`, `designinfos`, and `machinenames` to match the number of samples being processed.
- Each stamp sample should have its own folder under the input path, named in a specific format (e.g., `waferID_CUBE_cubeID`).

## Example Configuration
- **Input Path**: `C:\\Users\\762093\\Documents\\WYKO_DATA`
- **Output Path**: `C:\\Users\\762093\\Documents\\WYKO_DATA\\OUTPUTS\\output_debug`
- **Datasets**: `['NEHZX_CUBE_167', 'NEHZX_CUBE_161']`
- **Colors**: `[[1, 0.5, 0], [0.5, 0, 0.5], [1, 0.5, 0], [0, 0, 1], [1, 0, 1]]`
- **Edge Detection Parameter**: `3`
- **Delta Threshold**: `3 nm`
- **Anomaly Threshold**: `25 nm`

Recycle Data: 
- This section uses any previously exported "Crown Number" and "Crown Profile" excel data. You can specify the path it checks. 
- It will check any preivously exported that matches the currently analysed datasets, if they do match it instead uses the old data than regenerating the new data.
- This is to save analysis time
- DO NOT USE THIS IF YOU WANT TO USE MAIN PROCESSING (it overrwrites this section)
- THIS SECTION DOES NOT GENERATE LOAD IMAGES, SO CANNOT GENERATE TOPOGRAPHY PLOTS


1st released version: 'WykoDataProcessor_Laser_Stamps_FastAuto_PSI_v1', as a 700 line matlab script.
on July 25, 2024.
Developed by Cheng Chen, Mech CAG.

1st upgraded version: 'WykoDataProcessor_Laser_Stamps_VK'
on 24/09/2024.
Now supported here as a Python Repository
Developed by Vadan Khan, R&D Physicist at STST. 
