# Wyko Data Processor for Optical Topography Data Samples
Developed by **Vadan Khan**, R&D Physicist at STST. 

## Overview
This script processes data files with the extension `.opd`, which are data formats for several [Bruker tools](https://www.bruker.com/en/products-and-solutions/test-and-measurement/3d-optical-profilers.html) (common in the micro-electronics industry). This repository is built to handle **Wyko topographical .opd image files** from the [Wyko Optical Profiler](https://www.bruker.com/en/products-and-solutions/test-and-measurement/3d-optical-profilers/hd9800.html). It is designed to handle many topographies simultaneously, generating and saving plots and statistics tables automatically. The main tasks/requirements include:

1. **`.opd` File Reading API**: A custom built file parser to read `.opd` files from Wyko Tools
2. **Topographical Image Processing**: flatten and align input topgraphies, and extract profiles along a surface for statistic extraction.
3. **Calculating Physical Parameters** for each sample, including height of the centre compared to edge ("Crown height") and plane angles of the sample.
4. **Auto-plotting analysis results**
5. **Generating statistics summary tables in Excel**, with the excel exports designed to be portable to other analysis tools such as JMP.

## Key Project Features
- **Edge Detection**: Uses parameters to detect edges in the data.
- **Laser Orientation**: Estimates rotation and coordinate systems.
- **Plane Fitting**: Flattens the image data for analysis.
- **Profile Extraction**: Extracts YCrown and XCrown profiles, applying filters to remove unphysical data points.
- **Data Analysis Suite**: A large variety of plots to analyse and compare your data to your requirements. Includes a bespoke interactive 3D topography plot of any input image.
- **Statistics Calculation**: Computes mean, sigma, max, min, and range for bowing values.
- **Dynamic Array Sizes**: Supports different array sizes within an analysis batch.
- **Variable Grouping of Data**: Supports grouping of input datasets via `DESIGN_INFOS`, which alters output plots to compare and analyse population trends
- **Recycling Data**: Uses previously exported "Crown Number" and "Crown Profile" Excel data to save analysis time.

### Exported Data Files
- **Raw Topography Results**: This exports the key parameter outputs from the topography reading, so that the data can be portable for custom analysis of a population. The statistics export excludes datapoints outputs of an IQR range, so this stores that data for reference.
    - **Crown Numbers in Excel File**: This stores the calculated crown numbers and Angles, with seperate excel files for each dataset in `DATASETS`
    - **Crown Profiles in Excel File**: This stores the profiles extracted from topographical images, which are the lines of points for across the y midsection and x midsection. With seperate excel files for each dataset in `DATASETS`.
- **Statistics on Crown Heights**: Exports statistics on Crown Heights and Angles, and Correlation between the two, into an excel file that has a row entry for each dataset in `DATASETS`. This has been designed to be portable to analysis in JMP, but also compatible for any toher format using Excel.

### Exported Plots
- **Crown Height Box Plots**: Bowing (Crown height) values of each dataset in `DATASETS` (or seperate groups if using `DESIGN_INFOS`) compared in a box plot. The individual values are plotted to the right of the plot for additional clarity as to what makes up each population.
- **Combined Profile Plots**: Plots of the Profiles of samples, from lines drawn across the midpoint lines of 2d images. The colours are grouped by each dataset in `DATASETS` (or seperate groups if using `DESIGN_INFOS`).
- **Raw Topography Plots**: Raw topographical image colour map plots, for every input image. Seperate figure for each dataset in `DATASETS`.
- **Flattened/Aligned Topography Plots**: Flattened and Aligned (removing slopes / angles in the data) topographical colour map plot, for every input image. Seperate figure for each dataset in `DATASETS`.
- **Angle Plots**: Plot of Plane Angles of the samples (which were used to align the Flattened Topgraphy plots). These angles are plotted against the crown heights of a dataset, with a seperate figure for each dataset in `DATASETS`. 
  - Roll angle (tilt along the x axis)
  - Pitch Angle (tilt along the y axis)
  - Yaw angle (rotation about the z axis normal to the image)
- **Individual Profile Plots**: Same as the Combined Profile Plots, but with a seperate figure for each dataset in `DATASETS`. Each profile is given a seperate colour to clearly identify each sample (and any defect sample).
- **Single Sample Interactive 3D plot**: This is a custom built interactive plot for exploring the topography of a single sample. Can change the aspect ratio of the z height to accentuate any variations as per requirements.
- **Scatter Plot**: Similar Purpose as Box plots, comparing a populations of data for each dataset in `DATASETS` (specifically the mean crown height). Colours can be grouped by seperate groups if using `DESIGN_INFOS`. 2 seperate figures are generated, where one has YCrown against XCrownP, and the other YCrown against XCrownN.

## General User Guide
- `main.ipynb` should be the only file you ever need to use for day to day use. It imports function implementations from several other scripts in `vadan_topography_wyko`. Plot function definitions are generally kept in the `main.ipynb` script as these might need to be quickly altered regularly, but the rest of the code should not need adjustment commonly so is kept modular in other scripts.
- `main.ipynb` is a jupyter notebook, which means that a long script is split up into seperate cells to be run individually. This is key to enabling a dynamic analysis environment, which is quick to run and only runs the code that is needed at the time.
- **Imports**, **User Input**, **Processing and Image Analysis** cells must always be run, in that order, as this caches the main topographical file reading / image processing / calculations (the main computaion and so runtime being in the `MAIN PROCESSING` cell)
- But after those cells, you can run the cells that you require and ignore the ones you don't (preventing excessive load on unnecessary features). You can hide the code of all the cells for common use, and hide the outputs of the cells that you are not interested in.

### Important Notes
- Each dataset of .opd files should have its own folder under the `INPUTPATH`, named in a specific format (e.g., `waferID_CUBE_cubeID`).
- Each individual `.opd` file in each folder should follow either filename format: `"Row_{}_Col_{}_"` or `"row{}column{}"`

### Usage Instructions
1. Run `Package Import Section` cell.
2. Define the required inputs in the `USER INPUT / CONFIGURATION SECTION` cell, which will be the main cell you need to change to use the code:
  - `INPUTPATH`: a path to a set of data, which should have folders (names specified in `DATASETS`) that contain the `.opd` files in them
  - `OUTPUTPATH`: a path where all the code outputs are stored, will generate the required folders if they don't exist currently
  - `CAMPAIGN_NAME`: a string name that will be added onto the filename of many of the stored plots, to help differentiate plots of the same data if need be.
  - `DATASETS`: a list of strings, that contains the name of each folder "dataset" you want the code to analyse
  - `DESIGN_INFOS` and `GROUP_BY_DESIGN_INFO`: `DESIGN_INFOS` is a list of strings that can be used to name and group the datasets as you wish, where names are assigned in the same order as `DATASETS`. `GROUP_BY_DESIGN_INFO` is a boolean variable that can be set to true if you want to group using `DESIGN_INFOS`.
  - `STAMP_IDS`: these similarly have to be matched by order to `DATASETS`, and are only used in the **Statistics on Crown Heights** exports.
  - `ROWS` and `COLS`: if each dataset has the same numbering of data, can specify the number of rows and columns here.
  - `DYNAMIC_ARRAYS`, `ROW_DYNAMIC` and `COLUMN_DYNAMIC`: where one can enable dynamic array sizes (number of datapoints in each dataset), and specify the number of rows in each dataset sequentially in a list.
  - **Filter Settings**: Where one can configure the sensitivity of the two filters that are applied onto the profiles.
  - **Plotting Lims**: Where one can change the colour map scale, or y axis limits, of the plots
  - `IMGQUAL`: Changes the quality of all the plots saved
  - **Image Detection Parameters**: Specific parameters for the edge detection, should generally not be adjusted.
  - `PLOT_BY_COLUMN`: Boolean Variable that changes the way lasers are numbered in an array. If set to false, laser index number as you read across the page (like english text). If set to true laser index number increases as you go down the column (like traditional japenense script).
3. Run the `USER INPUT / CONFIGURATION SECTION` cell
4. Run the `Preprocessing Steps` cell
5. Run the `Initialisation of Data Storage Variables` cell
6. Run the `MAIN PROCESSING` cell, which runs all the image parsing and image processing and calculations. The runtime depends on the number of input images, where one can expect ~4seconds of runtime per 21 array. In other words, this can analyse around 5 images per second.
7. If you want to export Raw data or Statistics, run desired cells under **Results Exports**
8. If you want to generate Plots, run desired cells under **Plots**.

### Recycle Data cell (experimental)
- This is to save analysis time and so **can be run instead of MAIN PROCESSING if desired**
- This cell uses any previously exported "Crown Number" and "Crown Profile" Excel data. You can specify the path it checks.
- It will check any previously exported data that matches the currently analyzed datasets. If they match, it uses the old data instead of regenerating new data.
- **DO NOT USE THIS IF YOU WANT TO USE MAIN PROCESSING** (it overwrites this section).
- **THIS SECTION DOES NOT GENERATE FULL TOPOGRAPHY IMAGES, SO CANNOT GENERATE TOPOGRAPHY PLOTS**.

## Version History
- 1st released version: 'WykoDataProcessor_Laser_Stamps_FastAuto_PSI_v1.m', as a 700 line matlab script.
on July 25, 2024.
Developed by Cheng Chen, Mech CAG.

- 1st upgraded version: 'WykoDataProcessor_Laser_Stamps_VK.m'
on 24th September, 2024. 
Developed by Vadan Khan, R&D Physicist at STST. 

- Now supported here as a Python Repository. First Completed Beta Build: 10th December, 2024. 
Developed by Vadan Khan, R&D Physicist at STST. 

