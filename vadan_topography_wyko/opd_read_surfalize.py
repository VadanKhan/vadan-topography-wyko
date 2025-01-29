import surfalize as sf
import numpy as np
import matplotlib as plt


def read_opd_surfalize(filename):
    # Load the .opd file
    opd_file = sf.load(filename)

    # Extract the blocks
    blocks = opd_file.blocks

    # Extract parameters
    parameters = opd_file.parameters
    parameters_value = {param.name: param.value for param in parameters}

    # Extract raw image data
    image_raw = opd_file.data

    # Assuming the parameters include 'Pixel_size' and 'Wavelength'
    pixel_size = parameters_value.get("Pixel_size")
    wavelength = parameters_value.get("Wavelength")

    # Process the image data
    image_raw = np.array(image_raw) * wavelength

    # Find and replace invalid pixel values with NaN
    idx = np.where(image_raw >= 1e10)
    image_raw[idx] = np.nan

    # Flip and transpose the image to match the desired format
    image_raw_flipped = np.transpose(image_raw)
    image_raw = np.flipud(image_raw_flipped)

    # Return the blocks, parameters, and raw image data
    return blocks, parameters_value, image_raw


test_file = "C:\\Users\\762093\\Documents\\WYKO_DATA\\KORAT_CUBE_161\\\Row_1_Col_1_.fc.opd"
blocks, params, image_raw = read_opd_surfalize(test_file)
# Debug plot of image_raw
plt.figure()
plt.imshow(image_raw, cmap="jet")
plt.title("image_raw")
plt.xlabel("Column Pixel")
plt.ylabel("Row Pixel")
plt.colorbar()
