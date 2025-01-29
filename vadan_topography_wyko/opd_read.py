import numpy as np
import struct
import codecs
import matplotlib.pyplot as plt


def read_wyko_opd(filename):
    # print("meow")
    # Open the file in binary mode and read its content
    with open(filename, "rb") as fid:
        E = fid.read()

    # Convert the content to the specified encoding
    E2 = codecs.encode(E.decode("ISO-8859-1"), "ISO-8859-1")

    # Find the first instance of the word 'Directory'
    ind_meta = E2.find(b"Directory")
    ind_dict = ind_meta
    # print(ind_meta)

    # Initialize variables
    Directorybytes = 6002
    Block_len = 24
    BlockID = 1
    BLOCKS = []

    # Parse the directory
    while ind_meta + 21 <= Directorybytes:
        block = {
            "name": E2[ind_meta : ind_meta + 16].decode("ISO-8859-1"),
            "type": struct.unpack("h", E2[ind_meta + 16 : ind_meta + 18])[0],
            "Length": struct.unpack("i", E2[ind_meta + 18 : ind_meta + 22])[0],
            "BlockTail": struct.unpack("h", E2[ind_meta + 22 : ind_meta + 24])[0],
        }
        BLOCKS.append(block)
        BlockID += 1
        ind_meta += Block_len

    # Extract parameters
    Parameters = ["Pixel_size", "Wavelength"]
    ParametersValue = {}

    for param in Parameters:
        for block in BLOCKS:
            if block["name"].startswith(param):
                # Find the index of the current block
                current_block_index = BLOCKS.index(block)

                # Sum the lengths of all blocks before the current block
                # NOTE THAT THIS ADDS 2, NOT 3.
                ind_meta = 2 + sum(b["Length"] for b in BLOCKS[:current_block_index])

                # Extract the raw bytes
                raw_bytes = E2[ind_meta : ind_meta + block["Length"]]
                ParametersValue[param] = struct.unpack("f", raw_bytes)[0]

                # Print the block, raw bytes, and block index for debugging
                # print(f"Block for {param}: {block}")
                # print(f"Raw bytes for {param}: {list(raw_bytes)}")
                # print(f"Block index for {param}: {current_block_index}")
                break

    # Read RAW_DATA
    # Calculate the starting index for reading raw data by summing the lengths of the first three
    # blocks
    ind_main = ind_dict + BLOCKS[0]["Length"] + BLOCKS[1]["Length"] + BLOCKS[2]["Length"]
    # print(ind_main)

    # Read the X and Y dimensions of the data
    Xsize = int(struct.unpack("h", E2[ind_main : ind_main + 2])[0])
    Ysize = int(struct.unpack("h", E2[ind_main + 2 : ind_main + 4])[0])

    # Read the number of bytes per data point
    Nbytes_data_per_pixel = int(struct.unpack("h", E2[ind_main + 4 : ind_main + 6])[0])
    # print(Nbytes_data_per_pixel)

    # Move the index forward by 6 bytes to the start of the pixel data
    ind_main += 0

    # Initialize an array to hold the pixel data
    pixeldata = np.zeros(Xsize * Ysize, dtype=np.float32)

    # Read each pixel's data
    for pid in range(Xsize * Ysize):
        pixeldata[pid] = struct.unpack(
            "f",
            E2[
                ind_main
                + (pid * Nbytes_data_per_pixel) : ind_main
                + (pid * Nbytes_data_per_pixel)
                + Nbytes_data_per_pixel
            ],
        )[0]

    # Find and replace invalid pixel values with NaN
    idx = np.where(pixeldata >= 1e10)
    pixeldata[idx] = np.nan

    # Calculate the raw image data using the wavelength parameter
    VSIWavelength = ParametersValue["Wavelength"]
    # Scale by wavelength
    image_raw = np.reshape(pixeldata, (Xsize, Ysize)) * VSIWavelength
    -np.nanmean(
        pixeldata
    ) * VSIWavelength  # NOTE THE ORDER OF XSIZE AND YSIZE, differs between matlab and python

    # Step 1 to getting image in desired format (vertically aligned)
    image_raw_flipped = np.transpose(image_raw)

    # Step 1 to getting image in desired format (agreeing with orientation in previous code
    # and previous conventions for angles)
    image_raw = np.flipud(image_raw_flipped)

    # # Debug plot of image_raw
    # plt.figure()
    # plt.imshow(image_raw, cmap='jet')
    # plt.title('image_raw')
    # plt.xlabel('Column Pixel')
    # plt.ylabel('Row Pixel')
    # plt.colorbar()

    # Return the blocks, parameters, and raw image data
    return BLOCKS, ParametersValue, image_raw
