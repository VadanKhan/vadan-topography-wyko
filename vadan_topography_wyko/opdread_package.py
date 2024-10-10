import numpy as np
import struct
import codecs

def read_wyko_opd(filename):
    # Open the file in binary mode and read its content
    with open(filename, 'rb') as fid:
        E = fid.read()

    # Convert the content to the specified encoding
    E2 = codecs.encode(E.decode('ISO-8859-1'), 'ISO-8859-1')

    # Find the first instance of the word 'Directory'
    ind = E2.find(b'Directory')
    print(ind)

    # Initialize variables
    Directorybytes = 6002
    Block_len = 24
    BlockID = 1
    BLOCKS = []

    # Parse the directory
    while ind + 21 <= Directorybytes:
        block = {
            'name': E2[ind:ind+16].decode('ISO-8859-1'),
            'type': struct.unpack('h', E2[ind+16:ind+18])[0],
            'Length': struct.unpack('i', E2[ind+18:ind+22])[0],
            'BlockTail': struct.unpack('h', E2[ind+22:ind+24])[0]
        }
        BLOCKS.append(block)
        BlockID += 1
        ind += Block_len

    # Extract parameters
    Parameters = ['Pixel_size', 'Wavelength']
    ParametersValue = {}

    for param in Parameters:
        for block in BLOCKS:
            if block['name'].startswith(param):
                # Find the index of the current block
                current_block_index = BLOCKS.index(block)

                # Sum the lengths of all blocks before the current block
                # NOTE THAT THIS ADDS 2, NOT 3. 
                ind = 2 + sum(b['Length'] for b in BLOCKS[:current_block_index])

                # Extract the raw bytes
                raw_bytes = E2[ind:ind+block['Length']]
                ParametersValue[param] = struct.unpack('f', raw_bytes)[0]
                
                # Print the block, raw bytes, and block index for debugging
                print(f"Block for {param}: {block}")
                print(f"Raw bytes for {param}: {list(raw_bytes)}")
                print(f"Block index for {param}: {current_block_index}")
                break

    return BLOCKS, ParametersValue