import numpy as np
import struct

#region Loop to read and process all opd files.
def read_wyko_opd(filename):
    with open(filename, 'rb') as f:
        # Read the first 6002 bytes to get the header
        header = f.read(6002)
        
        # Convert header to a byte array
        E2 = np.frombuffer(header, dtype=np.uint8)

        # Initialize parameters to find
        parameters = ['Pixel_size', 'Wavelength']
        parameter_indices = []
        parameter_values = []

        # Read directory bytes and block length
        directory_bytes = 6002
        block_len = 24
        ind = np.where(E2 == ord('D'))[0]  # Find index of 'D' in the header (assuming 'Directory' starts with 'D')

        # Extract blocks
        blocks = []
        block_id = 0
        
        while ind.size > 0 and ind[0] + 21 <= directory_bytes:
            block_name = E2[ind[0]:ind[0]+16].tobytes().decode('ISO-8859-1').strip()
            block_type = struct.unpack('h', E2[ind[0]+16:ind[0]+18])[0]
            block_length = struct.unpack('i', E2[ind[0]+18:ind[0]+22])[0]
            block_tail = struct.unpack('h', E2[ind[0]+22:ind[0]+24])[0]
            
            blocks.append({
                'name': block_name,
                'type': block_type,
                'length': block_length,
                'tail': block_tail
            })
            block_id += 1
            ind = ind + block_len

        # Find parameters in blocks
        for param in parameters:
            for j, block in enumerate(blocks):
                if block['name'].startswith(param):
                    parameter_indices.append(j)
                    ind = 2 + sum(b['length'] for b in blocks[:j]) + 1  # Calculate index for parameter value
                    param_value = struct.unpack('f', E2[ind:ind + block['length']])[0]  # Extract parameter value
                    parameter_values.append(param_value)
                    break

        # Print extracted values
        for param, value in zip(parameters, parameter_values):
            print(f'{param}: {value}')

        # Read pixel data dimensions
        Xsize = struct.unpack('H', header[28:30])[0]  # Assuming Xsize is stored at this offset
        Ysize = struct.unpack('H', header[30:32])[0]  # Assuming Ysize is stored at this offset
        
        # Read pixel data
        pixel_data = f.read(Xsize * Ysize * 4)  # 4 bytes per float
        pixel_array = np.frombuffer(pixel_data, dtype=np.float32).reshape((Ysize, Xsize))
        
        # Process data (e.g., replace invalid values)
        pixel_array[pixel_array >= 1e10] = np.nan
        
        return pixel_array