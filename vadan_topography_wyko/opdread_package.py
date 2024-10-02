import numpy as np
import struct
import codecs

#region Loop to read and process all opd files.
def read_wyko_opd(filename):
    # Open the file in binary mode and read its content
    with open(filename, 'rb') as fid:
        E = fid.read()

    # Convert the content to the specified encoding
    E2 = codecs.encode(E, 'ISO-8859-1')
        
    return E2