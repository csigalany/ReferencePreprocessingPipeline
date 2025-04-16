'''
Created on Feb 2, 2018

@author: albert
'''

import numpy as np
from numpy import *
import astropy.io.fits as pf
from array import array
import os
import math


NaN = -2147483648/256
maxv = 2147483647/256
minv = (-2147483648+1)/256

def putData_to_bigendian_float(data, link, name_noext):
    data = data.flatten()
    float_array = array('f', data)
    #float_array.byteswap(inplace=True)

    #...I store this and open it again as a workaround...
    fh = open(link+name_noext+"_float_littleE.bin", "wb")
    float_array.tofile(fh)
    fh.close()

    fh = open(link+name_noext+"_float_littleE.bin", "r")
    data = np.fromfile(fh, dtype="float32")
    data.byteswap(inplace=True)
    fh.close()
    #I then delete this File
    os.remove(link+name_noext+"_float_littleE.bin")

    fh = open(link+name_noext+"_float_bigE.bin", "wb")
    data.tofile(fh)
    fh.close()

def FITS2bin_bigendian_float(link, name_noext):
    FITSfile = pf.open(link+name_noext+".fits")
    data = FITSfile[0].data
    #data[data<minv] = NaN
    #data[data>maxv] = NaN
    #datafixed = double2fixed(data)
    data = data.flatten()
    float_array = array('f', data)
    #float_array.byteswap(inplace=True)

    #...I store this and open it again as a workaround...
    fh = open(link+name_noext+"_float_littleE.bin", "wb")
    float_array.tofile(fh)
    fh.close()

    fh = open(link+name_noext+"_float_littleE.bin", "r")
    data = np.fromfile(fh, dtype="float32")
    data.byteswap(inplace=True)
    fh.close()
    #I then delete this File
    os.remove(link+name_noext+"_float_littleE.bin")

    fh = open(link+name_noext+"_float_bigE.bin", "wb")
    data.tofile(fh)
    fh.close()

def bin2FITS_n_bigendian_float(link, name_noext, ext, wvl, pol, x, y):
    f = open(link+name_noext+"_float_bigE"+ext,"r")
    print (link+name_noext+"_float_bigE")

    data = np.fromfile(f,'>f4')

    float_array = data.astype("float32")
    print("FLOATARRAY")
    #size = int(float_array.shape[0]/wvl/pol)
    #sizeNew = int(math.sqrt(size))


    if(wvl == 1 and pol==1):
        float_array = float_array.reshape((x, y))
    elif(pol==1):
        float_array = float_array.reshape((wvl, x, y))
    else:
        float_array = float_array.reshape((wvl, pol, x, y))

    intermImage = pf.PrimaryHDU(data = float_array)
    FITSImage = pf.HDUList([intermImage])
    FITSImage.writeto(link+name_noext+".fits",overwrite='true')
