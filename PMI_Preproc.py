'''
Created on Feb 2, 2018

@author: albert
'''
import sys
sys.path.append('.')

import numpy as np
from numpy import *

import astropy.io.fits as fits
import numpy as np
from numpy import *
from array import array
import os

import Functions_DPU_related
import Functions_pipeline

def makeDummyData(linkInput, nameInput, linkDark, nameDark, linkFlat, nameFlat, linkDemod, nameDemod):
    #################################################################
    #     ---------------- Create Dummy Data -----------------      #
    #################################################################

    #I create some dummy data for testing my pipeline.
    #Note: result to the preprocessing should be all ones.
    print("I am making dummy data now.")

    inputData = np.ones((6,4,2048,2))*2
    dark = np.ones((2048,2))
    flat = np.ones((6,4,2048,2))
    #make identity matrix
    demod = np.zeros((4,4,2048,2))
    demod[0,0] = 1
    demod[1,1] = 1
    demod[2,2] = 1
    demod[3,3] = 1

    #Now, store thsi data in binaries.
    Functions_DPU_related.putData_to_bigendian_float(inputData, linkInput, nameInput)
    Functions_DPU_related.putData_to_bigendian_float(dark, linkDark, nameDark)
    Functions_DPU_related.putData_to_bigendian_float(flat, linkFlat, nameFlat)
    Functions_DPU_related.putData_to_bigendian_float(demod, linkDemod, nameDemod)   

#################################################################
#     ----------------------- SETUP -----------------------     #
#################################################################

#Where is your data?
linkInput = "/Users/albert/Dropbox/PHD_work/Data/PMI_preprocTest/"
nameInput = "input"
linkDark = linkInput
nameDark = "dark"
linkFlat = linkInput
nameFlat = "flat"
linkDemod = linkInput
nameDemod = "demod"
linkOutput = linkInput
nameOutput = "output"
I2Q_offset = 0
I2U_offset = 0
I2V_offset = 0
I2Q_scale = 0
I2U_scale = 0
I2V_scale = 0

#If you need to create the dummy data, set this to 1. Needed only 1x.
dummyData = 1

if __name__ == '__main__':

    if (dummyData==1):
        makeDummyData(linkInput, nameInput, linkDark, nameDark, linkFlat, nameFlat, linkDemod, nameDemod)
        
    #################################################################
    #     ----------------- File conversions -----------------      #
    #################################################################

    #Convert binaries to FITS
    Functions_DPU_related.bin2FITS_n_bigendian_float(linkInput, nameInput, ".bin", 6, 4, 2048, 2)
    Functions_DPU_related.bin2FITS_n_bigendian_float(linkDark, nameDark, ".bin", 1, 1, 2048, 2)
    Functions_DPU_related.bin2FITS_n_bigendian_float(linkFlat, nameFlat, ".bin", 6, 4, 2048, 2)
    Functions_DPU_related.bin2FITS_n_bigendian_float(linkDemod, nameDemod, ".bin", 4, 4, 2048, 2)


    #################################################################
    #     --------------------- Get data ---------------------      #
    #################################################################

    inputData = Functions_pipeline.getData(linkInput, nameInput, ".fits")    
    darkData = Functions_pipeline.getData(linkDark, nameDark, ".fits")
    flatData = Functions_pipeline.getData(linkFlat, nameFlat, ".fits")    
    demodData = Functions_pipeline.getData(linkDemod, nameDemod, ".fits")

    #################################################################
    #     --------------------- Pipeline ---------------------      #
    #################################################################

    intermediateData = Functions_pipeline.applyDark(inputData, 1, darkData, 1, -1)
    intermediateData = Functions_pipeline.applyFlat(intermediateData, flatData, 24, 1, -1)
    intermediateData = Functions_pipeline.demodulateImages(intermediateData, demodData, 1)
    intermediateData = Functions_pipeline.correctI2QUV(intermediateData, I2Q_scale, I2U_scale, I2V_scale)

    #################################################################
    #     --------------------- Put data ---------------------      #
    #################################################################

    Functions_pipeline.putData(intermediateData, linkOutput, nameOutput)

    #################################################################
    #     ----------------- File conversion -----------------      #
    #################################################################

    #Convert FITS to binary
    Functions_DPU_related.FITS2bin_bigendian_float(linkOutput, nameOutput)