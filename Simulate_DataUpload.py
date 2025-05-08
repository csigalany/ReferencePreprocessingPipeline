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

linkData = "/Users/albert/Dropbox/PHD_work/Data/Paper2/Calib/HRT/"
linkNameMod = "Mod/polcal_hires_0-2047_0-2047_wobble_corrected_avg"
linkNameFlat = "Flat/0024150120_im"

#------------------------------------------------------------------
#------------------------------------------------------------------
#Reading input
#------------------------------------------------------------------
#------------------------------------------------------------------
#4x4
Mod = Functions_pipeline.getData(linkData, linkNameMod, ".fits")  
#24x2048x2048
Flat24 = Functions_pipeline.getData(linkData, linkNameFlat, ".fits")  
# shape 24x2048x2048 into 6x4x2048x2048
Flat = Flat24.reshape(6, 4, 2048, 2048)
Functions_pipeline.putData(Flat, linkData, linkNameFlat+"_6by4")

#We generate the demodulation matrix here, but could be also read from a file
#4x4
Demod = np.linalg.inv(Mod)

#------------------------------------------------------------------
#------------------------------------------------------------------
#On-ground part
#------------------------------------------------------------------
#------------------------------------------------------------------
#Create uploaded flat
#Demodulate the flat
Flat_demod = Functions_pipeline.matrixMul1D(Flat, Demod, 1)
#Pick out the continuum wavelengths, and the 4 polarisation states at continuum, 9 images total
#Index of continuum wavelength
ContIndex = 0
#init flat that will be uploaded
Flat_upload = np.zeros((9,Flat.shape[2],Flat.shape[3]))
#Copy all wavelengths, stokes I
Flat_upload[0:6,:,:] = Flat_demod[:,0,:,:]
#Copy Stokes Q,U,V from continuum wavelength
Flat_upload[6:] = Flat_demod[ContIndex,1:,:,:]

Functions_pipeline.putData(Flat_upload, linkData, linkNameFlat+"_uploadVersion")

#------------------------------------------------------------------
#------------------------------------------------------------------
#On-board part
#------------------------------------------------------------------
#------------------------------------------------------------------
#Now create again a flat out of this
#First copy the information
Flat_demod = np.zeros((6,4,Flat.shape[2],Flat.shape[3]))
Flat_demod[:,0,:,:] = Flat_upload[0:6,:,:]
for i in range(0,6):
    Flat_demod[i,1:,:,:] = Flat_upload[6:,:,:]
#Now modulate again
Flat = Functions_pipeline.matrixMul1D(Flat_demod, Mod, 1)

#------------------------------------------------------------------
#------------------------------------------------------------------
#Storing the result
#------------------------------------------------------------------
#------------------------------------------------------------------
Functions_pipeline.putData(Flat, linkData, linkNameFlat+"_OnBoardFlat_6by4")
# shape 6x4x2048x2048 into 24x2048x2048
Flat24 = Flat.reshape(24, 2048, 2048)
Functions_pipeline.putData(Flat24, linkData, linkNameFlat+"_OnBoardFlat")


