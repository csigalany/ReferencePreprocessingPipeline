import astropy.io.fits as fits
import numpy as np

'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
FUNCTIONS
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''
def interpolateImages(image1, image2, dist1I, distI2):
    imageInterp = (image1 * distI2 + image2 * dist1I) / (dist1I + distI2)
    return imageInterp

'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
FUNCTIONS FOR PIPELINE
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''
#Get data from a fits file
def getData(link, name, ext):
    FITSHandle = fits.open(link+name+ext)
    #Uncomment this if you are testing, and need file info.
    #FITSHandle.info()

    hdr = FITSHandle[0].header
    #Uncomment this if you are testing, and need file info.
    #print(repr(hdr))
    data = FITSHandle[0].data.astype("float32")

    return data


#Store data to fits file
def putData(data, link, name):
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(link+name+'.fits', overwrite='true')

#Apply prefilter. Prefilter is interpolated between existing prefilter images.
#data -> what to apply it on, 24 images in it: [wvl,pol,x,y]
#wvltsData -> array containing the wvlts of the data
#prefilte -> prefilter data, the 49 images
#prefScale -> prefilter will be divided by this scale factor
#wvltsPref -> all the wavelengths of the prefilter
#direction = 1 -> multiply with prefilter
#direction = -1 -> divide by prefilter
def applyPrefilter(data, wvltsData, prefilter, prefScale, wvltsPref, direction):
    print("...Applying prefilter...")
    prefToApply = np.zeros((6,prefilter.shape[1],prefilter.shape[2]))

    for i in range(0,6):
        wvlCurr = wvltsData[i]
        valueClosest = min(wvltsPref, key=lambda x:abs(x-wvlCurr))
        indexClosest = wvltsPref.index(valueClosest)
        if (valueClosest < wvlCurr):
            indexBefore = indexClosest
            indexAfter = indexClosest + 1
        else:
            indexAfter = indexClosest
            indexBefore = indexClosest - 1

        dist1I = abs(wvltsPref[indexBefore] - wvltsData[i])
        distI2 = abs(wvltsPref[indexAfter] - wvltsData[i])

        prefToApply[i,:,:] = interpolateImages(prefilter[indexBefore],
        prefilter[indexAfter], dist1I, distI2)

        #Remove scale factor from prefilter
        prefToApply[i,:,:] = prefToApply[i,:,:] / prefScale

    if(data.shape[2] != prefToApply.shape[1]):
        FOV_Start_y = int(prefToApply.shape[1]/2 - data.shape[2]/2)
        FOV_End_y = int(prefToApply.shape[1]/2 + data.shape[2]/2)
        prefToApply = prefToApply[:,FOV_Start_y:FOV_End_y,:]
    if(data.shape[3] != prefToApply.shape[2]):
        FOV_Start_x = int(prefToApply.shape[2]/2 - data.shape[3]/2)
        FOV_End_x = int(prefToApply.shape[2]/2 + data.shape[3]/2)
        prefToApply = prefToApply[:,:,FOV_Start_x:FOV_End_x]

    dataPrefApplied = np.zeros(data.shape)
    for i in range(0,4):
        if(direction == 1):
            dataPrefApplied[:,i,:,:] = data[:,i,:,:] * prefToApply
        elif(direction == -1):
            dataPrefApplied[:,i,:,:] = data[:,i,:,:] / prefToApply / 8
        else:
            print("Invalid direction! Must be 1 (mult) or -1 (div).")
    return dataPrefApplied

#Apply Dark image
#data -> what to apply it on, 24 images in it: [wvl,pol,x,y]
#scalingdata -> data will be divided by this scale factor
#dark -> dark data, 1 image
#scalingdark -> dark will be divided by this scale factor
#direction = 1 -> add dark
#direction = -1 -> subtract dark
def applyDark(data, scalingdata, dark, scalingdark, direction):
    print("...Apply dark...")
    if (dark.shape[0]<data.shape[2] or dark.shape[1]<data.shape[3]):
        print("Size of dark is too small!", dark.shape[0], data.shape[0], dark.shape[1], data.shape[1])
        raise ValueError("Size of dark is too small for the input data.")

    FOV_Start_x = int(dark.shape[1]/2 - data.shape[3]/2)
    FOV_End_x = int(dark.shape[1]/2 + data.shape[3]/2)
    FOV_Start_y = int(dark.shape[0]/2 - data.shape[2]/2)
    FOV_End_y = int(dark.shape[0]/2 + data.shape[2]/2)

    #print("DetectorArea Dark:",FOV_Start_x, FOV_End_x)

    dark = (dark[FOV_Start_y:FOV_End_y,FOV_Start_x:FOV_End_x].astype("float") / scalingdark) * scalingdata
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            if direction == 1:
                data[i,j,:,:] = data[i,j,:,:] + dark
            elif direction == -1:
                data[i,j,:,:] = data[i,j,:,:] - dark
            else:
                print("Invalid direction! Use 1 for +, -1 for -.")
                raise ValueError("Invalid direction.")
    return data

#Apply Flat field
#data -> what to apply it on, 24 images in it: [wvl,pol,x,y]
#flat -> flat data, 1/6/24 images in it: [wvl,pol,x,y], only 24 is implemented
#noFlats -> number of flats, so far only 24 implemented
#scalingflat-> flat will be divided by this scale factor
#direction = 1 -> multiply by flat
#direction = -1 -> divide by flat
def applyFlat(data, flat, noFlats, scalingflat, direction):
    print("...Apply flat...")

    if (noFlats == 24):
        if (flat.shape[0]<data.shape[0] or flat.shape[1]<data.shape[1] or flat.shape[2]<data.shape[2] or flat.shape[3]<data.shape[3]):
            print("Size of flat is too small!")
            raise ValueError("Size of flat is too small for the input data.")
        if direction == 1:
            for i in range(0,6):
                    data[i,j,:,:] = data[i,j,:,:] * (flat[i,j,:,:]*scalingflat)
        elif direction == -1:
            data = data / flat
        else:
            print("Invalid direction! Use 1 for *, -1 for /.")
            raise ValueError("Invalid direction.")
    elif (noFlats == 6):
        if (flat.shape[0]<data.shape[0] or flat.shape[1]<data.shape[2] or flat.shape[2]<data.shape[3]):
            print("Size of flat is too small!")
            raise ValueError("Size of flat is too small for the input data.")
        if direction == 1:
            for i in range(0,6):
                    data[i,:,:,:] = data[i,:,:,:] * (flat[i,:,:]*scalingflat)
        elif direction == -1:
            for i in range(0,6):
                    data[i,:,:,:] = data[i,:,:,:] / (flat[i,:,:]*scalingflat)
        else:
            print("Invalid direction! Use 1 for *, -1 for /.")
            raise ValueError("Invalid direction.")
    else:
        print("Option not yet implemented!")
        raise ValueError("Unavailable option.")

    print("Flat mean:", np.mean(flat))
    return data

#Modulate data
#data -> what to apply it on, 24 images in it: [wvl,pol,x,y]
#matrix -> modulation matrix[4,4,x,y]
#scalingmatrix-> modulation matrix will be divided by this scale factor
def modulateImages(data, matrix, scalingmatrix):
    print("...Modulate...")
    if (matrix.shape[0]<data.shape[3] or matrix.shape[1]<data.shape[2]):
        print("Size of matrix is too small!")
        raise ValueError("Size of demodulation matrix is too small for the input data.")

    reshapedMM = np.zeros((data.shape[2], data.shape[3], 4, 4))

    FOV_Start_x = int(matrix.shape[0]/2 - data.shape[3]/2)
    FOV_End_x = int(matrix.shape[0]/2 + data.shape[3]/2)
    FOV_Start_y = int(matrix.shape[1]/2 - data.shape[2]/2)
    FOV_End_y = int(matrix.shape[1]/2 + data.shape[2]/2)

    for i in range(0,4):
        for j in range(0,4):
            reshapedMM[:,:,i,j] = matrix[FOV_Start_y:FOV_End_y,
            FOV_Start_x:FOV_End_x,i*4+j] / scalingmatrix

    InputMod = np.zeros((data.shape[2], data.shape[3], data.shape[1], data.shape[0]))
    ResultMod = np.zeros((data.shape[2], data.shape[3], data.shape[1], data.shape[0]))

    for j in range(0,data.shape[0]):
        for i in range(0,data.shape[1]):
            InputMod[:,:,i,j] = data[j,i,:,:]
        #Do modulation
        ResultMod[:,:,:,j:j+1] = np.matmul(reshapedMM,InputMod[:,:,:,j:j+1])

    #Resort the result into viewable image
    OutpMod = np.zeros((data.shape[0],data.shape[1],data.shape[2], data.shape[3]))
    for j in range(0,data.shape[0]):
        for i in range(0,data.shape[1]):
            OutpMod[j,i,:,:] = ResultMod[:,:,i,j]
            #scaling 4 is from the artefact of the matrix multiplication

    #remove NaNs
    OutpMod[np.where(OutpMod!=OutpMod)] = 8388608

    return OutpMod

#Demodulate data
#data -> what to apply it on, 24 images in it: [wvl,pol,x,y]
#matrix -> demodulation matrix[4,4,x,y]
#scalingmatrix-> demodulation matrix will be divided by this scale factor
def demodulateImages(data, matrix, scalingmatrix):
    print("...Demodulate...")
    print(matrix.shape)
    print(data.shape)
    if (matrix.shape[3]<data.shape[3] or matrix.shape[2]<data.shape[2]):
        print("Size of matrix is too small!")
        raise ValueError("Size of demodulation matrix is too small for the input data.")

    print(data.shape)
    reshapedMM = np.zeros((data.shape[2], data.shape[3], 4, 4))
    print(reshapedMM.shape)

    FOV_Start_x = int(matrix.shape[3]/2 - data.shape[3]/2)
    FOV_End_x = int(matrix.shape[3]/2 + data.shape[3]/2)
    FOV_Start_y = int(matrix.shape[2]/2 - data.shape[2]/2)
    FOV_End_y = int(matrix.shape[2]/2 + data.shape[2]/2)

    for i in range(0,4):
        for j in range(0,4):
            reshapedMM[:,:,i,j] = matrix[i,j,FOV_Start_y:FOV_End_y,
            FOV_Start_x:FOV_End_x] / scalingmatrix / 4
            #scaling 4 is from the artefact of the matrix multiplication

    InputMod = np.zeros((data.shape[2], data.shape[3], data.shape[1],
    data.shape[0]))
    ResultMod = np.zeros((data.shape[2], data.shape[3], data.shape[1],
    data.shape[0]))

    for j in range(0,data.shape[0]):
        for i in range(0,data.shape[1]):
            InputMod[:,:,i,j] = data[j,i,:,:]
        #Do modulation
        ResultMod[:,:,:,j:j+1] = np.matmul(reshapedMM,InputMod[:,:,:,j:j+1])

    #Resort the result into viewable image
    OutpMod = np.zeros((data.shape[0], data.shape[1], data.shape[2],
    data.shape[3]))
    for j in range(0,data.shape[0]):
        for i in range(0,data.shape[1]):
            OutpMod[j,i,:,:] = ResultMod[:,:,i,j]

    return OutpMod

#Reorder images for RTE inversion in DPU
#inp -> what to apply it on, 24 images in it: [wvl,pol,x,y], should be Stokes
#wvl -> number of wavelengths
#pol -> number of polarisation states
def reorderImg2RTE(inp, wvl, pol):
    print("...Reordering for RTE...")
    if (wvl*pol == 24):
        #reorder for output
        array_reord = np.ones((inp.shape[2]*inp.shape[3],24))
        for k in range(0,inp.shape[2]):
            for l in range (0,inp.shape[3]):
                for i in range(0,pol):
                    for j in range(0,wvl):
                        array_reord[k*(inp.shape[3])+l,i+j+(wvl-1)*i] = inp[j,i,k,l]

        return array_reord
    else:
        print("Mode not yet implemented.")
        raise ValueError("Option not implemented.")

#Reorder images from RTE inversion in DPU to view them
#inp -> what to apply it on, 24 images in it: [wvl,pol,x,y], should be Stokes
#noIm -> number of images in total (implmented only for 4)
def reorderRTE2Img(inp, noIm):
    if (noIm == 4):
        outp = np.zeros(inp.shape)
        outp_flat = np.zeros((noIm, inp.shape[1]*inp.shape[2]))
        inp_flat = inp.flatten()
        for i in range(0,noIm):
            outp_flat[i,:] = inp_flat[i::noIm]
            outp[i,:,:] = outp_flat[i,:].reshape(inp.shape[1],inp.shape[2])
        return outp
    else:
        print("Mode not yet implemented.")
        raise ValueError("Option not implemented.")

#Correct I->Q,U,V
#coeffQ, coeffU, coeffV -> cross-talk coefficients
#This function needs more work!!!!
def correctI2QUV(inp, coeffQ, coeffU, coeffV, scaling):
    print("I2Q (a):", coeffQ * scaling)
    print("I2U (b):", coeffU * scaling)
    print("I2V (d):", coeffV * scaling)

    corrImQ = inp[:,0,:,:] * coeffQ * scaling
    corrImU = inp[:,0,:,:] * coeffU * scaling
    corrImV = inp[:,0,:,:] * coeffV * scaling

    inp[:,1,:,:] = inp[:,1,:,:] - corrImQ
    inp[:,2,:,:] = inp[:,2,:,:] - corrImU
    inp[:,3,:,:] = inp[:,3,:,:] - corrImV
    return inp

#Correct V->Q,U
#coeffQ, coeffU, coeffV -> cross-talk coefficients
#This function needs more work!!!!
def correctV2QU(inp, coeffQ, coeffU, scaling):
    print("V2Q (c):", coeffQ * scaling)
    print("V2U (e):", coeffU * scaling)
    
    corrImQ = inp[:,3,:,:] * coeffQ * scaling
    corrImU = inp[:,3,:,:] * coeffU * scaling

    inp[:,1,:,:] = inp[:,1,:,:] - corrImQ
    inp[:,2,:,:] = inp[:,2,:,:] - corrImU
    
    return inp

'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
OTHER USEFUL FUNCTIONS
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
'''
def prepareDataForHelix(datain, rows, cols, conInt, wvlRel):
    thumbnaildata = np.zeros((cols,rows,4,6))

    print("contint:", conInt)
    for i in range(0,4):
        for j in range(0,6):
            thumbnaildata[:,:,i,j] = datain[j,i,0:cols,0:rows]

    thumbnaildata[np.where(thumbnaildata!=datain)]=0
    thumbnaildata[thumbnaildata == np.inf] = 0
    thumbnaildata[thumbnaildata == -np.inf] = 0
    thumbnaildata[thumbnaildata == np.nan] = 0

    #create new fits with extension
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU()
    hdu3 = fits.ImageHDU()
    new_hdul = fits.HDUList([hdu1, hdu2, hdu3])
    print(thumbnaildata.shape)
    new_hdul[0].data = thumbnaildata

    #HMI:
    #new_hdul[1].data = [6173.341-0.1725, 6173.341-0.1035, 6173.341-0.0345,
    #6173.341+0.0345, 6173.341+0.1035, 6173.341+0.1725]
    #PHI:
    new_hdul[1].data = [6173.341+wvlRel[0], 6173.341+wvlRel[1],
    6173.341+wvlRel[2], 6173.341+wvlRel[3], 6173.341+wvlRel[4],
    6173.341+wvlRel[5]]
    new_hdul[2].data = np.ones((cols,rows))*conInt
    return new_hdul
