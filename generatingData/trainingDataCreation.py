from generatingData.generatingHTML import generateMap, saveConvert, \
     generateMapWithBoulders
import multiprocessing as mp
from functools import partial
import numpy as np
from PIL import Image
import io

#-------------------------------------------------------------------------------
#generating actual maps

def singleCreateMap(oneArea, 
    outputPath = 'C:/Users/jlomb/Documents/Personal Coding/Python/MP/MPExtensions/testOutput/',
    gps_key = 'lnglat'):

    tempMap = generateMap(gpsCoord = oneArea[gps_key])

    saveConvert(tempMap, imageLabel = 'train_' + oneArea['area_name'] + '__' +  oneArea['url'].split('/')[-2],
        localFolder = outputPath)

    return None

def createBoulderMap(oneArea, listOfNeighbors,
    gps_key = 'lnglat'):

    extractGPSNeighbors = [x[gps_key] for x in listOfNeighbors]

    tempMap = generateMapWithBoulders(gpsCoord = oneArea[gps_key],
        listOfNeighbors=extractGPSNeighbors)

    return tempMap

#-------------------------------------------------------------------------------------
def modifyImageName(listOfImages):

    dictList = []
    for _, data in enumerate(listOfImages):

        grabAfterDunder = data.split("__")[1]
        grabBeforePeriod = grabAfterDunder.split(".")[0]

        dictList.append({'filename':data, 'uniqueID': grabBeforePeriod})


    return dictList

def arrayNPEquality(array1, array2):

    if (array1 == array2).all():

        return True

    return False

def readInMap(createdMap):

    img_data = createdMap._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    arrayImage = np.asarray(img)

    return arrayImage

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def findBoulders(imageArray, subArray = np.array([238,75,43,255])):

    xdim = imageArray.shape[1]     
    sliceDim = imageArray.shape[2]
    pixelList = []

    for i in range(0,xdim-1):
        
        pixelSlice = imageArray[:,i,:].flatten()

        windowView = rolling_window(pixelSlice,len(subArray))
        results = np.where(np.all(windowView == subArray, axis=1))[0]

        if results.size != 0:
            colIndex = convertSliceIndex(results,sliceDim)

            listCoords = [(i,x) for x in colIndex]

            pixelList.append(listCoords)

    unlistPixel = [item for sublist in pixelList for item in sublist]

    return unlistPixel

def extractCoord(pixelCoords):

    ##BIG BIG assumption: For zoom = 20 JAL manually figured out the min
    ## number of pixels to make a square and these come in pairs of 4
    ##For another zoom, this won't work. 

    seqToGrab = len(pixelCoords)

    grabEveryFourth = pixelCoords[0:seqToGrab:4]

    return grabEveryFourth

def convertSliceIndex(slice: int, sliceDim):

    convSlice = slice / sliceDim

    intVersion = convSlice.astype(int)

    return intVersion

##If coords are 'close' then pick one
def cleanUpCoords(oneCoordSet, pixelTol = 30):

    arrayTol = [pixelTol,pixelTol]
    cleanCoords = []
    trackDuplicates = []

    for index, data in enumerate(oneCoordSet):

        pixelDifferences = abs(np.subtract(data,oneCoordSet))

        findClosePixels = np.where(np.all(pixelDifferences < arrayTol, axis=1))[0]
    
        removeCurrentIndex = findClosePixels[findClosePixels != index]

        toListIndex = removeCurrentIndex.tolist()

        trackDuplicates += toListIndex

        if(index not in trackDuplicates):
            cleanCoords.append(data)
   
    return cleanCoords

##Every single photo wil have a boulder in the middle
def add_midpoint(list_of_coords, midPoint = (683,322)):

    midPoint_list = [midPoint]

    all_coords = list_of_coords + midPoint_list

    return all_coords
#-------------------------------------------------------------------------------

def processPixelBoulder(gpsData, neighbor):

    mapToProcess = createBoulderMap(gpsData, neighbor)
    
    arrayImage = readInMap(mapToProcess)

    boulderCoords = findBoulders(arrayImage)

    extractPoint = extractCoord(boulderCoords)

    addMidpoint = add_midpoint(extractPoint)

    cleanCoordinates = cleanUpCoords(addMidpoint)

    fileName = gpsData['url'].split('/')[-2]

    dictResult = {'file':fileName, 'pixelCoords':cleanCoordinates}

    return dictResult

#------------------------------------------------------------------------------
#Parallel fxns

def parallelProcessAllPixels(gpsData, allNeighbors, fxn = processPixelBoulder):

    pool = mp.Pool(11)

    results = pool.starmap(fxn,zip(gpsData,allNeighbors))
    pool.close()

    return results

def parallelizeMapCreation(listOfAreas,fxnToParallel = singleCreateMap,
    outputPath ='C:/Users/jlomb/Documents/Personal Coding/Python/MP/MPExtensions/testOutput/',
    **kwargs):

    neighbors = kwargs.get('listOfNeighbors', None)

    pool = mp.Pool(11)

    pool.map(partial(fxnToParallel,outputPath=outputPath), listOfAreas, neighbors)
    pool.close()
    pool.join() 

    return None