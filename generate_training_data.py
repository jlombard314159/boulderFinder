from generatingData.neighboringPoints import convertToDict, extractPointsWithNeighbors, \
     findTreePairs, grabNeighbors, transformCoords, updateAreaWithPairs

from scrapingData.openBetaScraper import extractAreasWithBoulders, parseAllFiles, \
    splitAreasAndSubAreas, extractRoutesWithBoulders, \
    extractAreasWithBoulders, extractStates, syncUpRoutesWithArea, \
    correctGPS

from scrapingData.qaqcOpenBetaData import removeByKeyword, removeIncorrectGPS, \
    duplicateGPSRemover, modifyNameForOutput

from generatingData.trainingDataCreation import modifyImageName, parallelProcessAllPixels,\
     parallelizeMapCreation, singleCreateMap

from generatingData.generateBoundingBox import grabBaseXML, grabImageLabel, \
    generateAllXML, mergePixelsToImages

import ee


def main():

    ee.Initialize()
    baseDir = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/'

    fullData = parseAllFiles(fileParent=baseDir + '/rawData/')

    areaData, routeData = splitAreasAndSubAreas(parsedAllData=fullData)

    boulderRoutesOnly = extractRoutesWithBoulders(listOfRoutes=routeData)

    areaData = extractAreasWithBoulders(listOfAreas=areaData,
                                        listOfRoutes=boulderRoutesOnly)

    areaData = extractStates(listOfAreas=areaData,
                            listOfStates=['Wyoming', 'Colorado', 'California', 'Utah', 'Montana', 'Idaho', 'Oregon',
                                        'Washington'])

    subsettedRoutes = syncUpRoutesWithArea(listOfRoutes=boulderRoutesOnly,
                                        listOfAreas=areaData)

    areaClean = removeIncorrectGPS(areaData)
    routeClean = removeIncorrectGPS(subsettedRoutes, key='parent_lnglat')

    areaClean = duplicateGPSRemover(listOfAreas=areaClean)

    areaClean = removeByKeyword(listOfAreas=areaClean,
        keywords = ['Concrete','Building','Builder','CentralPark','Downtown'])

    routeClean = syncUpRoutesWithArea(listOfRoutes=routeClean,
                                        listOfAreas=areaClean)


    correctAreaGPS = correctGPS(areaClean,key = 'lnglat')

    correctAreaGPS = modifyNameForOutput(correctAreaGPS)

    kdCoord = transformCoords(listOfPlaces=correctAreaGPS)

    closestPairs = findTreePairs(kdCoord)

    pairedAreas = updateAreaWithPairs(listOfPlaces=correctAreaGPS, closestPairs = closestPairs)

    areasWithNeighbors = extractPointsWithNeighbors(pairedAreas, neighborCount=1)

    # grabThree = [x for x in pairedAreas if len(x['neighbors']) == 40]

    # testSet = [pairedAreas[i] for i in grabThree[0]['neighbors']]

    parallelizeMapCreation(areasWithNeighbors, fxnToParallel=singleCreateMap,
        outputPath = baseDir + 'testOutput/images/')

    neighborsList = grabNeighbors(areasWithNeighbors, convertToDict(pairedAreas))

    pixelCoords = parallelProcessAllPixels(areasWithNeighbors, neighborsList)
    
    baseXML = grabBaseXML(baseXMLPath=baseDir + '/generatingData/')
    
    imagePaths = grabImageLabel(path = baseDir + 'testOutput/images/')
    
    modifiedImagePaths = modifyImageName(imagePaths)

    finalImageList = mergePixelsToImages(pixelCoords, modifiedImagePaths)

    generateAllXML(baseXML,imageList=finalImageList,outputXMLPath=baseDir+'testOutput/labels/')

    return None


if __name__ == '__main__':   
    main()