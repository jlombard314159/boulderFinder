from generatingData.neighboringPoints import convertToDict, extractPointsWithNeighbors, \
     findTreePairs, grabNeighbors, transformCoords, updateAreaWithPairs

from scrapingData.openBetaScraper import parseAllFiles

from scrapingData.webScraperTools import extract_for_manual, \
 removeIncorrectLatLng, add_url

from scrapingData.qaqcOpenBetaData import removeByKeyword, \
    duplicateGPSRemover, modifyNameForOutput

from generatingData.trainingDataCreation import modifyImageName, parallelProcessAllPixels,\
     parallelizeMapCreation, singleCreateMap

from generatingData.generateBoundingBox import grabBaseXML, grabImageLabel, \
    generateAllXML, mergePixelsToImages

from generatingData.postprocess import find_empty_images, remove_file

import ee


def setup(numNeighbors = 30):

    ee.Initialize()
    baseDir = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/web_json'

    fullData = parseAllFiles(fileParent=baseDir + '/hand_picked/')

    unlistData = [item for sublist in fullData for item in sublist]

    areaClean = removeIncorrectLatLng(unlistData)

    remove_dupl = duplicateGPSRemover(listOfAreas=areaClean)

    remove_dupl = removeByKeyword(listOfAreas=remove_dupl,
        keywords = ['Concrete','Building','Builder','CentralPark','Downtown'])

    correctAreaGPS = modifyNameForOutput(remove_dupl)

    add_url(correctAreaGPS)

    kdCoord = transformCoords(listOfPlaces=correctAreaGPS)

    closestPairs = findTreePairs(kdCoord)

    pairedAreas = updateAreaWithPairs(listOfPlaces=correctAreaGPS, closestPairs = closestPairs)

    areasWithNeighbors = extractPointsWithNeighbors(pairedAreas, neighborCount=numNeighbors)

    return pairedAreas, areasWithNeighbors

def main():

    home_dir = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/'
    baseDir = 'D:/boulder-finder/training data_hand_picked_google/'

    pairedAreas, areasWithNeighbors = setup(numNeighbors=0)

    # neighborCount = [len(x['neighbors']) for x in areasWithNeighbors]
    # grabThree = [x for x in pairedAreas if len(x['neighbors']) == 5]
    # testSet = [pairedAreas[i] for i in grabThree[0]['neighbors']]

    all_area_dict = convertToDict(pairedAreas)

    neighborsList = grabNeighbors(areasWithNeighbors, all_area_dict)

    parallelizeMapCreation(areasWithNeighbors, fxnToParallel=singleCreateMap,
        outputPath = baseDir + 'images/')

    pixelCoords = parallelProcessAllPixels(areasWithNeighbors, neighborsList)
    
    baseXML = grabBaseXML(baseXMLPath=home_dir + '/generatingData/')
     
    imagePaths = grabImageLabel(path = baseDir + 'images/')
    
    modifiedImagePaths = modifyImageName(imagePaths)

    finalImageList = mergePixelsToImages(pixelCoords, modifiedImagePaths)

    generateAllXML(baseXML,imageList=finalImageList,outputXMLPath=baseDir+'labels/')

    #Noticing errors with some imagery (showing up as white if it doesn't exist)
    empty_images = find_empty_images(baseDir + 'images/')
    empty_xml = [x.replace('.png','.xml') for x in empty_images]
    empty_xml = [x.replace('images','labels') for x in empty_xml]

    remove_file(empty_images)
    remove_file(empty_xml)

    return None

def manual_moving():

    baseDir = 'D:/boulder-finder/training data_20neighbor_goog_web/to_hand_label/'

    pairedAreas, areasWithNeighbors = setup()

    manual_testing = extract_for_manual(areasWithNeighbors)

    urls = [x['url'] for x in manual_testing]

    all_point_dict = convertToDict(pairedAreas)
    
    new_neighbors = grabNeighbors(manual_testing, all_point_dict)

    parallelizeMapCreation(manual_testing, fxnToParallel=singleCreateMap,
        outputPath = baseDir + '/images/')

    # areas = [x['area_name'] for x in manual_testing]
    # extra_manual_testing = areas.index('offwidthboulder')

    pixelCoords = parallelProcessAllPixels(manual_testing,
        new_neighbors)
    
    baseXML = grabBaseXML(baseXMLPath='C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/generatingData/')
    
    imagePaths = grabImageLabel(path = baseDir + 'images/')
    
    modifiedImagePaths = modifyImageName(imagePaths)

    finalImageList = mergePixelsToImages(pixelCoords, modifiedImagePaths)

    generateAllXML(baseXML,imageList=finalImageList,outputXMLPath=baseDir+'labels/')
    
    ##FOR HAND LABELING THAT JAL GAVE UP ON CUZ IT SUCKS
    # file_names = [extract_file_name(x) for x in manual_testing]
    # xml_names = [x.replace('.png','.xml') for x in file_names]

    # input_dir = 'D:/boulder-finder/training data_20neighbor_goog_web/'
    # output_dir = input_dir + 'to_hand_label/'

    # find_and_move(input_dir + 'labels/', output_dir= output_dir, to_move = xml_names)

    # [x for x in manual_testing if x['area_name'] == '1014boulder']


if __name__ == '__main__':   
    main()