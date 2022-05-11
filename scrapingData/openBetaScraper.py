import json
from os import listdir, walk


def parseAllFiles(fileParent):
    subFolders = listdir(fileParent)

    files = []
    for i in range(len(subFolders)):
        _, _, filenames = next(walk(fileParent + subFolders[i]))
        filenames = [subFolders[i] + '/' + x for x in filenames]
        files.append(filenames)

    allFiles = [item for sublist in files for item in sublist]

    fullPaths = [fileParent + x for x in allFiles]

    data = []
    for i in fullPaths:
        with open(i) as f:
            data.append(json.load(f))

    return data


def splitAreasAndSubAreas(parsedAllData, parentKey='url', childKey='fa'):
    parentData = [x for x in parsedAllData if parentKey in x]
    childData = [x for x in parsedAllData if childKey in x]
    return parentData, childData


def extractRoutesWithBoulders(listOfRoutes):

    bouldersOnly = [x for x in listOfRoutes if 'boulder' in x['type']]
    extractMetaData = [x['metadata'] for x in bouldersOnly]

    return extractMetaData


def extractAreasWithBoulders(listOfAreas, listOfRoutes):
    
    parentSector = [x['mp_sector_id'] for x in listOfRoutes]

    setParent = set(parentSector)

    extractedAreas = [x for x in listOfAreas if x['url'].split('/')[-2] in setParent]

    return extractedAreas


def extractStates(listOfAreas, listOfStates):
    setOfStates = set(listOfStates)
    subsettedState = [x for x in listOfAreas if x['us_state'] in setOfStates]

    return subsettedState

def syncUpRoutesWithArea(listOfRoutes, listOfAreas):
    parentLatLong = [x['lnglat'] for x in listOfAreas]
    
    unique_data = set([tuple(x) for x in parentLatLong])
    subsettedRoutes = [x for x in listOfRoutes if tuple(x['parent_lnglat']) in unique_data]

    return subsettedRoutes

def grabUniqueLatLong(listToUnique,key):

    lngLatOnly = [x[key] for x in listToUnique]

    uniqueItems = [list(item) for item in set(tuple(row) for row in lngLatOnly)]

    return uniqueItems

def correctGPS(listToCorrect,key):

    for data in listToCorrect:
        data[key].reverse()

    return listToCorrect