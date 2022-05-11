from collections import Counter
import re
import functools
# --------------------------------------------------------
# if two different names have same GPS -> remove
# If lng isnt negative remove or lat is negative -> remove
# if there is a single route for an area -> remove

def removeIncorrectGPS(listToRemove, key = 'lnglat'):

    correctGPS = [x for x in listToRemove if x[key][0] < 0]
    correctGPS = [x for x in correctGPS if x[key][1] > 0]

    return correctGPS

def duplicateGPSRemover(listOfAreas):

    dup_free = []
    dup_free_set = set()    

    for x in listOfAreas:
        if tuple(x['lnglat']) not in dup_free_set:
            dup_free.append(x)
            dup_free_set.add(tuple(x['lnglat']))

    return dup_free

def matchByKeyword(keyword,areaToCheck):

    if areaToCheck.isalnum() == False:
        areaToCheck = re.sub('[^A-Za-z0-9]+', '', areaToCheck)

    boolValue = bool(re.search(keyword,areaToCheck))

    return boolValue

def removeByKeyword(listOfAreas,keywords=['Concrete','Building','Builder']):

    finalList = []
    for _, value in enumerate(listOfAreas):
        
        matchByKeywordValue = functools.partial(matchByKeyword,areaToCheck = value['area_name'])
        ifAnyTrueDiscardArea = list(map(matchByKeywordValue,keywords)) ##Fix
        
        if True not in ifAnyTrueDiscardArea:
            finalList.append(value)

    return finalList

def removeSingleRoutes(listOfRoutes):

    numberOfOccur = Counter([x['parent_sector'] for x in listOfRoutes])

    #Find keys where value is 1

    valuesExtract = list(numberOfOccur.values())

    return None


def modifyNameForOutput(listOfRoutes):

    modifiedList = []
    for _, data in enumerate(listOfRoutes):
        data['area_name'] = re.sub('[^A-Za-z0-9]+', '', data['area_name'])
        modifiedList.append(data)

    return modifiedList
