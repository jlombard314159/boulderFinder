from math import cos, asin, sqrt, pi, sin
from functools import partial
from scipy import spatial


def extractUniqueValues(listOfRoutes,key):

    values = [x[key] for x in listOfRoutes]
    uniqueValues = [list(x) for x in set(tuple(x) for x in values)]

    return uniqueValues

def removeKeyFromList(listOfRoutes):

    listOfRoutes = [ x.pop('close_points_index') for x in listOfRoutes]

    return listOfRoutes

def cartesian(latitude, longitude, elevation = 0):
    # Convert to radians
    latitude = latitude * (pi / 180)
    longitude = longitude * (pi / 180)

    R = 6371 # 6378137.0 + elevation  # relative to centre of the earth
    X = R * cos(latitude) * cos(longitude)
    Y = R * cos(latitude) * sin(longitude)
    Z = R * sin(latitude)
    return (X, Y, Z)

def transformCoords(listOfPlaces):

    places = []
    for _, row in enumerate(listOfPlaces):
        coordinates = row['lnglat']
        cartesian_coord = cartesian(*coordinates)
        places.append(cartesian_coord)

    return places

def findTreePairs(coordList, distanceValue = 0.4):

    tree = spatial.KDTree(coordList) # r = 1 is about 1000m I think
    closest = tree.query_pairs(r =distanceValue)

    return closest

def addEmptyNeighbor(dictionary):

    dictionary['neighbors'] = []

    return dictionary

def updateAreaWithPairs(listOfPlaces,closestPairs):

    listVersion = list(closestPairs)

    listOfPlaces = [addEmptyNeighbor(x) for x in listOfPlaces]

    for _, data in enumerate(listVersion):

        firstIndex = data[0]
        secondIndex = data[1]

        listOfPlaces[firstIndex]['neighbors'].append(secondIndex)
        listOfPlaces[secondIndex]['neighbors'].append(firstIndex)

    return listOfPlaces

def addIndexToList(listOfPoints):

    for index, data in enumerate(listOfPoints):

        data['index'] = index

    return listOfPoints

def extractPointsWithNeighbors(listOfPoints, neighborCount = 2):

    listOfPoints = addIndexToList(listOfPoints)

    nearbyNeighbors = [x for x in listOfPoints if len(x['neighbors']) >= neighborCount]

    return nearbyNeighbors

def convertToDict(all_areas):

    new_dict = {}
    for item in all_areas:
            name = item['index']
            new_dict[name] = item

    return new_dict

def grabNeighbors(pointsToQuery, all_points_dict: dict):

    new_neighbors = []
    for item in pointsToQuery:

        all_values = [all_points_dict[x] for x in item['neighbors']]

        new_neighbors.append(all_values)

    return new_neighbors