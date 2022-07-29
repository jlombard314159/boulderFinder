from generatingData.trainingDataCreation import parallelizeMapCreation, singleCreateMap
uniqueID = 'JAL'

baseDir = 'D:/boulder-finder/training data_hand label/'

listOfCoords = [{'area_name':'coyote rocks', 'lnglat':[41.272830, -105.3935], 'unique-id': uniqueID + '1'},
                #{'area_name':'tunnels', 'lnglat':[40.67332, -105.85189], 'unique-id': uniqueID + '2'},
                {'area_name':'bunker1', 'lnglat':[41.229436, -105.328922], 'unique-id': uniqueID + '3'},
                {'area_name':'bunker2', 'lnglat':[41.230119, -105.321827], 'unique-id': uniqueID + '4'},
                {'area_name':'rf-roof area', 'lnglat':[40.87269, -105.57872], 'unique-id': uniqueID + '5'},
                {'area_name':'mastodon', 'lnglat':[40.85941, -105.52624], 'unique-id': uniqueID + '6'},
                {'area_name':'voo-beer crack', 'lnglat':[41.17096, -105.32999], 'unique-id': uniqueID + '8'},
                {'area_name':'roofranch', 'lnglat':[41.21777, -105.32499], 'unique-id': uniqueID + '9'}]

#listOfCoords = [{'area_name':'gandalf','lnglat':[40.665460, -105.808020],'unique-id':uniqueID + '10'}]

for data in listOfCoords:
    data['url'] = 'temp/temp/temp'


parallelizeMapCreation(listOfCoords, fxnToParallel=singleCreateMap,
    outputPath = baseDir + 'images/')