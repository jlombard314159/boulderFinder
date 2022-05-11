import enum
from os import listdir, sep
from os.path import isfile, join
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import glob
from functools import partial
import re

def generateVOCCoord(coord = [683,322], default_size = [1366, 675]):

    boxLength = 30

    xMin = coord[0]-boxLength/2
    xMax = coord[0]+boxLength/2
    yMin = coord[1]-boxLength/2
    yMax = coord[1]+boxLength/2

    if (yMin <0 ): yMin = 0
    if (xMin <0 ): xMin = 0

    if (xMax > default_size[0]): xMax = default_size[0]
    if (yMax > default_size[1]): yMax = default_size[1]

    labelDict = {'xmin':int(xMin),'xmax':int(xMax),
        'ymin':int(yMin), 'ymax':int(yMax)}

    return labelDict

def generateVOCObject(vocRow):

    xmlString = r"""<object>
    <name>boulder</name>
    <occluded>0</occluded>
    <bndbox>
    <xmin>{xmin}</xmin>
    <ymin>{ymin}</ymin>
    <xmax>{xmax}</xmax>
    <ymax>{ymax}</ymax>
    </bndbox>
    </object>""".format(xmin=vocRow['xmin'], 
        ymin=vocRow['ymin'], 
        xmax=vocRow['xmax'], 
        ymax=vocRow['ymax'])

    return xmlString

def generateET(baseXML,xmlString):

    rawStringBase = str(ET.tostring(baseXML.getroot()))

    #Purely hardcoded to get rid of last annotation
    removeLastTag = rawStringBase[:-14]

    removeLastTag = removeLastTag + xmlString + "\n </annotation>"

    removeLastTag = removeLastTag.replace("\\n",'\n')
    removeLastTag = removeLastTag.replace("b'",'')

    finalXML = ET.ElementTree(ET.fromstring(removeLastTag))

    return finalXML

def grabImageLabel(path):

    onlyFiles = [f for f in listdir(path) if isfile(join(path, f))]

    return onlyFiles
  

def grabBaseXML(baseXMLPath):

    xml_file = sorted(glob.glob(baseXMLPath + sep + '*.xml'))[0]

    tree = ET.parse(xml_file)
    return tree

def modifyXMLValue(xmlToEdit,replaceTag,findTag='filename'):

    xmlToEdit.find('.//' + findTag).text = replaceTag

    return xmlToEdit

def modifyFileName(fileName):

    newFile = re.sub('boulderpixels', 'train', fileName)

    return newFile

def mergePixelsToImages(pixelList, imageList):

    for _, data in enumerate(pixelList):

        data['file']= modifyFileName(data['file'])

    for _, image in enumerate(imageList):

        image['pixels'] = [x['pixelCoords'] for x in pixelList if x['file'] in image['filename']]

    return imageList

def generateAllXML(xmlToEdit,imageList,outputXMLPath):

    for image in imageList:

        imageID = image['filename']

        xmlToEdit = modifyXMLValue(xmlToEdit = xmlToEdit, replaceTag=imageID)

        xmlAddedBoulders = ''
        
        for coord in image['pixels'][0]:
            vocRow = generateVOCCoord(coord = coord)
            xmlAddedBoulders += generateVOCObject(vocRow = vocRow)

        xmlFinal = generateET(xmlToEdit,xmlAddedBoulders)

        imageNameNoPNG = imageID.split('.')[0]

        xmlFinal.write(outputXMLPath + imageNameNoPNG + '.xml')

    return None

def countBoulderXML(xml):

    tree = ET.parse(xml)

    root = tree.getroot()

    boulders = root.findall('.//object')

    count_of_boulders = len(boulders)

    return count_of_boulders

def findAllXML(xmlPath):

    onlyFiles = [f for f in listdir(xmlPath) if isfile(join(xmlPath,f))]

    data_dict = {}
    for _, data in enumerate(onlyFiles):

        noExt = data.split('.')[0]
        extract_uniqueID = noExt.split("__")[1]

        boulder_count = countBoulderXML(xmlPath + data)

        data_dict[extract_uniqueID] = boulder_count

    return data_dict  

def findSmallBBoxXMLValues(xmlToEdit, pixelTol = 5):
    
    xMinCoord = int(xmlToEdit.find('.//' + 'xmin').text)
    xMaxCoord = int(xmlToEdit.find('.//' + 'xmax').text)
    yMinCoord = int(xmlToEdit.find('.//' + 'ymin').text)
    yMaxCoord = int(xmlToEdit.find('.//' + 'ymax').text)
    
    xDist = xMaxCoord - xMinCoord
    yDist = yMaxCoord - yMinCoord

    if (xDist < pixelTol) or (yDist < pixelTol):
        return False

    return True

def extractNonBBoxXML(xmlRoot):

    stringRoot = str(ET.tostring(xmlRoot))

    grabBeforeObject = stringRoot.split('object')[0]

    cleanBeforeObject = grabBeforeObject[:-1]
    # cleanBeforeObject = grabBeforeObject.replace('\\n\\n\\n<','')
    cleanBeforeObject = cleanBeforeObject.replace("b'","")
    # cleanBeforeObject = cleanBeforeObject.replace("\\n    <","")

    return cleanBeforeObject

def cleanBBoxes(xmlFile):

    tree = ET.parse(xmlFile)

    root = tree.getroot()

    boulders = root.findall('.//object')

    to_keep = []
    for oneBoulder in boulders:

        if findSmallBBoxXMLValues(oneBoulder):
            string_version = str(ET.tostring(oneBoulder))
            cleaned_string = string_version.replace("b'",'')
            to_keep.append(cleaned_string)

    long_et_string = ''.join(to_keep)

    xmlFileNoBoulders = extractNonBBoxXML(root)

    combine_et_strings = xmlFileNoBoulders + long_et_string + '</annotation>'

    cleaned_combined = combine_et_strings.replace("'",'')
    cleaned_combined = cleaned_combined.replace("\\n",'\n')

    new_file = ET.ElementTree(ET.fromstring(cleaned_combined))

    return new_file

def cleanXMLBboxes(path, output_path):

    onlyFiles = [f for f in listdir(path) if isfile(join(path,f))]

    for _, data in enumerate(onlyFiles):

        new_xml_file = cleanBBoxes(path + data)
        new_xml_file.write(output_path + data + '.xml')

    return None