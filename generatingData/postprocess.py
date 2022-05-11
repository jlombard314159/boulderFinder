import os
import glob
from PIL import Image
import io
import numpy as np

def all_white_image(image_array):

    oneSlice = image_array[0,:,0:3]

    uniqueVals = np.unique(oneSlice).tolist()

    if (len(uniqueVals) == 1) & (uniqueVals[0] == 221):
        return True

    return False

def find_empty_images(dataPath):

    empty_image_list = []
    for file in sorted(glob.glob(dataPath + os.sep + "*")):
        
        img = Image.open(file)
        arrayImage = np.asarray(img)

        if all_white_image(arrayImage):
            empty_image_list.append(file)

    return empty_image_list

def remove_file(fileList):

    for _, data in enumerate(fileList):

        os.remove(data)

    return None

def find_missing_images(dataPath, uniqueIDs):
    
    still_to_make = []

    for file in sorted(glob.glob(dataPath + os.sep + "*")):
        
        noExt = file.split('.')[0]
        extract_uniqueID = noExt.split("__")[1]

        if extract_uniqueID in uniqueIDs:

            still_to_make.append(extract_uniqueID)

    return still_to_make
