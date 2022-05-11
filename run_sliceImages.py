from generatingData.sliceImages import bboxes, extractXMLToTxt, slice_images

from generatingData.generateBoundingBox import cleanXMLBboxes
    
# data_dir = 'D:/boulder-finder/training data_hand_picked_mapbox/'

# data_dir = 'D:/boulder-finder/training data_hand label/'

data_dir = 'D:/boulder-finder/training data_hand_picked_google/'

dataPath = data_dir + 'images'

tile_size = (300,300)
tile_overlap = 0
ANN_SRC = data_dir + 'labels' #'cleaned_labels'

emptyImageList = bboxes(tile_size=tile_size, tile_overlap=0,
                           ANN_SRC=ANN_SRC, ANN_DST= data_dir + 'sliced/labels')

cleanXMLBboxes(path = data_dir + 'sliced/labels/',
               output_path=data_dir + 'sliced/cleaned_labels/')


IMG_DST = data_dir + 'sliced/images'
emptyIMG_DST = data_dir + 'emptyImages'

slice_images(tile_size=(300,300), dataPath=dataPath,
             IMG_DST=IMG_DST, emptyIMG_DST=emptyIMG_DST,
             imagesToNotMake=emptyImageList)
#--------------------------------------------------------------

label_dir = data_dir + '/sliced/cleaned_labels/'

extractXMLToTxt(xmlPath = label_dir,
                newPath = label_dir)