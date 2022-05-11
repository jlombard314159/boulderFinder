import os
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# import image_bbox_slicer as ibs
import glob
from pascal_voc_writer import Writer
from enum import Enum
from itertools import compress
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from math import sqrt, ceil, floor
from modeling.ssd.predict import detect

#----------------------------------------------------
def calc_columns_rows(n):
    num_columns = int(ceil(sqrt(n)))
    num_rows = int(ceil(n / float(num_columns)))
    return (num_columns, num_rows)

def bboxes(tile_size, tile_overlap, ANN_SRC,ANN_DST):

    img_no = 1
    emptyImageList = []

    for xml_file in sorted(glob.glob(ANN_SRC + os.sep + '*.xml')):
        root, objects = extract_from_xml(xml_file)
        im_w, im_h = int(root.find('size')[0].text), int(
            root.find('size')[1].text)
        im_filename = root.find('filename').text.split('.')[0]
        extn = root.find('filename').text.split('.')[1]

        tile_w, tile_h = tile_size
        tiles = get_tiles((im_w, im_h), tile_size, tile_overlap)

        for tile in tiles:
            img_no_str = '{:06d}'.format(img_no)
            voc_writer = Writer('{}{}.{}'.format(img_no_str,im_filename, extn), tile_w, tile_h)
            empty_count = 0
            for obj in objects:
                obj_lbl = obj[-4:]
                points_info = which_points_lie(obj_lbl, tile)

                if points_info == Points.NONE:
                    empty_count += 1
                    continue

                elif points_info == Points.ALL:       # All points lie inside the tile
                    new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                               obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                elif points_info == Points.P1:
                    new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                               tile_w, tile_h)

                elif points_info == Points.P2:
                    new_lbl = (0, obj_lbl[1] - tile[1],
                               obj_lbl[2] - tile[0], tile_h)

                elif points_info == Points.P3:
                    new_lbl = (obj_lbl[0] - tile[0], 0,
                               tile_w, obj_lbl[3] - tile[1])

                elif points_info == Points.P4:
                    new_lbl = (0, 0, obj_lbl[2] - tile[0],
                               obj_lbl[3] - tile[1])

                elif points_info == Points.P1_P2:
                    new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                               obj_lbl[2] - tile[0], tile_h)

                elif points_info == Points.P1_P3:
                    new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                               tile_w, obj_lbl[3] - tile[1])

                elif points_info == Points.P2_P4:
                    new_lbl = (0, obj_lbl[1] - tile[1],
                               obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                elif points_info == Points.P3_P4:
                    new_lbl = (obj_lbl[0] - tile[0], 0,
                               obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                voc_writer.addObject(obj[0], new_lbl[0], new_lbl[1], new_lbl[2], new_lbl[3],
                                     obj[1], obj[2], obj[3])

            img_no += 1
            if empty_count == len(objects):
                emptyImageList.append(img_no_str)
                continue
            else:
                voc_writer.save(
                    '{}{}{}{}.xml'.format(ANN_DST, os.sep,im_filename,img_no_str))


    print('Obtained {} annotation slices!'.format(img_no-1))
    return emptyImageList

def extract_from_xml(file):
    """Extracts useful info (classes, bounding boxes, filename etc.) from annotation (XML) file.
    Parameters
    ----------
    file : str
        /path/to/xml/file.
    Returns
    ----------
    Element, list
        Element is the root of the XML tree.
        list contains info of annotations (objects) in the file.
    """
    objects = []
    tree = ET.parse(file)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        pose = 'Unknown'
        truncated = '0'
        difficult = '0'
        if obj.find('pose') is not None:
            pose = obj.find('pose').text

        if obj.find('truncated') is not None:
            truncated = obj.find('truncated').text

        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text

        bbox = obj.find('bndbox')
        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        for point in bbox:
            if point.tag == 'xmin':
                xmin = point.text
            elif point.tag == 'ymin':
                ymin = point.text
            elif point.tag == 'xmax':
                xmax = point.text
            elif point.tag == 'ymax':
                ymax = point.text
        value = (name,
                 pose,
                 int(truncated),
                 int(difficult),
                 int(xmin.split('.')[0]),
                 int(ymin.split('.')[0]),
                 int(xmax.split('.')[0]),
                 int(ymax.split('.')[0])
                 )
        objects.append(value)
    return root, objects

def which_points_lie(label, tile):

    # 0,1 -- 2,1
    # |        |
    # 0,3 -- 2,3
    points = [False, False, False, False]

    if (tile[0] <= label[0] and tile[2] >= label[0]):
        if (tile[1] <= label[1] and tile[3] >= label[1]):
            points[0] = True
        if (tile[1] <= label[3] and tile[3] >= label[3]):
            points[2] = True

    if (tile[0] <= label[2] and tile[2] >= label[2]):
        if (tile[1] <= label[1] and tile[3] >= label[1]):
            points[1] = True
        if (tile[1] <= label[3] and tile[3] >= label[3]):
            points[3] = True

    if sum(points) == 0:
        return Points.NONE
    elif sum(points) == 4:
        return Points.ALL

    elif points[0]:
        if points[1]:
            return Points.P1_P2
        elif points[2]:
            return Points.P1_P3
        else:
            return Points.P1

    elif points[1]:
        if points[3]:
            return Points.P2_P4
        else:
            return Points.P2

    elif points[2]:
        if points[3]:
            return Points.P3_P4
        else:
            return Points.P3

    else:
        return Points.P4

class Points(Enum):
    """An Enum to hold info of points of a bounding box or a tile.
    Used by the method `which_points_lie` and a private method in `Slicer` class.
    See `which_points_lie` method for more details.
    Example
    ----------
    A box and its points
    P1- - - - - - -P2
    |               |
    |               |
    |               |
    |               |
    P3- - - - - - -P4
    """

    P1, P2, P3, P4 = 1, 2, 3, 4
    P1_P2 = 5
    P1_P3 = 6
    P2_P4 = 7
    P3_P4 = 8
    ALL, NONE = 9, 10

def get_tiles(img_size, tile_size, tile_overlap = 0):

    tiles = []
    img_w, img_h = img_size
    tile_w, tile_h = tile_size
    stride_w = int((1 - tile_overlap) * tile_w)
    stride_h = int((1 - tile_overlap) * tile_h)
    for y in range(0, img_h-tile_h+1, stride_h):
        for x in range(0, img_w-tile_w+1, stride_w):
            x2 = x + tile_w
            y2 = y + tile_h
            tiles.append((x, y, x2, y2))
    return tiles

def slice_images(tile_size, dataPath,
                 IMG_DST, emptyIMG_DST, imagesToNotMake,
                 tile_overlap = 0):

    img_no = 1

    for file in sorted(glob.glob(dataPath + os.sep + "*")):
        file_name = file.split(os.sep)[-1].split('.')[0]
        file_type = file.split(os.sep)[-1].split('.')[-1].lower()

        im = Image.open(file)

        tiles = get_tiles(im.size, tile_size, tile_overlap)
        new_ids = []
        for tile in tiles:
            new_im = im.crop(tile)
            img_id_str = str('{:06d}'.format(img_no))

            img_no += 1

            if(img_id_str in imagesToNotMake):
                # new_im.save(
                #     '{}{}{}{}.{}'.format(emptyIMG_DST, os.sep, file_name, img_id_str, file_type))
                continue
            else:
                new_im.save(
                    '{}{}{}{}.{}'.format(IMG_DST, os.sep,file_name, img_id_str, file_type))
                new_ids.append(img_id_str)

    print('Obtained {} image slices!'.format(img_no - 1))

    return None

def subXML(treeToSub,findAllQuery):

    formatted = treeToSub.findall(findAllQuery)
    formatted = [int(x.text) for x in formatted]

    return formatted

def extractXMLToTxt(xmlPath,newPath):

    outputDF = pd.DataFrame()

    for xml_file in sorted(glob.glob(xmlPath + os.sep + '*.xml')):
        tree = ET.parse(xml_file)

        uniqueID = xml_file.split(os.sep)[-1].split('.')[0]

        xmin = subXML(tree, findAllQuery='.//xmin')
        xmax = subXML(tree, findAllQuery='.//xmax')
        ymax = subXML(tree, findAllQuery='.//ymax')
        ymin = subXML(tree, findAllQuery='.//ymin')

        dictToDF = {'XMin':xmin, 'XMax':xmax, 'YMin':ymin, 'YMax':ymax}
        indivOutputDF = pd.DataFrame(dictToDF)
        indivOutputDF['Class'] = 'Boulder'
        indivOutputDF['UniqueImageID'] = uniqueID

        outputDF = outputDF.append(indivOutputDF)

    outputDF.to_csv(newPath + 'allData_sliced.csv',
                   header=list(outputDF.columns),
                   index=None, sep=',', mode='a')

    return None

def detect_a_slice(sliced_image, model, device):

    target2label = {1: 'Boulder', 0: 'background'}

    bbs, labels, scores = detect(sliced_image, model, min_score=0.75, max_overlap=0.50,top_k=200, device=device)
    labels = [target2label[c.item()] for c in labels]

    return bbs, labels, scores

def untile_bbs(tile_dim, bound_boxes) -> list: 

    add_x = tile_dim[0]
    add_y = tile_dim[1]

    for index, sublist in enumerate(bound_boxes):

        sublist[0] += add_x
        sublist[2] += add_x

        sublist[1] += add_y
        sublist[3] += add_y

        bound_boxes[index] = sublist

    return bound_boxes

def apply_tile_operation(image, tile_size, model, device):

    tiles = get_tiles(image.size, tile_size)

    bbs_list = []
    labels_list = []
    scores_list = []
    for tile in tiles:
        new_im = image.crop(tile)
       
        bbs, labels, scores = detect_a_slice(new_im, model, device)
        
        if 'Boulder' in labels:
            bbs = untile_bbs(tile, bbs)

            bbs_list.append(bbs)
            labels_list.append(labels)
            scores_list.append(scores)

    return bbs_list, labels_list, scores_list
