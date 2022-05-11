from torch_snippets import *
from torch.utils.data import Dataset
import pandas as pd
import cv2 as cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET

data_dir = 'D:/boulder-finder/training data_hand_picked_mapbox/sliced/'

label_dir = data_dir + 'cleaned_labels/'

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

# extractXMLToTxt(xmlPath = label_dir,
#                 newPath = label_dir)



bboxData = pd.read_csv(data_dir + '/cleaned_labels/allData_sliced.csv', sep = ",")


class OpenImages(Dataset):
    def __init__(self, df, image_folder):
        self.root = image_folder
        self.df = df
        self.unique_images = df['UniqueImageID'].unique()

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, ix):
        image_id = self.unique_images[ix]
        image_path = f'{self.root}/{image_id}.PNG'
        image = cv2.imread(image_path, 1)[...,::-1] # conver BGR to RGB
        h, w, _ = image.shape
        df = self.df.copy()
        df = df[df['UniqueImageID'] == image_id]
        boxes = df['XMin,YMin,XMax,YMax'.split(',')].values
        boxes = boxes.astype(np.uint16).tolist()
        classes = df['Class'].values.tolist()
        return image, boxes, classes, image_path

debug_manual = ['train_westboulder__107460134011822', 'train_aquamanboulder__108905811000431',
 'train_boardwalkbouldernwface__108533402001228', 'train_iceberg2__106502617005267', 
 'train_lacunarock__109285759005748', 'train_thecavalryboulder__113198558010363',
  'train_monoblockrock__116590285006571', 'train_snackboulder__106382719009471',
   'train_cookieboulder__120311884002467', 'train_chocolateslab__106977704002216', 
   'train_pitbbqboulder__113273835007660', 'train_semisaucestone__110514571009038',
    'train_deinonychusboulder__111746826003050', 'train_millerblock__107845758006430',
     'train_topperrock__109411321010962', 'train_keymasterboulder__107827210005654']

debug_work = ['train_abaloneboulder__110514443000042', 'train_krampusrock__111336653005728',
 'train_nevertrustamanbunboulder__116989342006879', 'train_nosebleedboulder__109507979006947',
  'train_alphaboulder__108023229000241', 'train_thrillerboulder__107764914010739', 
  'train_zaftigboulder__109715771012083', 'train_redacted__106505755008328', 
  'train_diagonboulder__109135403003156', 'train_ifritboulder__111954402005306', 
  'train_thefablebloc__112243711010463', 'train_jumpchumpclump__106073944005567', 
  'train_obeboulder__107697617007006', 'train_highbrowarea__106273779004967',
   'train_icarusstone__113707678005254', 'train_xeriscapeboulder__111009270012013']

debug_not_working_2 = ['train_falseup20boulder__106040085003839', 
'train_skyestone__106320055009327', 'train_lavaboulder__110174546005823',
 'train_poeblock__107470885007736', 'train_quagmireblock__110760169008145',
  'train_slabberwockyboulder__107255858009350', 'train_harmonicboulder__109419067004802', 
  'train_slabaliciousboulder__108435612009340', 'train_tremorstone__109832199011096',
   'train_jasonvsfreddyblock__108056647005478', 'train_cryingdolphinboulder__107948237002759', 
   'train_sacrificialboulder__106079044008778', 'train_poledancerstone__116815165007761',
    'train_acousticstone__118774127000156', 'train_hairtriggerwall__106279682004739', 
    'train_outerbanksboulder__109724452007221']

bbox_debug = bboxData[bboxData['UniqueImageID'].isin(debug_manual)]

ds = OpenImages(df=bbox_debug, image_folder=data_dir + 'images/')

for index in range(16):
    im, bbs, labels, fpath = ds[index]
    show(im, bbs=bbs, texts=labels, sz=10)
