from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torch_snippets.loader import show
import io
from generatingData.generatingHTML import generateMap, basemaps
from generatingData.sliceImages import apply_tile_operation
from modeling.ssd.model_utils import SSD300
import torch
##Fetch image

DATA_ROOT = 'D:/boulder-finder/training data_hand_picked_mapbox/sliced/'

def create_map(createdMap):

    img_data = createdMap._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    rgb_conv = img.convert('RGB')

    return rgb_conv

##Predict on it

def extract_boulders(boxes, scores, score_thres = 0.50):

    unlisted_boxes = [item for sublist in boxes for item in sublist]

    boulder_list = []
    for boulders, boulder_score in zip(unlisted_boxes, scores):

        indices = [i for i, x in enumerate(boulder_score) if x > score_thres]

        good_scoring_boulders = [boulders[i] for i in indices]

        boulder_list.append(good_scoring_boulders)

    return boulder_list

def unlist_data(listed_data):
    unlist = [item for sublist in listed_data for item in sublist]
    return unlist

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SSD300(2,device)
checkpoint = torch.load(DATA_ROOT + 'SSD300_mb_50ep.pt')
model.load_state_dict(checkpoint['model_state_dict'])

red_feather = [40.86473, -105.52428]

my_map = generateMap(gpsCoord = red_feather)
img_conversion = create_map(my_map)

bbs, labels, scores = apply_tile_operation(img_conversion, tile_size= (300,300),
                                           model = model, device = device)
## unSlice

unlisted_boxes = unlist_data(bbs)
unlisted_labels = unlist_data(labels)
unlisted_scores = unlist_data(scores)

label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(unlisted_labels,unlisted_scores)]

show(img_conversion, bbs=unlisted_boxes, texts= label_with_conf)

##get new XML coords somehow (This is  code I definitely don't have)