# import cv2

# def increase_brightness(img, value=30):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)

#     lim = 255 - value
#     v[v > lim] = 255
#     v[v <= lim] += value

#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img

# baseDir = 'D:/boulder-finder/training data_hand label/images/'

# img = cv2.imread(baseDir + 'train_gandalf__temp.png')
# cv2.imshow('image window', img)

# new_img = increase_brightness(img,90)
# cv2.imshow('image window', new_img)

from torchvision import transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=[0,0.25], hue = [0,0.5], saturation=[0,0.25], contrast=[0, 0.25])
    ])
}

from torch_snippets import *
import pandas as pd
import torch 
import collections, os, torch
from PIL import Image
from torchvision import transforms
import glob
from sklearn.model_selection import train_test_split
from modeling.ssd.predict import detect
from modeling.ssd.model_utils import SSD300, MultiBoxLoss
from modeling.ssd.utils import modify_pixels

DATA_ROOT = 'D:/boulder-finder/training data_hand label/sliced/'

IMAGE_ROOT = f'{DATA_ROOT}images'
df = pd.read_csv(f'{DATA_ROOT}cleaned_labels/allData_sliced.csv')

df = modify_pixels(df)

DF_RAW = df

df = df[df['UniqueImageID'].isin(df['UniqueImageID'].unique().tolist())]

label2target = {l:t+1 for t,l in enumerate(DF_RAW['Class'].unique())}
label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)



def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.to(device).float()
    
class OpenDataset(torch.utils.data.Dataset):
    w, h = 300, 300
    def __init__(self, df, image_dir=IMAGE_ROOT, transform=None):
        self.image_dir = image_dir
        self.files = glob.glob(self.image_dir+'/*')
        self.df = df
        self.image_infos = df.UniqueImageID.unique()
        logger.info(f'{len(self)} items loaded')

        self.transform = transform
        
    def __getitem__(self, ix):
        # load images and masks
        image_id = self.image_infos[ix]
        img_path = find(image_id, self.files)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255.

        data = df[df['UniqueImageID'] == image_id]
        labels = data['Class'].values.tolist()

        data = data[['XMin','YMin','XMax','YMax']].values
        
        data[:,[0,2]] *= self.w
        data[:,[1,3]] *= self.h
        
        boxes = data.astype(np.uint32).tolist() # convert to absolute coordinates
       
        return img, boxes, labels

    def collate_fn(self, batch):
        images, boxes, labels = [], [], []
        for item in batch:
            img, image_boxes, image_labels = item
            img = preprocess_image(img)[None]
            images.append(img)
            boxes.append(torch.tensor(image_boxes).float().to(device)/300.)
            labels.append(torch.tensor([label2target[c] for c in image_labels]).long().to(device))
        images = torch.cat(images).to(device)
        return images, boxes, labels

    def __len__(self):
        return len(self.image_infos)


train_ds = OpenDataset(df)
train_ds_aug = OpenDataset(df,transform = data_transforms['train'])
train_ds_total = torch.utils.data.ConcatDataset([train_ds, train_ds_aug])


t1, t2, t3 = train_ds[0]


train_loader = DataLoader(train_ds_total, batch_size=16, collate_fn=train_ds.collate_fn,
                          drop_last=True, shuffle=True)


def train_batch(inputs, model, criterion, optimizer):
    model.train()
    N = len(train_loader)
    images, boxes, labels = inputs
    _regr, _clss = model(images)
    loss = criterion(_regr, _clss, boxes, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

n_epochs = 100
model = SSD300(num_classes,device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)

modelPath = 'D:/boulder-finder/training data_hand_picked_mapbox/sliced/SSD300_mb_60ep.pt'

checkpoint = torch.load(modelPath)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


#-----------------

log = Report(n_epochs=n_epochs)
logs_to_print = 5
for epoch in range(n_epochs):
    _n = len(train_loader)
    for ix, inputs in enumerate(train_loader):
        loss = train_batch(inputs, model, criterion, optimizer)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, trn_loss=loss.item(), end='\r')



torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, DATA_ROOT + 'SSD300_mb_60ep_updated_hand_label.pt')