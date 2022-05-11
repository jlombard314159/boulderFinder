from numpy.lib.function_base import average
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
from modeling.ssd.utils import extract_list_of_bbs, modify_pixels
import matplotlib.pyplot as plt
import pandas as pd

DATA_ROOT = 'D:/boulder-finder/training data_hand_picked_google/sliced/'
# DATA_ROOT = 'D:/boulder-finder/training data_hand_picked_mapbox/sliced/'

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

data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=[0,0.25], hue = [0,0.5], saturation=[0,0.25], contrast=[0, 0.25])
    ])
}


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
       
        return img, boxes, labels, image_id

    def collate_fn(self, batch):
        images, boxes, labels, image_names = [], [], [], []
        for item in batch:
            img, image_boxes, image_labels, image_id = item
            img = preprocess_image(img)[None]
            images.append(img)
            boxes.append(torch.tensor(image_boxes).float().to(device)/300.)
            labels.append(torch.tensor([label2target[c] for c in image_labels]).long().to(device))
            image_names.append(image_id)
        images = torch.cat(images).to(device)
        return images, boxes, labels, image_names

    def __len__(self):
        return len(self.image_infos)


trn_ids, val_ids = train_test_split(df.UniqueImageID.unique(), test_size=0.10, random_state=99)
trn_df, val_df = df[df['UniqueImageID'].isin(trn_ids)], df[df['UniqueImageID'].isin(val_ids)]
len(trn_df), len(val_df)

train_ds = OpenDataset(trn_df)

train_ds_aug = OpenDataset(trn_df,transform = data_transforms['train'])

train_ds_total = torch.utils.data.ConcatDataset([train_ds, train_ds_aug])

t1, t2, t3, t4 = train_ds_aug[0]

test_ds = OpenDataset(val_df)

#------------------------------------------------------------------------------------------------
train_loader = DataLoader(train_ds_total, batch_size=32, collate_fn=train_ds.collate_fn, drop_last=True, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, collate_fn=test_ds.collate_fn, drop_last=True)

def train_batch(inputs, model, criterion, optimizer):
    model.train()
    N = len(train_loader)
    images, boxes, labels, _ = inputs
    _regr, _clss = model(images)
    loss = criterion(_regr, _clss, boxes, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
    
@torch.no_grad()
def validate_batch(inputs, model, criterion):
    model.eval()
    images, boxes, labels, _ = inputs
    _regr, _clss = model(images)
    loss = criterion(_regr, _clss, boxes, labels)
    return loss

##Based on trail and error 40 is where val_loss is minimized and train loss is relatively small
n_epochs = 100
model = SSD300(num_classes,device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.00001, max_lr = 0.1, cycle_momentum=False)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)

log = Report(n_epochs=n_epochs)

for epoch in range(71, n_epochs+1):
    _n = len(train_loader)
    for ix, inputs in enumerate(train_loader):
        loss = train_batch(inputs, model, criterion, optimizer)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, trn_loss=loss.item(), end='\r')

    _n = len(test_loader)
    for ix,inputs in enumerate(test_loader):
        val_loss = validate_batch(inputs, model, criterion)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, val_loss=val_loss.item(), end='\r')

    ##Use with preloaded model
    if(epoch % 10) == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'log':log
            }, DATA_ROOT + 'SSD300_mb_' + str(epoch) +'ep.pt')

#----------------------------------------------------------------------

torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'log':log,
            }, DATA_ROOT + 'SSD300_goog_40ep.pt')

log.plot()


#Find min avg val and non val loss
from math import ceil

def average_loss(loss_vals):

    extract_loss = [x['loss'] for x in loss_vals]
    epochs = [x['epoch'] for x in loss_vals]

    output_df = pd.DataFrame(data={'loss':extract_loss, 'epoch':epochs})

    avg_losses = output_df.groupby('epoch').mean()
    avg_losses['epoch'] = avg_losses.index

    return avg_losses

def convert_epoch(logs):

    new_log = []
    for data in logs:
        epoch = ceil(data[0])
        temp_dict = {'epoch':epoch, 'loss':data[1]}
        new_log.append(temp_dict)


    return new_log

validation_loss = convert_epoch(log.val_loss)
train_loss = convert_epoch(log.trn_loss)

avg_val = average_loss(validation_loss)
avg_train = average_loss(train_loss)

plt.plot(avg_train['epoch'],avg_train['loss'])
plt.show()

plt.plot(avg_val['epoch'],avg_val['loss'])
plt.show()

#--------------------------------------------------------------------------------
checkpoint = torch.load(DATA_ROOT + 'SSD300_goog_70ep.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
log = checkpoint['log']

boulder_found = []
raw_df = pd.read_csv(f'{DATA_ROOT}cleaned_labels/allData_sliced.csv')
# for _ in range(len(test_ds)):
for _ in range(10):
    image_id = choose(train_ds.image_infos)
    img_path = find(image_id, train_ds.files)
    original_image = Image.open(img_path, mode='r').convert("RGB")
    bbs, labels, scores = detect(original_image, model, min_score=0.75, max_overlap=0.50,top_k=200, device=device)
    labels = [target2label[c.item()] for c in labels]
    label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(labels,scores)]

    if 'Boulder' in labels:
        boulder_found.append({'img_path':img_path,'img_id':image_id,
            'conf':label_with_conf})

    # print(bbs, label_with_conf)
    pred = show(original_image, bbs=bbs, texts=label_with_conf, text_sz=10)

    actual_bbs_df = raw_df[raw_df['UniqueImageID'] == image_id]
    actual_bbs = extract_list_of_bbs(actual_bbs_df)
    actual = show(original_image, bbs=actual_bbs)
