from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torch_snippets.loader import *

# Load model checkpoint
# checkpoint = 'checkpoint_ssd300.pth.tar'
# checkpoint = torch.load(checkpoint)
# start_epoch = checkpoint['epoch'] + 1
# print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
# model = checkpoint['model']

resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect(original_image, model, min_score, max_overlap, top_k, device, suppress=None):
 
    image_reshape = resize(original_image)
    image_tensor = to_tensor(image_reshape)
    image = normalize(image_tensor).to(device)
    
    regr_, clss_ = model(image[None])
    boxes, labels, confs = model.detect_objects(regr_, clss_,
                                                min_score=min_score,
                                                max_overlap=max_overlap, top_k=top_k)
    boxes = boxes[0].to('cpu')
    confs = [s.item() for s in confs[0]]
    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height])[None]
    bbs = boxes * original_dims
    bbs = bbs.cpu().detach().numpy().astype(np.int16).tolist()
    # labels = [rev_label_map[l] for l in labels[0].to('cpu').tolist()]
    labels = labels[0]
    return bbs, labels, confs

# if __name__ == '__main__':
#     img_path = '/media/ssd/ssd data/VOC2007/JPEGImages/000001.jpg'
#     original_image = Image.open(img_path, mode='r')
#     original_image = original_image.convert('RGB')
#     detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()