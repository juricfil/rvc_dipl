import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import glob
import random
import numpy as np
import matplotlib.pyplot as plt 

train_image_path = glob.glob("/home/filip/diplomski-code/data/rvc_uint8/images/train/cityscapes-34/*.png")
train_mask_path = glob.glob("/home/filip/diplomski-code/data/rvc_uint8/annotations/train/cityscapes-34/*.png")

val_image_path = glob.glob("/home/filip/diplomski-code/data/rvc_uint8/images/val/cityscapes-34/*.png")
val_mask_path = glob.glob("/home/filip/diplomski-code/data/rvc_uint8/annotations/val/cityscapes-34/*.png")

class MyDataset(Dataset):
    def __init__(self, image_path, mask_path,train=True):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transformT = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.2893, 0.3238, 0.2822], std=[0.1046, 0.1057, 0.1025])

    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask==k] = self.mapping[k]
        return mask

    def transform(self, image, mask):
        resize = transforms.Resize(size=(100,100), interpolation=Image.NEAREST) #Resize
        image = resize(image)
        mask = resize(mask)
        #Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(60,60))
        image = TF.crop(image,i,j,h,w)
        mask = TF.crop(mask, i ,j, h, w)
        #Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        mask = Image.open(self.mask_path[index])
        image, mask = self.transform(image, mask)
        image = self.transformT(image)
        image = self.normalize(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        #mask = self.mask_to_class(mask) 
        mask = mask.long()
        return image, mask

    def __len__(self):
        return len(self.image_path)


train_dataset = MyDataset(train_image_path, train_mask_path, train=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = MyDataset(val_image_path, val_mask_path, train=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

batch_x, batch_y = next(iter(val_dataloader))
print('x shape    ', batch_x.shape)
