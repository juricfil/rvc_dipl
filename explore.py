#pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from torch.utils import data
import random
from PIL import Image, ImageOps
from model.model import custom_DeepLabv3
from torchvision import transforms

# Create the model
model = custom_DeepLabv3(256)

# Load state_dict
model.load_state_dict(torch.load('DeepLabV3_mIoU-0170.pt'))

# Create the preprocessing transformation here
transform = transforms.ToTensor()

# load your image(s)
img = Image.open('/home/filip/diplomski-code/data/rvc_uint8/images/val/cityscapes-34/frankfurt_000000_000294_leftImg8bit.png')

# Transform
input = transform(img)

# unsqueeze batch dimension, in case you are dealing with a single image
input = input.unsquueeze(0)

# Set model to eval
model.eval()

# Get prediction
output = model(input)