import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

model_path = "/home/filip/diplomski-code/model_folder/DeepLabV3_mIoU-0.219.pt"
image_path = "/home/filip/diplomski-code/data/rvc_uint8/images/train/cityscapes-34/aachen_000000_000019_leftImg8bit.png"

# Load the trained model
if torch.cuda.is_available():
    model = torch.load(model_path)
else:
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Set the model to evaluate mode
model.eval()

#Changes to input for inference
transformT = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.2893, 0.3238, 0.2822], std=[0.1046, 0.1057, 0.1025])

image = Image.open(image_path)
resize = transforms.Resize(size=(100,100), interpolation=Image.NEAREST) #Resize
image = resize(image)
image = transformT(image)
image = normalize(image)

#creates mini batch of 1, model expects that kind of input
image = image.unsqueeze(0)

print(image.shape)

if torch.cuda.is_available():
    image = image.to('cuda')

with torch.no_grad():
    output = model(image)['out'][0]
output_predictions = output.argmax(0)

"""
# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(256)])[ns.byte().cpu().numpy()).resize(image.size)
r.putpalette(colors):, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictio
"""

plt.imshow(output_predictions.byte().cpu().numpy())
plt.show()