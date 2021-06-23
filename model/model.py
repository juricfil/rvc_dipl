import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def custom_DeepLabv3(out_channel):
  model = deeplabv3_resnet101(pretrained=False)
  model.classifier = DeepLabHead(2048, out_channel)

  #model u training mode
  model.train()
  return model
