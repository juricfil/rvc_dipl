import torch
import argparse
from torch import nn
import numpy as np
from model.model import custom_DeepLabv3
from model.metrics import mIoU, pixel_accuracy
from dataloader.dataloader_test import train_dataset, val_dataset
import matplotlib.pyplot as plt

def get_argparser():
    parser = argparse.ArgumentParser()
    #Dataset Options
    parser.add_argument("--dataset",type=str,default='cityscapes',
    choices=['cityscapes','kitti','mapillary','viper','wilddash'],help="Name of dataset")
    parser.add_argument("--epochs", type=int, default = 100,
                        help = "number of epochs to train for(default = 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="set the learning rate(default = 1e-3")
    #Ako cu ga koristiti
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--batch_size",type=int,default=16,help="set the batch size(default=16)")
    #Dodati jos loss funkcija ako ce se eksperimentirati 
    parser.add_argument("--loss_function",type=str, default="CrossEntropyLoss",choices=["CrossEntropyLoss"],help="define loss type")

    return parser

opts = get_argparser().parse_args()
learning_rate = opts.learning_rate
batch_size = opts.batch_size
epochs = opts.epochs

model=custom_DeepLabv3(256)
model=model.to('cuda')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

train_losses = []
val_losses = []
val_iou = []; val_acc = []
train_iou = []; train_acc = []
min_loss = np.inf
decrease = 1 ; not_improve=0

for epoch in range(epochs):
    running_loss = 0
    iou_score=0
    accuracy = 0
    print(f"Epoch {epoch+1}\n-------------------------------")
    #training
    for i, data in enumerate(train_dataloader):
        model.train()
        # Compute prediction and loss
        x, y = data
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        pred = model(x)
        #print(pred['out'].shape)     
        loss = loss_fn(pred['out'],y)
        #eval metrics
        iou_score += mIoU(pred['out'], y)
        accuracy +=pixel_accuracy(pred['out'], y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #loss
        running_loss += loss.item()
        train_losses.append(running_loss/len(train_dataloader))

    #validation
    val_loss = 0
    val_accuracy = 0
    val_iou_score = 0
    for i, data in enumerate(val_dataloader):
        model.eval()
        with torch.no_grad():
            x, y = data
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            #print(x.shape)
            pred = model(x)
            #print(pred['out'].shape)
            #eval metrics
            val_iou_score += mIoU(pred['out'], y)
            val_accuracy += pixel_accuracy(pred['out'], y)
            loss = loss_fn(pred['out'], y)
            val_loss += loss.item()
            val_losses.append(val_loss/len(val_dataloader))

    #metrics report
    val_iou.append(val_iou_score/len(val_dataloader))
    train_iou.append(iou_score/len(train_dataloader))
    train_acc.append(accuracy/len(train_dataloader))
    val_acc.append(val_accuracy/ len(val_dataloader))
    print("Epoch:{}/{}..".format(epoch+1, epochs),
            "Train Loss: {:.3f}..".format(running_loss/len(train_dataloader)),
            "Val Loss: {:.3f}..".format(val_loss/len(val_dataloader)),
            "Train mIoU:{:.3f}..".format(iou_score/len(train_dataloader)),
            "Val mIoU: {:.3f}..".format(val_iou_score/len(val_dataloader)),
            "Train Acc:{:.3f}..".format(accuracy/len(train_dataloader)),
            "Val Acc:{:.3f}..".format(val_accuracy/len(val_dataloader)))
    
    #saving model with min loss
    if min_loss > (val_loss/len(val_dataloader)):
        print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (val_loss/len(val_dataloader))))
        min_loss = (val_loss/len(val_dataloader))
        decrease += 1
        if decrease % 5 == 0:
            print('saving model...')
            torch.save(model, 'model_folder/DeepLabV3_mIoU-{:.3f}.pt'.format(val_iou_score/len(val_dataloader)))

#plot history
history = {'train_loss' : train_losses, 'val_loss': val_losses,
               'train_miou' :train_iou, 'val_miou':val_iou,
               'train_acc' :train_acc, 'val_acc':val_acc}

def plot_loss(history):
    plt.plot(history['val_loss'], label='val', marker='*')
    plt.plot( history['train_loss'], label='train', marker='*')
    plt.title('Loss per epoch'); plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('plt_loss_score_SGD-norm.png')
    plt.show()
    
def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU',  marker='*')
    plt.title('Score per epoch'); plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('plt_miou_score_SGD-norm.png')
    plt.show()
    
def plot_acc(history):
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy',  marker='*')
    plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('plt_acc_score_SGD-norm.png')
    plt.show()

plot_loss(history)
plot_score(history)
plot_acc(history)