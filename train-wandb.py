import torch
import argparse
from torch import nn
import numpy as np
from model.model import custom_DeepLabv3
from model.metrics import mIoU, pixel_accuracy
from dataloader.dataloader_test import train_dataset, val_dataset
import matplotlib.pyplot as plt
import wandb

#login weights&biases
wandb.login()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = dict(
    epochs=100,
    batch_size=16,
    learning_rate=1e-2,
    optimizer = "Adam",
    loss = "CrossEntropyLoss",
    dataset="cityscapes",
    architecture="DeepLabV3")

def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="rvc_diplomski", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_dataloader, val_dataloader, loss_fn, optimizer = make(config)
      print(model)

      # and use them to train the model
      train(model, train_dataloader,val_dataloader, loss_fn, optimizer, config)

    return model

def make(config):
    # Make the data
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Make the model
    model = custom_DeepLabv3(256).to(device)

    # Make the loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_dataloader, val_dataloader, loss_fn, optimizer

def train_log(loss, mIoU, Acc, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def val_log(loss, mIou, Acc, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, config):
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_fn, log="all", log_freq=10)
    train_losses = []
    val_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []
    min_loss = np.inf
    decrease = 1 ; not_improve=0

    for epoch in range(config.epochs):
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
        print("Epoch:{}/{}..".format(epoch+1, config.epochs),
                "Train Loss: {:.3f}..".format(running_loss/len(train_dataloader)),
                "Val Loss: {:.3f}..".format(val_loss/len(val_dataloader)),
                "Train mIoU:{:.3f}..".format(iou_score/len(train_dataloader)),
                "Val mIoU: {:.3f}..".format(val_iou_score/len(val_dataloader)),
                "Train Acc:{:.3f}..".format(accuracy/len(train_dataloader)),
                "Val Acc:{:.3f}..".format(val_accuracy/len(val_dataloader)))
        # where the magic happens
        wandb.log({"epoch": epoch, "Train loss": running_loss/len(train_dataloader), "Val Loss":val_loss/len(val_dataloader),"Train mIou":iou_score/len(train_dataloader),"Val mIou":val_iou_score/len(val_dataloader),"Train Acc":accuracy/len(train_dataloader),"Val Acc":val_accuracy/len(val_dataloader)}, step=epoch)
        print(f"Epoch" + str(epoch+1) + f" examples: {loss:.3f}")
        
        #saving model with min loss
        if min_loss > (val_loss/len(val_dataloader)):
            print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (val_loss/len(val_dataloader))))
            min_loss = (val_loss/len(val_dataloader))
            decrease += 1
            if decrease % 5 == 0:
                print('saving model...')
                torch.onnx.export(model, x, "DeepLabV3_mIoU-{:.3f}.onnx".format(val_iou_score/len(val_dataloader)))
                wandb.save("DeepLabV3_mIoU-{:.3f}.onnx".format(val_iou_score/len(val_dataloader)))
        #Save finalnog modela
        if epoch == (config.epochs-1):
            torch.onnx.export(model, x, "DeepLabV3_mIoU-{:.3f}.onnx".format(val_iou_score/len(val_dataloader)))
            wandb.save("DeepLabV3_mIoU-{:.3f}.onnx".format(val_iou_score/len(val_dataloader)))



model = model_pipeline(config)

