import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Residual_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super(Residual_Block, self).__init__()
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, padding=0, stride=stride)
        
        self.in_1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.in_2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        
        res_block = self.relu(self.in_1(x))
        res_block = self.conv1(res_block)
        res_block = self.relu(self.in_2(res_block))
        res_block = self.conv2(res_block)
        s = self.skip_conv(x)
        skip = res_block + s
        return skip

class Decoder_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Decoder_Block, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block = Residual_Block(in_channels + out_channels, out_channels, stride=1)

    def forward(self, x, skip):
        
        dec_block = self.upsample(x)
        dec_block = torch.cat([dec_block, skip], axis=1)
        dec_block = self.res_block(dec_block)
        return dec_block

class double_conv(nn.Module):

    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=1,maxpool=False):
        super(double_conv,self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()   
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class max_down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(max_down,self).__init__()

        self.max_down = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels,out_channels)
        )

    def forward(self, x):
        x = self.max_down(x)
        return x

class upsample(nn.Module):
    
    def __init__(self, in_channels, out_channels, in_conv=None, bilinear=False):
        super(upsample,self).__init__()

        if bilinear:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels)
        
        if in_conv != None:
            self.conv = double_conv(in_conv, out_channels)
            if bilinear:
                self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            else:
                self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        
        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out


def train_model(model, data_loader, val_loader, epochs, steps_per_epoch, device, optim, iou, dice, precision, recall):

    outputs = []
    highest_dice = 0.0
    highest_iou = 0.0
    highest_prec = 0.0
    highest_rec = 0.0
    for epoch in range(epochs):
        print('-'*20)
        for i, (img, annotation) in enumerate(data_loader):
            img = img.to(device)
            annotation = annotation.to(device)
            
            output = model(img)
            iou_loss = iou(output, annotation)
            dice_loss = dice(output, annotation)
            precision_met = precision(output, annotation)
            recall_met = recall(output, annotation)
            
            optim.zero_grad()
            iou_loss.backward()
            optim.step()

            if highest_iou < 1-iou_loss.item():
                highest_iou = 1-iou_loss.item()

            if highest_dice < dice_loss:
                highest_dice = dice_loss

            if highest_prec < precision_met:
                highest_prec = precision_met

            if highest_rec < recall_met:
                highest_rec = recall_met    
            
            if (int(i+1))%(steps_per_epoch//5) == 0:
                print(f"epoch {epoch+1}/{epochs}, step {i+1}/{steps_per_epoch}, IoU score = {1-iou_loss.item():.4f}, Precision = {precision_met:.4f}, Recall = {recall_met:.4f}, F1/Dice score: {dice_loss:.4f}")

        model.eval()
        for img, annotation in val_loader:
            img = img.to(device)
            annotation = annotation.to(device)

            output = model(img)
            iou_loss = iou(output, annotation)
            dice_loss = dice(output, annotation)
            precision_met = precision(output, annotation)
            recall_met = recall(output, annotation)

        print(f"validation loss: IoU score = {1-iou_loss.item():.4f}, Precision = {precision_met:.4f}, Recall = {recall_met:.4f}, F1/Dice score: {dice_loss:.4f}")
        
        outputs.append((img, annotation, output))

    print("-"*20)
    print(f"highest values, IoU score = {highest_iou:.4f}, Precision = {highest_prec:.4f}, Recall = {highest_rec:.4f}, F1/Dice score: {highest_dice:.4f}")

    return model, outputs

def visualize_images(img, annot,n_classes):

    plt.figure(figsize=(50, 10))
    for j in range(10):
        if j >= 10: break
        image = np.squeeze(img[j].cpu().numpy())
        plt.subplot(2, 10, j+1)
        plt.title(f"Input img for {j+1}")
        plt.imshow(image,cmap="gray")

    for k in range(10):
        if k >= 10: break
        annotation = np.squeeze(annot[k].cpu().numpy())
        annotation = 255/n_classes*(annotation[0]+annotation[1]*2)
        plt.subplot(2, 10, 10+k+1)
        plt.title(f"Annotation for img {k+1}")
        plt.imshow(annotation,cmap="gray")

    plt.show()

def visualize_images_culanes(img, annot):

    plt.figure(figsize=(50, 10))
    for j in range(10):
        if j >= 10: break
        image = np.squeeze(img[j].cpu().numpy())
        plt.subplot(2, 10, j+1)
        plt.title(f"Input img for {j+1}")
        plt.imshow(image,cmap="gray")

    for k in range(10):
        if k >= 10: break
        annotation = np.squeeze(annot[k].cpu().numpy())
        output = np.zeros((annotation.shape[1],annotation.shape[2],3))
        gray = 255/4*(annotation[0]+annotation[1]*2+annotation[2]*3+annotation[3]*4)
        output[gray==255/4] = [0,0,255]
        output[gray==255/4*2] = [0,255,0]
        output[gray==255/4*3] = [255,0,0]
        output[gray==255/4*4] = [255,255,0]
        output = output.astype(np.uint8)
        plt.subplot(2, 10, 10+k+1)
        plt.title(f"Annotation for img {k+1}")
        plt.imshow(output,cmap="gray")

    plt.show()

def visualize_predictions(outputs,epochs,n_classes):

    plt.figure(figsize=(15, 50))
    for i in range(epochs):   

        image = np.squeeze(outputs[i][0][0].detach().cpu().numpy())
        ground_truth = np.squeeze(outputs[i][1][0].detach().cpu().numpy())
        ground_truth = 255/n_classes*(ground_truth[0]+ground_truth[1]*2)
        prediction = np.squeeze(outputs[i][2][0].detach().cpu().numpy())
        prediction = 255/n_classes*(prediction[0]+prediction[1]*2)

        i += 1

        plt.subplot(epochs, 3, 3*i-2)
        plt.title(f"Image for epoch {i}")
        plt.imshow(image,cmap="gray")
        plt.subplot(epochs, 3, 3*i-1)
        plt.title(f"Ground truth for epoch {i}")
        plt.imshow(ground_truth,cmap="gray")
        plt.subplot(epochs, 3, 3*i)
        plt.title(f"Prediction for epoch {i}")
        plt.imshow(prediction,cmap="gray")

    plt.show()