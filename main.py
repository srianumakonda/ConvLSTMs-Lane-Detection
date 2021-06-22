import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
from preprocess import *
from utils import *
import loss

from unet import *

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_classes = 2
resize = (128,128)
epochs = 10
batch_size = 16
lr = 1e-3
weight_decay = 1e-7
train_dir = "carla-dataset/train/images"
annotation_dir = "carla-dataset/train/annotations/"
model_dir = "models/vgg_encoder.pth"

carla_dataset = CarlaDataset(train_dir,annotation_dir,resize=resize)
data_loader = DataLoader(carla_dataset, batch_size=batch_size, shuffle=True)
img, annot = next(iter(data_loader))
# print(f"dataset length: {len(carla_dataset)} \nimg shape: {img.shape}, annotations shape: {annot.shape}")
# visualize_images(img,annot,n_classes)

model = UNet_Model(in_channels=1,out_channels=n_classes,padding=True,affine=True,track_running_stats=True).to(device)
# summary(model,(1,resize[0],resize[1]))

optim = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
steps_per_epoch = len(carla_dataset) // batch_size
iou = loss.IoULoss()
dice = loss.DiceLoss()

model, outputs = train_model(model, data_loader, epochs, steps_per_epoch, device, optim, iou, dice)
visualize_predictions(outputs,epochs,n_classes)