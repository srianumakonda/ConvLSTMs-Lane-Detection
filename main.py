import torch
import adabound
from torch.utils.data import DataLoader
import torchvision.models as models
from torchsummary import summary
import matplotlib.pyplot as plt
from preprocess import *
from utils import *
import loss

from unet import *
from segnet import *
from resnet50unet import *

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_classes = 2
resize = (128,128)
epochs = 10
batch_size = 16
lr = 1e-3
train_dir = "carla-dataset/train/images"
annotation_dir = "carla-dataset/train/annotations"
model_dir = "models/unet.pth"

carla_dataset = CarlaDataset(train_dir,annotation_dir,resize=resize)
# culanes_dataset = CuLanesDataset(train_dir,annotation_dir,5,resize)
data_loader = DataLoader(carla_dataset, batch_size=batch_size, shuffle=False)
# img, annot = next(iter(data_loader))
# print(f"dataset length: {len(carla_dataset)} \nimg shape: {img.shape}, annotations shape: {annot.shape}")
# visualize_images(img,annot,n_classes)

model = ResNet50_UNet(pretrained=True,in_channels=1,out_channels=n_classes,affine=True,track_running_stats=True).to(device)
# summary(model,(1,resize[0],resize[1]))

optim = adabound.AdaBound(model.parameters(), lr=lr, final_lr=0.1)
steps_per_epoch = len(carla_dataset) // batch_size
iou = loss.IoULoss()
dice = loss.DiceLoss()
precision = loss.Precision()
recall = loss.Recall()

model, outputs = train_model(model, data_loader, epochs, steps_per_epoch, device, optim, iou, dice, precision, recall)
visualize_predictions(outputs,epochs,n_classes)