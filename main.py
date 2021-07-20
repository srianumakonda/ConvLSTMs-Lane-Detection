import torch
import adabound
from torch.utils.data import DataLoader
import torchvision.models as models
from torchsummary import summary
import matplotlib.pyplot as plt
from convlstm_models import *
from models import *
from utils import *

torch.cuda.empty_cache()
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 3
n_classes = 2
resize = (128,128)
epochs = 10
batch_size = 64
lr = 1e-3
train_dir = "carla-dataset/train/images"
annotation_dir = "carla-dataset/train/annotations"
val_dir = "carla-dataset/validation/images"
val_annotation_dir = "carla-dataset/validation/annotations"
model_dir = "models/vgg16unet.pth"

train_dataset = CarlaDataset(train_dir,annotation_dir,resize=resize)
val_dataset = CarlaDataset(val_dir,val_annotation_dir,resize=resize)
# culanes_dataset = CuLanesDataset(train_dir,annotation_dir,5,resize)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# img, annot = next(iter(data_loader))
# print(f"dataset length: {len(train_dataset)}, validation length: {len(val_dataset)} \nimg shape: {img.shape}, annotations shape: {annot.shape}")
# visualize_images(img,annot,n_classes)

# remember: input_dim = 2 if loc="end" or input_dim = 512 if loc = "bridge"
# model = VGG16UNetLSTM(in_channels=in_channels,out_channels=n_classes,input_dim=2,n_layers=2,loc="end").to(device)
model = VGG16UNet(in_channels=in_channels,out_channels=n_classes).to(device)
# model = VGG16_UNet(pretrained=False,in_channels=1,out_channels=n_classes).to(device)
summary(model,(in_channels,resize[0],resize[1]),device="cpu")

#TODO: WHEN NEW DATASET COMES, MAKE SURE TO PREPROCESS IT SO THAT IT CAN BE SET TO THE AMOUNT OF PROPER SEQUENCES. REFER TO 
# https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/dataset.py 
# print(model(torch.randn(16,5,in_channels,128,128)).shape) #b, t, c, h, w

# optim = adabound.AdaBound(model.parameters(), lr=lr, final_lr=0.1)
# steps_per_epoch = len(carla_dataset) // batch_size
# iou = IoULoss()
# dice = DiceLoss()
# precision = Precision()
# recall = Recall()

# model, outputs = train_model(model, data_loader, val_loader, epochs, steps_per_epoch, device, optim, iou, dice, precision, recall)
# torch.save(model.state_dict(),model_dir)
# print("Model saved")
# visualize_predictions(outputs,epochs,n_classes)