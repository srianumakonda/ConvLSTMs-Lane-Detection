import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class CarlaDataset(Dataset):
    
    def __init__(self, root_dir, annotation_dir, resize=(128,128), transform=None):
        
        """
        @param:
            root_dir (str): path of the input images from the CARLA simulator
            annotation_dir (str): path of the input annotation images
            resize(tup): pass in a tuple of the new image + annotation input sizes   
            transform(torvision.transforms): input a torchvision.transforms attribute to the class for further data preprocessing.
            
        """
        
        self.root_dir = root_dir
        self.annotation_dir = annotation_dir
        self.resize = resize
        self.dataset = []
        
        for (img_name,annotation_name) in zip(os.listdir(self.root_dir),os.listdir(self.annotation_dir)):
            self.dataset.append([os.path.join(root_dir,img_name),os.path.join(annotation_dir,annotation_name)])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, annotation = self.dataset[idx]
        img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,self.resize)
        img = img/255.0
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        
        annotation = cv2.imread(annotation,0)
        annotation = cv2.resize(annotation,self.resize)
        annotation = annotation.astype(np.float32)
        
        left_lane = annotation.copy()
        left_lane[left_lane!=1] = 0
        right_lane = annotation.copy()
        right_lane[right_lane!=2] = 0
        right_lane[right_lane==2] = 1
        
        annotation = np.stack([left_lane,right_lane])

        img = torch.from_numpy(img)
        annotation = torch.from_numpy(annotation) 
        
        return img, annotation