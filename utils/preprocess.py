import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import glob

class CuLanesDataset(Dataset):
    
    def __init__(self, train_dir, annotation_dir, factor, resize):
        
        """
        @param:
            root_dir (str): path of the input images from the CARLA simulator
            annotation_dir (str): path of the input annotation images
            factor(int): input of teh resize factor for input images        
        """
        
        self.train_dir = train_dir
        self.annotation_dir = annotation_dir
        self.factor = factor
        self.resize=resize
        self.dataset = []

        img_name = glob.glob(train_dir+"/*/*.jpg")
        annotation_name = glob.glob(annotation_dir+"/*/*.png")
        
        for (img,annotation) in zip(img_name,annotation_name):
            self.dataset.append([img,annotation])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, annotation = self.dataset[idx]
        img = cv2.imread(img,0)
        resize = (img.shape[1]//self.factor,img.shape[0]//self.factor)
        if self.resize:
            resize = self.resize
        img = cv2.resize(img,resize)
        img = img/255.0
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        
        annotation = cv2.imread(annotation,0)
        annotation = cv2.resize(annotation,resize)
        annotation = annotation.astype(np.float32)
        
        outer_left_lane = annotation.copy()
        outer_left_lane[outer_left_lane!=1] = 0

        middle_left_lane = annotation.copy()
        middle_left_lane[middle_left_lane!=2] = 0
        middle_left_lane[middle_left_lane==2] = 1

        middle_right_lane = annotation.copy()
        middle_right_lane[middle_right_lane!=3] = 0
        middle_right_lane[middle_right_lane==3] = 1

        outer_right_lane = annotation.copy()
        outer_right_lane[outer_right_lane!=4] = 0
        outer_right_lane[outer_right_lane==4] = 1

        annotation = np.stack([outer_left_lane,middle_left_lane,middle_right_lane,outer_right_lane])

        img = torch.from_numpy(img)
        annotation = torch.from_numpy(annotation) 
        
        return img, annotation

class CarlaDataset(Dataset):
    
    def __init__(self, train_dir, annotation_dir, resize=(128,128)):
        
        """
        @param:
            root_dir (str): path of the input images from the CARLA simulator
            annotation_dir (str): path of the input annotation images
            resize(tup): pass in a tuple of the new image + annotation input sizes              
        """
        
        self.train_dir = train_dir
        self.annotation_dir = annotation_dir
        self.resize = resize
        self.dataset = []
        
        for (img_name,annotation_name) in zip(os.listdir(self.train_dir),os.listdir(self.annotation_dir)):
            self.dataset.append([os.path.join(self.train_dir,img_name),os.path.join(self.annotation_dir,annotation_name)])

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread("CULanes/driver_161_90frame_labels/06030819_0755.MP4/00000.png",0)

    outer_left_lane = img.copy()
    outer_left_lane[outer_left_lane!=1] = 0

    middle_left_lane = img.copy()
    middle_left_lane[middle_left_lane!=2] = 0
    middle_left_lane[middle_left_lane==2] = 1

    middle_right_lane = img.copy()
    middle_right_lane[middle_right_lane!=3] = 0
    middle_right_lane[middle_right_lane==3] = 1

    outer_right_lane = img.copy()
    outer_right_lane[outer_right_lane!=4] = 0
    outer_right_lane[outer_right_lane==4] = 1

    annotation = np.stack([outer_left_lane,middle_left_lane,middle_right_lane,outer_right_lane])
    
    image = 255/4*(annotation[0]+annotation[1]*2+annotation[2]*3+annotation[3]*4)

    zeros = np.zeros((590,1640,3))
    zeros[image==255/4] = [0,0,255]
    zeros[image==255/4*2] = [0,255,0]
    zeros[image==255/4*3] = [255,0,0]
    zeros[image==255/4*4] = [255,255,0]
    
    # output = np.where(annotation[0]==1,np.array([0,0,255]),output)
    #     output = np.where(annotation[1]==1,np.array([0,255,0]),output)
    #     output = np.where(annotation[2]==1,np.array([255,0,0]),output)
    #     output = np.where(annotation[3]==1,np.array([255,255,0]),output)



    plt.imshow(zeros,cmap="gray")
    plt.show()