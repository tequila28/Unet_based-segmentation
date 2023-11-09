import os

import cv2
import torchvision
from PIL import Image
import torch
from torch.utils.data import Dataset
def my_train_Dataset(root_dir,val):
    train_image=[]
    lable_image=[]
    train=os.listdir(os.path.join(root_dir, 'train'))
    if val=='no':
      for i in range(len(train)):
          if i%2==0 and i<12000:
            lable_image.append(train[i])
          if i%2==1 and i<12000:
            train_image.append(train[i])
      return  train_image,lable_image

    if val=='yes':
      for i in range(len(train)):
          if i%2==0 and i>=12000:
            lable_image.append(train[i])
          if i%2==1 and i>=12000:
            train_image.append(train[i])
      return  train_image,lable_image


class Train_Dataset(Dataset):
    def __init__(self, root_dir, val,transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((1024, 1024)),
        torchvision.transforms.ToTensor(),
    ])):
        self.root_dir = root_dir
        self.transform = transform
        self.train_images,self.lable_images = my_train_Dataset('C:/Users/86159/PycharmProjects/Dlinknet/Road-Extraction-master/road_dataset',val=val)
    def __len__(self):
        return len(self.train_images)
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, 'train', self.train_images[index])
        label_path = os.path.join(self.root_dir, 'train', self.lable_images[index])
        image = Image.open(image_path)
        label  = Image.fromarray(cv2.imread(label_path, cv2.IMREAD_GRAYSCALE))
        image.show()
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

