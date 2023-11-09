import os

import torchvision
from PIL import Image
import torch
from torch.utils.data import Dataset
def my_train_Dataset(root_dir):
    train_image=[]
    lable_image=[]
    train=os.listdir(os.path.join(root_dir, 'training'))
    for i in range(len(train)):
        if i%2==0:
            train_image.append(train[i])
        if i%2==1:
            lable_image.append(train[i])
    return  train_image,lable_image

def my_test_Dataset(root_dir):
    test_image=[]
    lable_image=[]
    test=os.listdir(os.path.join(root_dir, 'testing'))
    for i in range(len(test)):
        if i%2==0:
            test_image.append(test[i])
        if i%2==1:
            lable_image.append(test[i])
    return  test_image,lable_image

class TrainDataset(Dataset):
    def __init__(self, root_dir, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((320, 480)),
        torchvision.transforms.ToTensor(),
    ])):
        self.root_dir = root_dir
        self.transform = transform
        self.train_images,self.lable_images = my_train_Dataset('C:/Users/86159/PycharmProjects/FCN语义分割/dataset')
    def __len__(self):
        return len(self.train_images)
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, 'training', self.train_images[index])
        label_path = os.path.join(self.root_dir, 'training', self.lable_images[index])
        image = Image.open(image_path)
        label = Image.open(label_path)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((320, 480)),
                                                                           torchvision.transforms.ToTensor()])):
        self.root_dir = root_dir
        self.transform = transform
        self.test_images,self.lable_images = my_test_Dataset('C:/Users/86159/PycharmProjects/FCN语义分割/dataset')
    def __len__(self):
        return len(self.test_images)
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, 'testing', self.test_images[index])
        label_path = os.path.join(self.root_dir, 'testing', self.lable_images[index])
        image = Image.open(image_path)
        label = Image.open(label_path)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label