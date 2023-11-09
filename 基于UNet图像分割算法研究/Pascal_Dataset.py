import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torch.utils.data import Dataset


# 21个类
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# 每个类对应的RGB值
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

# 下边就是将label中每种颜色映射成0-20的数字
cm2lbl = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引


def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵




def img_transforms(im, label):
    im_tfs = torchvision.transforms.Compose([
        torchvision.transforms.Resize((320, 480)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    g=torchvision.transforms.Resize((320, 480))
    im = im_tfs(im)
    label = g(label)
    label = image2label(label)
    label = torch.from_numpy(label)

    return im, label


class VOCSegmentationDataset(Dataset):
    def __init__(self, dir, transform=img_transforms):
        self.dir=dir
        self.images = []
        self.lables=[]
        self.transform = transform
        with open(dir,'r') as f:
            for line in f.readlines():
                self.images.append(line.strip())

        with open(dir, 'r') as f:
            for line in f.readlines():
                self.lables.append(line.strip())
        #print(self.images)
        #print(self.lables)
    def __len__(self):
        #print(len(self.images))
        return len(self.images)

    def __getitem__(self, index):
        # 读取图像和标签
          image_path = os.path.join('../pythonProject1/Pascal_data/VOCdevkit/VOC2012/JPEGImages', self.images[index] + '.png')
          label_path = os.path.join('../pythonProject1/Pascal_data/VOCdevkit/VOC2012/SegmentationClass', self.lables[index] + '.png')
          #print(image_path)
          #print(label_path)
          image = Image.open(image_path)
          label = Image.open(label_path).convert('RGB')

          #l=np.array(label)
          #print(np.max(l))


        # 进行预处理操作
          if self.transform is not None:
              image, label = self.transform(image, label)
              return image, label