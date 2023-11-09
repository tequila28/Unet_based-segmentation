import numpy as np
import torchvision
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage, ToTensor, transforms
from  Pascal_Dataset import VOCSegmentationDataset
from  Person_dataset import TestDataset
from Road_dataset import  Train_Dataset


#test_dir= 'Pascal_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
#test_dataset=VOCSegmentationDataset(test_dir)

test_dir= 'C:/Users/86159/PycharmProjects/Dlinknet/Road-Extraction-master/road_dataset'
test_dataset=Train_Dataset(test_dir,val='yes')


#train_val_dir = 'C:/Users/86159/PycharmProjects/FCN语义分割/dataset'
#test_dataset = TestDataset(train_val_dir)


colors = [
    (0, 0, 0),          # 背景
    (128, 0, 0),        # aeroplane
    (0, 128, 0),        # bicycle
    (128, 128, 0),      # bird
    (0, 0, 128),        # boat
    (128, 0, 128),      # bottle
    (0, 128, 128),      # bus
    (128, 128, 128),    # car
    (64, 0, 0),         # cat
    (192, 0, 0),        # chair
    (64, 128, 0),       # cow
    (192, 128, 0),      # diningtable
    (64, 0, 128),       # dog
    (192, 0, 128),      # horse
    (64, 128, 128),     # motorbike
    (192, 128, 128),    # person
    (0, 64, 0),         # potted plant
    (128, 64, 0),       # sheep
    (0, 192, 0),        # sofa
    (128, 192, 0),      # train
    (0, 64, 128),       # tv/monitor
]

def Accuracy(output, label):
    # 将输出转换为预测的类别
    pred = torch.argmax(output, dim=1)
    # 计算预测正确的像素数
    correct_pixels = torch.sum(pred == label)
    # 计算总的像素数
    total_pixels = label.numel()
    # 计算精度
    accuracy = correct_pixels.float() / total_pixels
    return accuracy

def visualize_output(output,type):
    # 将输出和标签的张量转换为图像格式，并将像素值缩放到 0-1 的范围
    if type=='voc':
       output = output.detach()
       output = output.argmax(1).squeeze(0).cpu().numpy()
       color_image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
       for i in range(len(colors)):
           color_image[output == i] = colors[i]
       #print(color_image.shape)
       plt.imshow(color_image)
       plt.axis("off")
       plt.show()
    if type=='person':
        output = output.detach()
        output = output.argmax(1).squeeze(0).cpu().numpy()
        plt.imshow(output)
        plt.axis("off")
        plt.show()
    if type=='road':
        output = output.detach()
        output = output.argmax(1).squeeze(0).cpu().numpy()
        #output = output.squeeze(0).cpu().numpy()
        #output=output.squeeze(0)
        plt.imshow(output)
        plt.axis("off")
        plt.show()


def visualize_lable(lable,type):
    # 将输出和标签的张量转换为图像格式，并将像素值缩放到 0-1 的范围
    if type == 'voc':
        lable = lable.detach()
        #print(output.size())
        lable = lable.squeeze(0).cpu().numpy()
        #print(torch.sum(output))
        color_image = np.zeros((320, 480, 3), dtype=np.uint8)
        #print(lable.shape)
        for i in range(len(colors)):
            color_image[lable == i] = colors[i]
        #print(color_image.shape)
        plt.imshow(color_image)
        plt.axis("off")
        plt.show()
    if type=='person':
        lable = lable.detach()
        lable = lable.squeeze(0).cpu().numpy()
        plt.imshow(lable)
        plt.axis("off")
        plt.show()
    if type=='road':
        lable = lable.detach()
        lable = lable.squeeze(0).cpu().numpy()
        plt.imshow(lable)
        plt.axis("off")
        plt.show()




def compute_miou(y_pred, y_true):

    y_pred=torch.argmax(y_pred,dim=1)
    if y_true.size() != y_pred.size():
        raise ValueError("Input shapes do not match.")

    # 计算IoU为背景和前景
    iou_list = []
    for label in [0, 1]:  # 0 for background, 1 for foreground
        intersection = torch.logical_and(y_true == label, y_pred == label).sum().float()
        union = torch.logical_or(y_true == label, y_pred == label).sum().float()
        iou_list.append((intersection + 1e-6) / (union + 1e-6))

    # 计算mIoU
    mIoU = sum(iou_list) / len(iou_list)
    return mIoU

def test_net(net, device, batch_size=1):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    net.eval()
    accuracy = 0
    iou=0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            accuracy+=Accuracy(outputs,labels)
            labels = labels.squeeze(1).to(torch.int64)
            iou+=compute_miou(outputs,labels)
            #visualize_lable(labels,'road')
            #visualize_output(outputs,'road')
        print(iou/len(test_loader))
        print(accuracy / len(test_loader))


