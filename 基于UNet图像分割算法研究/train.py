from torch import nn
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import time

from unet import UNet
from  Pascal_Dataset import VOCSegmentationDataset
from test import test_net
from Person_dataset import TestDataset,TrainDataset
from Road_dataset import Train_Dataset
from test import  compute_miou


class dice_ce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_ce_loss, self).__init__()
        self.batch = batch
        self.relu=nn.ReLU()
        self.ce_loss = nn.CrossEntropyLoss()

    def soft_dice_coeff(self, y_true, y_pred):
        y_pred = self.relu(y_pred[:,1,:,:])
        smooth = 0.00001  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred,y_true):
        a = self.ce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return  a + b





def dataset(dataset):
    if dataset=='voc':
        global train_dataset
        global train_val_dataset
        train_dir = '../pythonProject1/Pascal_data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
        train_val_dir = '../pythonProject1/Pascal_data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt'
        train_dataset = VOCSegmentationDataset(train_dir)
        train_val_dataset= VOCSegmentationDataset(train_val_dir)
    if dataset=='person':
        train_dir = 'C:/Users/86159/PycharmProjects/FCN语义分割/dataset'
        train_val_dir = 'C:/Users/86159/PycharmProjects/FCN语义分割/dataset'
        train_dataset = TrainDataset(train_dir)
        train_val_dataset = TestDataset(train_val_dir)
    if dataset=='road':
        train_dir = 'C:/Users/86159/PycharmProjects/Dlinknet/Road-Extraction-master/road_dataset'
        train_dataset = Train_Dataset(train_dir,val='no')
        train_val_dataset = Train_Dataset(train_dir,val='yes')




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


def train_net(net, device, epochs=25, batch_size=1, lr=1e-2):

    dataset('road')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_val_loader= DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #criterion=nn.CrossEntropyLoss()
    #criterion=nn.BCEWithLogitsLoss()
    criterion=dice_ce_loss()
    file_path="log.txt"




    # 训练模型
    for epoch in range(epochs):
        total_train_loss = 0
        total_train_val_loss = 0
        accuracy=0
        losss=0
        miou=0
        for i, (inputs,labels) in enumerate(train_loader, 0):
            net.train()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            labels=labels.squeeze(1).to(torch.int64)
            #visualize_output(outputs,'road')
            #visualize_lable(labels,'road')
            loss = criterion(outputs, labels)
            losss+=loss.item()
            total_train_loss+=loss.item()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, batch {i+1} loss: {loss.item()} ')
            # 保存模型
            if (i+1)%100==0:
                print(f'Epoch {epoch + 1}, batch {i+1}, average loss: {losss/100} ')
                losss=0
            if(i+1)%1000==0:
                with open('unet(road)' + '_train_' + file_path, 'a') as file:
                    file.write(f'Epoch {epoch + 1}, batch {i+1}, train loss: {total_train_loss / 1000} \n')
                total_train_loss = 0
                net.eval()
                for j, (inputs, labels) in enumerate(train_val_loader, 0):
                    with torch.no_grad():
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = net(inputs)
                        labels = labels.squeeze(1).to(torch.int64)
                        loss = criterion(outputs, labels)
                        total_train_val_loss += loss.item()
                        accuracy += Accuracy(outputs, labels)
                        miou+=compute_miou(outputs,labels)
                        if (j + 1) % 226 == 0:
                            with open('unet(road)' + '_val_' + file_path, 'a') as file:
                                file.write(f'Epoch {epoch + 1},  batch {i+1}, val loss: {total_train_val_loss / 226}, accuracy:{accuracy / 226}, miou:{miou / 226} \n')
                            total_train_val_loss = 0
                            accuracy = 0
                            miou=0


        scheduler.step()
        torch.save(net, 'unet(road)+.pth')









if __name__ == '__main__':
    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建模型并移动到设备


    #net =UNet(3,2).to(device)
    #net=FCN8s(2).to(device)
    #net=seunet(2).to(device)
    #net=seresunet(2).to(device)
    #net=resunet(2).to(device)



    net=torch.load("road_model/seresunet+(road)+.pth")
    start=time.time()

    #train_net(net, device)
    end=time.time()
    print(f'训练时间为 {end-start}s')


    # 训练模型

    test_net(net,device)











