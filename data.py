import os

import easydict
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from yaml import BaseLoader
import model
from dataset import MyDataset
import dataset
from config import *

net = model.MyModel(weights=ResNet50_Weights,num_classes=num_classes).cuda()
resnet50 = model.resnet50(weights=ResNet50_Weights,num_classes=num_classes).cuda()
source_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip()
])
source_dataset = MyDataset(source_dataset_path, transform=source_data_transform)
source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
target_dataset_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip()
])
target_train_dataset = MyDataset(target_dataset_path, transform=target_dataset_transform)
target_train_dataloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
target_test_dataset = MyDataset(target_dataset_path, transform=target_dataset_transform)
target_test_dataloader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=True)