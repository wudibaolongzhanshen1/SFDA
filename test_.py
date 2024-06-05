import os

import easydict
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from yaml import BaseLoader

import model
from dataset import MyDataset
import argparse

parser = argparse.ArgumentParser(description='SFDA')
parser.add_argument('--config', type=str, default='config.yaml', help='/path/config_file')
args = parser.parse_args()
config_file_path = args.config
args = yaml.load(open(config_file_path), Loader=BaseLoader)
args = easydict.EasyDict(args)
batch_size = int(args.data.dataloader.batch_size)
num_classes = int(args.data.dataset.n_total)
dataset_root_path = args.data.dataset.root_path
dirs = os.listdir(dataset_root_path)
source_dataset_path = os.path.join(dataset_root_path, dirs[int(args.data.dataset.source)])
target_dataset_path = os.path.join(dataset_root_path, dirs[int(args.data.dataset.target)])
logs_dir = args.log.root_dir
learning_rate = float(args.train.lr)
APM_update_freq = int(args.train.update_freq)
epochs = int(args.train.epochs)
momentum = float(args.train.momentum)
weight_decay = float(args.train.weight_decay)
net = net.MyModel(num_classes=num_classes)

source_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip()
])
source_dataset = MyDataset(source_dataset_path, transform=source_data_transform)
source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
for step, (imgs, labels) in enumerate(source_dataloader):
    print(imgs.shape, labels)
    y_s2t,y_t = net(imgs)
    print(y_s2t.shape,'\n', y_t.shape)
    break