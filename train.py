import os
import sys

import easydict
import numpy as np
import torch
import yaml
from easydl import inverseDecaySheduler, OptimWithSheduler, OptimizerManager
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from yaml import BaseLoader

import lib
import model
import updateAPM
from dataset import MyDataset
import argparse
from data import *

scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=epochs)

optimizer_finetune = OptimWithSheduler(
    optim.SGD(net.feature_extractor.parameters(), lr=learning_rate / 10.0, weight_decay=weight_decay, momentum=momentum,
              nesterov=True),
    scheduler)
optimizer_classifier_s2t = OptimWithSheduler(
    optim.SGD(net.classifier_s2t.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum,
              nesterov=True),
    scheduler)
optimizer_classifier_t = OptimWithSheduler(
    optim.SGD(net.classifier_t.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum,
              nesterov=True),
    scheduler)

best_acc = 0
for i in range(epochs):
    if i % APM_update_freq == 0:
        prototype_memory, num_prototype = updateAPM.APM_update()
    for step, (imgs_target, labels_target) in enumerate(target_train_dataloader):
        net.train()
        imgs_target = imgs_target.cuda()
        # 用来训练第一个分支的label
        labels_source = torch.argmax(resnet50(imgs_target), dim=1)
        proto_feat_tensor = torch.Tensor(prototype_memory)
        proto_feat_tensor = torch.from_numpy(
            np.concatenate(np.concatenate(proto_feat_tensor.numpy(), axis=0), axis=0))  # (3100 * 2048)

        features_target = net.feature_extractor(imgs_target)
        cls_traget = net.classifier_t(features_target)
        cls_source2target = net.classifier_s2t(features_target)

        proto_feat_tensor_normalized = lib.tensor_l2normalization(proto_feat_tensor).cuda()
        features_target_normalized = lib.tensor_l2normalization(features_target)  # (batch_size * 2048)
        # 计算sc
        tmp = torch.mm(features_target_normalized, proto_feat_tensor_normalized.permute(1, 0))  # (batchsize * 3100)
        sc = torch.avg_pool1d(tmp.unsqueeze(0), kernel_size=num_prototype, stride=num_prototype).squeeze(
            0)  # (batchsize * 31)
        sorted_sc_idx = torch.argsort(sc, dim=1, descending=True)  # (batchsize * 31)
        persudo_labels = sorted_sc_idx[:, 0]  # (batchsize)
        second_sc_labels = sorted_sc_idx[:, 1]  # 用来计算Mt2
        d = 1 - tmp  # (batchsize * 3100)
        mt1 = []
        mt2 = []
        for i_2 in range(len(imgs_target)):
            mt1.append(max(d[i_2][persudo_labels[i_2] * 100:persudo_labels[i_2] * 100 + 100]).item())
            mt2.append(max(d[i_2][second_sc_labels[i_2] * 100:second_sc_labels[i_2] * 100 + 100]).item())
        w = ((torch.tensor(mt1) - torch.tensor(mt2)) < 0).cuda()
        # 计算loss
        alpha = (2.0 / (1.0 + np.exp(-10 * step / float(epochs // 2))) - 1.0)
        l_s2t = loss_s2t(cls_source2target, labels_source)
        # 定义对象lt时reduction=none所以对batch的loss进行了保留，默认reduction=mean因此ls2t是一个数而lt是一个数组
        l_t = loss_t(cls_traget, persudo_labels).view(-1, 1).squeeze(1)  # (32)
        # 进行mask后再平均loss
        l_t = torch.mean(l_t * w, dim=0, keepdim=True).squeeze(0)
        l_total = (1 - alpha) * l_s2t + alpha * l_t
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], l_s2t: %f, l_t: %f, alpha: %f, l_total: %f' \
                         % (i, step + 1, len(target_train_dataloader), l_s2t.data.cpu().numpy(),
                            l_t.data.cpu().numpy(), alpha, l_total.data.cpu().numpy()))
        with OptimizerManager([optimizer_finetune, optimizer_classifier_s2t, optimizer_classifier_t]):
            l_total.backward()

    net.eval()
    acc = 0
    for step, (imgs_test, labels_test) in enumerate(target_test_dataloader):
        imgs_test = imgs_test.cuda()
        labels_test = labels_test.cuda()
        features_test = net.feature_extractor(imgs_test)
        cls_test = net.classifier_t(features_test)
        cls_test = torch.argmax(cls_test, dim=1)
        acc += torch.sum(cls_test == labels_test).float()
    acc = acc / len(labels_test)
    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), 'models/best_model.pth')
    print('\r epoch: %d, acc: %f ' \
          % (i, acc))
