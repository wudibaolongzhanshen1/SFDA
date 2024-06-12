"""
計算normalized self-entropy
l_x_t denotes the predicted probabilityby classifier C_t and N_c refers to the number of classes
"""
import numpy as np
import torch
import torch.nn.functional as F
from model import MyModel
from data import *
from config import *

"""
传入 3*224*224的图片，返回H(x_t)
"""
def calculate_H(num_classes:int,net:MyModel,x_t:torch.Tensor):
    x_t = x_t.unsqueeze(0)
    num_classes = torch.tensor(num_classes)
    _,l_x_t = net(x_t)
    l_x_t = torch.softmax(l_x_t,dim=1)
    H = -1 * (1/torch.log(num_classes)) * torch.sum(l_x_t * torch.log(l_x_t))
    return H.item()

"""
传入 batchsize*3*224*224的图片，返回batchsize个H(x_t)组成的list
"""
def calculate_Hs(num_classes:int,net:MyModel,imgs:torch.Tensor):
    num_classes = torch.tensor(num_classes)
    Hs = []
    for step,img in enumerate(imgs):
        x_t = img.unsqueeze(0)
        _,l_x_t = net(x_t)
        l_x_t = torch.softmax(l_x_t,dim=1)
        H = -1 * (1/torch.log(num_classes)) * torch.sum(l_x_t * torch.log(l_x_t))
        Hs.append(H.item())
    return Hs


"""
construct a class-wise entropy set
返回一维数组H0/..
"""
def calculate_H_class(imgs,net:MyModel,class_:int,H_s:list):
    H_class = []
    _,y_t = net(imgs)
    y_t = F.softmax(y_t,dim=1)
    y_t = torch.argmax(y_t,dim=1)
    for step,item in enumerate(y_t):
        if item == class_:
            H_class.append(H_s[step])
    return H_class


"""
返回： eta
"""
def calculate_eta(imgs,num_classes,net:MyModel,H_s:list):
    # Hc为Hc数组
    H_c = []
    for i in range(num_classes):
        H_c.append(calculate_H_class(imgs,net,i,H_s))
    F=[]
    for i in range(len(H_c)):
        if len(H_c[i]) == 0:
            continue
        F.append(min(H_c[i]))
    return max(F)

def calculate_M_c(imgs,net:MyModel,num_classes:int,class_:int,H_s:list):
    eta = calculate_eta(imgs,num_classes,net)
    _,y_t = net(imgs)
    y_t = F.softmax(y_t,dim=1)
    y_t = torch.argmax(y_t,dim=1)
    M_c = []
    M_c_imgs = []
    for step, item in enumerate(y_t):
        if item == class_ and H_s[step]<eta:
            M_c.append(net.feature_extractor(imgs[step]))
            M_c_imgs.append(imgs[step])
    return M_c,M_c_imgs

net = net.cuda()
def APM_update():
    H_c = []
    features = []
    available_cls = []
    for i in range(num_classes):
        H_c.append([])
        features.append([])
    feature_numpy_for_emergency = []
    #首先找到各类别的H(x_t)即Hc以及索引相对应的feature
    for step,(img,label) in enumerate(target_train_dataloader):
        img = img.cuda()
        feature = net.feature_extractor(img)
        before_softmax = net.classifier_t(feature)
        after_softmax = F.softmax(before_softmax,dim=1)
        pesudo_label = torch.argmax(after_softmax,dim=1)
        feature_numpy_for_emergency.append(feature.data.cpu().numpy())
        H = -1 * (1 / np.log(after_softmax.size(1))) * torch.sum(after_softmax * torch.log(after_softmax+1e-10),dim=1)
        for cls in range(num_classes):
            index_list = torch.where((pesudo_label==cls)==1)[0]
            if len(index_list) == 0:
                continue
            available_cls.append(cls)
            list_H = torch.gather(H,dim=0,index=index_list)
            list_feature = torch.gather(feature.cpu(),dim=0,index=index_list.cpu().unsqueeze(1).repeat(1,2048))
            H_c[cls].append(list_H.cpu().data)
            features[cls].append(list_feature.cpu().data)
    available_cls = np.unique(available_cls)
    prototype_memory = []

    eta = 0
    max_class_prototype_num = 0
    class_prototype_num = []
    for cls in range(num_classes):
        class_prototype_num.append([])
        if cls in available_cls:
            H_c[cls] = np.concatenate(H_c[cls])
            H_c[cls] = torch.from_numpy(H_c[cls])
            features[cls] = np.concatenate(features[cls])
            features[cls] = torch.from_numpy(features[cls])
            class_prototype_num[cls] = len(H_c[cls])
            tmp = min(H_c[cls])
            if tmp > eta:
                eta = tmp
        else:
            class_prototype_num[cls] = 0
        if class_prototype_num[cls] > max_class_prototype_num:
            max_class_prototype_num = class_prototype_num[cls]
    if max_class_prototype_num > 100:
        max_class_prototype_num = 100

    for i in range(num_classes):
        prototype_memory.append([])
    for cls in range(num_classes):
        if cls in available_cls:
            sort_idx = torch.argsort(H_c[cls])
            index_list = sort_idx[:class_prototype_num[cls]]
            if len(index_list) == 0:
                continue
            list_feature = torch.gather(features[cls],dim=0,index=index_list.unsqueeze(1).repeat(1,2048))
            list_feature = np.concatenate([list_feature] * (int(max_class_prototype_num / list_feature.shape[0]) + 1),
                                                  axis=0)
            list_feature = list_feature[:max_class_prototype_num, :]
            prototype_memory[cls].append(list_feature)
            #prototype_memory[0][0]: 291*2048
        else:
            # 后面的情况指的是没有预测为cls的样本，随便找一个样本的feature作为原型
            list_feature = np.concatenate([feature_numpy_for_emergency[0]] * int(max_class_prototype_num),
                                          axis=0)
            list_feature = list_feature[:max_class_prototype_num, :]
            prototype_memory[cls].append(list_feature)
    return prototype_memory,max_class_prototype_num