"""
計算normalized self-entropy
l_x_t denotes the predicted probabilityby classifier C_t and N_c refers to the number of classes
"""
import torch
import torch.nn.functional as F
from model import MyModel


def calculate_H(num_classes:int,net:MyModel,x_t:torch.Tensor):
    x_t = x_t.unsqueeze(0)
    num_classes = torch.tensor(num_classes)
    _,l_x_t = net(x_t)
    H = -1 * (1/torch.log(num_classes)) * torch.sum(l_x_t * torch.log(l_x_t))
    return H.item()

"""
construct a class-wise entropy set
"""
def calculate_H_c(imgs,num_classes,net:MyModel,class_:int):
    H_c = []
    _,y_t = net(imgs)
    y_t = F.softmax(y_t,dim=1)
    y_t = torch.argmax(y_t,dim=1)
    for step,item in enumerate(y_t):
        if item == class_:
            H_c.append(calculate_H(num_classes,net,imgs[step]))
    # 为什么返回的H_c为nan
    return H_c

