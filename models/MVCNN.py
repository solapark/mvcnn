import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):

    def __init__(self, name, nclasses=121, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)

        #self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
        #                 'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
        #                 'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
        #                 'person','piano','plant','radio','range_hood','sink','sofa','stairs',
        #                 'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.classnames= list({'water1': 0, 'water2': 1, 'pepsi': 2, 'coca1': 3, 'coca2': 4, 'coca3': 5, 'coca4': 6, 'tea1': 7, 'tea2': 8, 'yogurt': 9, 'ramen1': 10, 'ramen2': 11, 'ramen3': 12, 'ramen4': 13, 'ramen5': 14, 'ramen6': 15, 'ramen7': 16, 'juice1': 17, 'juice2': 18, 'can1': 19, 'can2': 20, 'can3': 21, 'can4': 22, 'can5': 23, 'can6': 24, 'can7': 25, 'can8': 26, 'can9': 27, 'ham1': 28, 'ham2': 29, 'pack1': 30, 'pack2': 31, 'pack3': 32, 'pack4': 33, 'pack5': 34, 'pack6': 35, 'snack1': 36, 'snack2': 37, 'snack3': 38, 'snack4': 39, 'snack5': 40, 'snack6': 41, 'snack7': 42, 'snack8': 43, 'snack9': 44, 'snack10': 45, 'snack11': 46, 'snack12': 47, 'snack13': 48, 'snack14': 49, 'snack15': 50, 'snack16': 51, 'snack17': 52, 'snack18': 53, 'snack19': 54, 'snack20': 55, 'snack21': 56, 'snack22': 57, 'snack23': 58, 'snack24': 59, 'green_apple': 60, 'red_apple': 61, 'tangerine': 62, 'lime': 63, 'lemon': 64, 'yellow_quince': 65, 'green_quince': 66, 'white_quince': 67, 'fruit1': 68, 'fruit2': 69, 'peach': 70, 'banana': 71, 'fruit3': 72, 'pineapple': 73, 'fruit4': 74, 'strawberry': 75, 'cherry': 76, 'red_pimento': 77, 'green_pimento': 78, 'carrot': 79, 'cabbage1': 80, 'cabbage2': 81, 'eggplant': 82, 'bread': 83, 'baguette': 84, 'sandwich': 85, 'hamburger': 86, 'hotdog': 87, 'donuts': 88, 'cake': 89, 'onion': 90, 'marshmallow': 91, 'mooncake': 92, 'shirimpsushi': 93, 'sushi1': 94, 'sushi2': 95, 'big_spoon': 96, 'small_spoon': 97, 'fork': 98, 'knife': 99, 'big_plate': 100, 'small_plate': 101, 'bowl': 102, 'white_ricebowl': 103, 'blue_ricebowl': 104, 'black_ricebowl': 105, 'green_ricebowl': 106, 'black_mug': 107, 'gray_mug': 108, 'pink_mug': 109, 'green_mug': 110, 'blue_mug': 111, 'blue_cup': 112, 'orange_cup': 113, 'yellow_cup': 114, 'big_wineglass': 115, 'small_wineglass': 116, 'glass1': 117, 'glass2': 118, 'glass3': 119, 'background':120}.keys())
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,40)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096,121)
            self.net_2._modules['0'] = nn.Linear(32768,4096)
    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model):

    def __init__(self, name, model, nclasses=121, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)

       # self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
       #                  'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
       #                  'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
       #                  'person','piano','plant','radio','range_hood','sink','sofa','stairs',
       #                  'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.classnames= list({'water1': 0, 'water2': 1, 'pepsi': 2, 'coca1': 3, 'coca2': 4, 'coca3': 5, 'coca4': 6, 'tea1': 7, 'tea2': 8, 'yogurt': 9, 'ramen1': 10, 'ramen2': 11, 'ramen3': 12, 'ramen4': 13, 'ramen5': 14, 'ramen6': 15, 'ramen7': 16, 'juice1': 17, 'juice2': 18, 'can1': 19, 'can2': 20, 'can3': 21, 'can4': 22, 'can5': 23, 'can6': 24, 'can7': 25, 'can8': 26, 'can9': 27, 'ham1': 28, 'ham2': 29, 'pack1': 30, 'pack2': 31, 'pack3': 32, 'pack4': 33, 'pack5': 34, 'pack6': 35, 'snack1': 36, 'snack2': 37, 'snack3': 38, 'snack4': 39, 'snack5': 40, 'snack6': 41, 'snack7': 42, 'snack8': 43, 'snack9': 44, 'snack10': 45, 'snack11': 46, 'snack12': 47, 'snack13': 48, 'snack14': 49, 'snack15': 50, 'snack16': 51, 'snack17': 52, 'snack18': 53, 'snack19': 54, 'snack20': 55, 'snack21': 56, 'snack22': 57, 'snack23': 58, 'snack24': 59, 'green_apple': 60, 'red_apple': 61, 'tangerine': 62, 'lime': 63, 'lemon': 64, 'yellow_quince': 65, 'green_quince': 66, 'white_quince': 67, 'fruit1': 68, 'fruit2': 69, 'peach': 70, 'banana': 71, 'fruit3': 72, 'pineapple': 73, 'fruit4': 74, 'strawberry': 75, 'cherry': 76, 'red_pimento': 77, 'green_pimento': 78, 'carrot': 79, 'cabbage1': 80, 'cabbage2': 81, 'eggplant': 82, 'bread': 83, 'baguette': 84, 'sandwich': 85, 'hamburger': 86, 'hotdog': 87, 'donuts': 88, 'cake': 89, 'onion': 90, 'marshmallow': 91, 'mooncake': 92, 'shirimpsushi': 93, 'sushi1': 94, 'sushi2': 95, 'big_spoon': 96, 'small_spoon': 97, 'fork': 98, 'knife': 99, 'big_plate': 100, 'small_plate': 101, 'bowl': 102, 'white_ricebowl': 103, 'blue_ricebowl': 104, 'black_ricebowl': 105, 'green_ricebowl': 106, 'black_mug': 107, 'gray_mug': 108, 'pink_mug': 109, 'green_mug': 110, 'blue_mug': 111, 'blue_cup': 112, 'orange_cup': 113, 'yellow_cup': 114, 'big_wineglass': 115, 'small_wineglass': 116, 'glass1': 117, 'glass2': 118, 'glass3': 119, 'background':120}.keys())
        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))

