import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random

class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True):
        #self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
        #                 'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
        #                 'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
        #                 'person','piano','plant','radio','range_hood','sink','sofa','stairs',
        #                 'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        
        self.classnames = list({'water1': 0, 'water2': 1, 'pepsi': 2, 'coca1': 3, 'coca2': 4, 'coca3': 5, 'coca4': 6, 'tea1': 7, 'tea2': 8, 'yogurt': 9, 'ramen1': 10, 'ramen2': 11, 'ramen3': 12, 'ramen4': 13, 'ramen5': 14, 'ramen6': 15, 'ramen7': 16, 'juice1': 17, 'juice2': 18, 'can1': 19, 'can2': 20, 'can3': 21, 'can4': 22, 'can5': 23, 'can6': 24, 'can7': 25, 'can8': 26, 'can9': 27, 'ham1': 28, 'ham2': 29, 'pack1': 30, 'pack2': 31, 'pack3': 32, 'pack4': 33, 'pack5': 34, 'pack6': 35, 'snack1': 36, 'snack2': 37, 'snack3': 38, 'snack4': 39, 'snack5': 40, 'snack6': 41, 'snack7': 42, 'snack8': 43, 'snack9': 44, 'snack10': 45, 'snack11': 46, 'snack12': 47, 'snack13': 48, 'snack14': 49, 'snack15': 50, 'snack16': 51, 'snack17': 52, 'snack18': 53, 'snack19': 54, 'snack20': 55, 'snack21': 56, 'snack22': 57, 'snack23': 58, 'snack24': 59, 'green_apple': 60, 'red_apple': 61, 'tangerine': 62, 'lime': 63, 'lemon': 64, 'yellow_quince': 65, 'green_quince': 66, 'white_quince': 67, 'fruit1': 68, 'fruit2': 69, 'peach': 70, 'banana': 71, 'fruit3': 72, 'pineapple': 73, 'fruit4': 74, 'strawberry': 75, 'cherry': 76, 'red_pimento': 77, 'green_pimento': 78, 'carrot': 79, 'cabbage1': 80, 'cabbage2': 81, 'eggplant': 82, 'bread': 83, 'baguette': 84, 'sandwich': 85, 'hamburger': 86, 'hotdog': 87, 'donuts': 88, 'cake': 89, 'onion': 90, 'marshmallow': 91, 'mooncake': 92, 'shirimpsushi': 93, 'sushi1': 94, 'sushi2': 95, 'big_spoon': 96, 'small_spoon': 97, 'fork': 98, 'knife': 99, 'big_plate': 100, 'small_plate': 101, 'bowl': 102, 'white_ricebowl': 103, 'blue_ricebowl': 104, 'black_ricebowl': 105, 'green_ricebowl': 106, 'black_mug': 107, 'gray_mug': 108, 'pink_mug': 109, 'green_mug': 110, 'blue_mug': 111, 'blue_cup': 112, 'orange_cup': 113, 'yellow_cup': 114, 'big_wineglass': 115, 'small_wineglass': 116, 'glass1': 117, 'glass2': 118, 'glass3': 119, 'background':120}.keys())
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            #print(all_files+"\n")
            ## Select subset for different number of views
            #stride = int(12/self.num_views) # 12 6 4 3 2 1
            #all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new


        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])



class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=3):
        #self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
        #                 'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
        #                 'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
        #                 'person','piano','plant','radio','range_hood','sink','sofa','stairs',
        #                 'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        
        self.classnames = list({'water1': 0, 'water2': 1, 'pepsi': 2, 'coca1': 3, 'coca2': 4, 'coca3': 5, 'coca4': 6, 'tea1': 7, 'tea2': 8, 'yogurt': 9, 'ramen1': 10, 'ramen2': 11, 'ramen3': 12, 'ramen4': 13, 'ramen5': 14, 'ramen6': 15, 'ramen7': 16, 'juice1': 17, 'juice2': 18, 'can1': 19, 'can2': 20, 'can3': 21, 'can4': 22, 'can5': 23, 'can6': 24, 'can7': 25, 'can8': 26, 'can9': 27, 'ham1': 28, 'ham2': 29, 'pack1': 30, 'pack2': 31, 'pack3': 32, 'pack4': 33, 'pack5': 34, 'pack6': 35, 'snack1': 36, 'snack2': 37, 'snack3': 38, 'snack4': 39, 'snack5': 40, 'snack6': 41, 'snack7': 42, 'snack8': 43, 'snack9': 44, 'snack10': 45, 'snack11': 46, 'snack12': 47, 'snack13': 48, 'snack14': 49, 'snack15': 50, 'snack16': 51, 'snack17': 52, 'snack18': 53, 'snack19': 54, 'snack20': 55, 'snack21': 56, 'snack22': 57, 'snack23': 58, 'snack24': 59, 'green_apple': 60, 'red_apple': 61, 'tangerine': 62, 'lime': 63, 'lemon': 64, 'yellow_quince': 65, 'green_quince': 66, 'white_quince': 67, 'fruit1': 68, 'fruit2': 69, 'peach': 70, 'banana': 71, 'fruit3': 72, 'pineapple': 73, 'fruit4': 74, 'strawberry': 75, 'cherry': 76, 'red_pimento': 77, 'green_pimento': 78, 'carrot': 79, 'cabbage1': 80, 'cabbage2': 81, 'eggplant': 82, 'bread': 83, 'baguette': 84, 'sandwich': 85, 'hamburger': 86, 'hotdog': 87, 'donuts': 88, 'cake': 89, 'onion': 90, 'marshmallow': 91, 'mooncake': 92, 'shirimpsushi': 93, 'sushi1': 94, 'sushi2': 95, 'big_spoon': 96, 'small_spoon': 97, 'fork': 98, 'knife': 99, 'big_plate': 100, 'small_plate': 101, 'bowl': 102, 'white_ricebowl': 103, 'blue_ricebowl': 104, 'black_ricebowl': 105, 'green_ricebowl': 106, 'black_mug': 107, 'gray_mug': 108, 'pink_mug': 109, 'green_mug': 110, 'blue_mug': 111, 'blue_cup': 112, 'orange_cup': 113, 'yellow_cup': 114, 'big_wineglass': 115, 'small_wineglass': 116, 'glass1': 117, 'glass2': 118, 'glass3': 119, 'background':120}.keys())

        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)

        return (class_id, im, path)

