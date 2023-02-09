import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
from tqdm import tqdm

from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
#parser.add_argument("-train_path", type=str, default="/data3/sjyang/dataset/MVCNN/MVCNN_labeled/*/train")
#parser.add_argument("-val_path", type=str, default="/data3/sjyang/dataset/MVCNN/MVCNN_labeled/*/val")
parser.add_argument("-test_path", type=str, default="/data3/sjyang/dataset/MVCNN/MVCNN_ReidResult/*/test")

parser.set_defaults(train=False)

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

def test_function(model,test_dataset):
    all_correct_points = 0
    all_points = 0

    wrong_class = np.zeros(121)
    samples_class = np.zeros(121)
    all_loss = 0
    
    model.eval()
    avgpool = nn.AvgPool1d(1,1)

    total_time = 0.0
    total_print_time = 0.0
    all_target = []
    all_pred = []
    loss_fn = nn.CrossEntropyLoss()
    softmax_fn = nn.Softmax(dim=1)

    save_path = args.test_path.split('_')[0] + '+mvcnn.txt'
    f = open(save_path,'w')
    #for _, data in enumerate(test_dataset,0):
    for data in tqdm(test_dataset):
        N,V,C,H,W = data[1].size()
        in_data = Variable(data[1]).view(-1,C,H,W).cuda()
        target = Variable(data[0]).cuda()

        out_data = model(in_data)
        #pred = torch.max(out_data, 1)[1]
        normed_out_data = softmax_fn(out_data)
        probs, pred = torch.max(normed_out_data, 1)

        all_loss += loss_fn(out_data, target).cpu().data.numpy()
        results = pred == target

        # save result
        #f = open('/data3/sjyang/MVCNN/reid_result.txt','a')
        for i in range(len(pred)):#batchsize 8 
            image_name = data[2][0][i].rsplit('_',1)[0]
            prediction = pred[i]
            prob = probs[i]
            #line_to_write = image_name + ',' + str(int(prediction)) + '\n'
            line_to_write = image_name + ',' + str(int(prediction)) + ',' + str(round(float(prob), 3)) + '\n'
            f.write(line_to_write)
        
        for i in range(results.size()[0]):
            if not bool(results[i].cpu().data.numpy()):
                wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
            samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
        correct_points = torch.sum(results.long())
        
        all_correct_points += correct_points
        all_points += results.size()[0]
       
    f.close()

    print('Total # of test models: ',all_points)
    val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
    acc = all_correct_points.float()/all_points
    val_overall_acc = acc.cpu().data.numpy()
    loss=all_loss/len(test_dataset)

    print('test mean class acc. : ', val_mean_class_acc)
    print('test overall acc. : ', val_overall_acc)
    print('test loss : ',loss)



if __name__ == '__main__':
    args = parser.parse_args()

    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    # STAGE 1
#    log_dir = args.name+'_stage_1'
#    create_folder(log_dir)
#    cnet = SVCNN(args.name, nclasses=120, pretraining=pretraining, cnn_name=args.cnn_name)
#
#    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#    
#    n_models_train = args.num_models*args.num_views
#
#    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
#    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
#
#    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
#    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
#    print('num_train_files: '+str(len(train_dataset.filepaths)))
#    print('num_val_files: '+str(len(val_dataset.filepaths)))
#    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
#    trainer.train(30)

    # STAGE 2
#    log_dir = args.name+'_stage_2'
#    create_folder(log_dir)
        
#    cnet = SVCNN(args.name, nclasses=120, pretraining=pretraining, cnn_name=args.cnn_name)
    #path = '/data3/sjyang/MVCNN/checkpoint/stage_1/model-00020.pth'
#    path = './mvcnn_stage_1/mvcnn/model-00020.pth'
#    cnet.load_state_dict(torch.load(path))

    
#    cnet_2 = MVCNN(args.name, cnet, nclasses=120, cnn_name=args.cnn_name, num_views=args.num_views)
#    del cnet

#    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
#    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
#    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

#    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
#    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
#    print('num_train_files: '+str(len(train_dataset.filepaths)))
#    print('num_val_files: '+str(len(val_dataset.filepaths)))
#    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
#    trainer.train(30)

    log_dir = args.name+'_test'
    create_folder(log_dir)
     
    svcnn = SVCNN(args.name, nclasses=121, pretraining=pretraining, cnn_name=args.cnn_name)
    cnet_test = MVCNN(args.name, svcnn ,nclasses=121, cnn_name=args.cnn_name, num_views=args.num_views)
    path_test = '/data3/sjyang/MVCNN/mvcnn_stage_2/mvcnn/model-00020.pth'
    cnet_test.load_state_dict(torch.load(path_test))
    cnet_test.cuda()
    #cnet_test.eval()
     
    #test_dataset = MultiviewImgDataset(args.test_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
    test_dataset = MultiviewImgDataset(args.test_path, scale_aug=False, rot_aug=False, num_views=args.num_views, test_mode = True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=10)
    print('num_test_files: '+str(len(test_dataset.filepaths)))

    test_function(cnet_test,test_loader)

    print('test is completed') 
