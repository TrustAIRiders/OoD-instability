#!/usr/bin/env python

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import random

import torch
import torchvision
import torch.nn.functional as F

from models.mobilenetv2 import MobileNetV2
from models.resnet import ResNet, Bottleneck
from models.resnet2 import ResNet_model2
from models.resnet3 import ResNet_model3, BasicBlock


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic  = True
torch.set_num_threads(1)

os.environ["CUDA_VISIBLE_DEVICES"]=str(0) 
device = torch.device("cuda:0")

BATCH_SIZE = 128
dataset_path = "./"

def load_data(model_full_name): 
    if "_resnet_" in model_full_name.lower():
        model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=10)
        
        if "_resnet_type_0" in model_full_name.lower():
            model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=10) #101
            
        if "_resnet_type_1" in model_full_name.lower():
            model = ResNet_model2() #101
            
        if "_resnet_type_2" in model_full_name.lower():
            model = ResNet_model3(BasicBlock, [18, 18, 18])    #110      
    
    if "_mobilenetv2_" in model_full_name.lower():
        model = MobileNetV2(num_classes=100)
        
    state_dict = torch.load("./saved_models/{}.ckpt".format(model_full_name))["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '')] = state_dict.pop(key)
            
    model.load_state_dict(state_dict)
    model.to(device)   
    model.eval()
    
    if "cifar10_" in model_full_name.lower():
        cifar_normalize_mean= (0.4914, 0.4822, 0.4465)
        cifar_normalize_std= (0.2023, 0.1994, 0.2010)
        
    if "cifar100_" in model_full_name.lower():
        cifar_normalize_mean = (0.5071, 0.4867, 0.4408)
        cifar_normalize_std = (0.2675, 0.2565, 0.2761)          
        
        
    cifar_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(cifar_normalize_mean, cifar_normalize_std)
    ])   
    
    cifar10_train_set = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=cifar_transform)
    cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_set, batch_size=BATCH_SIZE, shuffle=False)

    cifar10_test_set = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=cifar_transform)
    cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    cifar100_train_set = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=cifar_transform)
    cifar100_train_loader = torch.utils.data.DataLoader(cifar100_train_set, batch_size=BATCH_SIZE, shuffle=False)

    cifar100_test_set = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=cifar_transform)
    cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    svhn_set = torchvision.datasets.SVHN(root=dataset_path, split="train", download=True, transform=cifar_transform)
    svhn_loader = torch.utils.data.DataLoader(svhn_set, batch_size=BATCH_SIZE, shuffle=False)   
  
    if "cifar10_" in model_full_name:
        loaders = {
            "train": cifar10_train_loader,
            "test": cifar10_test_loader,
            "ood_svhn": svhn_loader,
            "ood_cifar100": cifar100_train_loader
        }            
        
    if "cifar100_" in model_full_name:
        loaders = {
            "train": cifar100_train_loader,
            "test": cifar100_test_loader,
            "ood_svhn": svhn_loader,
            "ood_cifar10": cifar10_train_loader
        }            

    return model, loaders


def obtain_features(model, x):
    if isinstance(model, ResNet_model2):
        for name, module in model._modules["resnet"]._modules.items():
            x = module(x)
            if name == 'layer4':
                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)
                return x    
    
    if isinstance(model, ResNet_model3):
        for name, module in model._modules.items():
            x = module(x)
            if name == 'layer3':
                x = F.avg_pool2d(x, x.size()[3])
                x = x.view(x.size(0), -1)
                return x
            
    if isinstance(model, ResNet):
        for name, module in model._modules.items():
            x = module(x)
            if name == 'layer4':
                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)
                return x
        
    if isinstance(model, MobileNetV2):
        for name, module in model._modules.items():
            x = module(x)
            if name == 'relu':
                x = F.avg_pool2d(x, 8)
                x = x.view(x.shape[0], -1)
                return x                

def get_df(model, loader, batch_size=BATCH_SIZE):
    all_out = []
    with torch.no_grad():
        for i_batch, (data, targets) in enumerate(loader):
            sys.stdout.write("{}/{}\r".format(i_batch, len(loader)))
            sys.stdout.flush()

            data = data.to(device)

            outputs = model(data)
            features = obtain_features(model, data)
            for i in range(len(data)):
                out = {}
                out["id"] = i_batch * batch_size + i
                out["original_label"] = targets[i].item()
                out["features"] = np.array(features[i].detach().cpu())
                out["classifier"] = np.array(outputs[i].detach().cpu())
                all_out.append(out)
                        
    df = pd.DataFrame(all_out)
    return df

def save_df(path, df):
    directory = os.path.dirname(path)
    Path(directory).mkdir(parents=True, exist_ok=True)

    np.set_printoptions(suppress=True, threshold=np.inf, precision=8, floatmode="maxprec_equal")

    df = df.rename(index={0: "id", 1: "original_label", 2: "features", 3: "classifier"})
    df.to_pickle(path)


for model_full_name in os.listdir("./saved_models/"):
    if model_full_name.startswith("."):
        continue
        
    model_full_name = model_full_name.split(".")[0]
    if "SPLIT_train_test_split_seed_" in model_full_name:
        continue    
        
    print(model_full_name)
    if os.path.isdir("./features_and_scores/{}/".format(model_full_name)):
        print("OK - Folder ./features_and_scores/{}/ alredy exist.".format(model_full_name))
        continue
        
    os.mkdir("./features_and_scores/{}/".format(model_full_name))
    os.mkdir("./features_and_scores/{}/features".format(model_full_name))
    print("OK - START ./features_and_scores/{}/".format(model_full_name))
    
    model, loaders = load_data(model_full_name)
    
    for loader_name in loaders:
        path = "./features_and_scores/{}/features/{}.pickle".format(model_full_name, loader_name)
        save_df(path, get_df(model, loaders[loader_name]))
        print(">>>>>> Saved:", path)
        torch.cuda.empty_cache()    
