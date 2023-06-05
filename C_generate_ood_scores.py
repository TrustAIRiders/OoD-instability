#!/usr/bin/env python

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import random

from models.resnet import ResNet, Bottleneck
from models.mobilenetv2 import MobileNetV2

from ood.max_softmax import MaxSoftmax
from ood.max_logits import MaxLogits
from ood.free_energy import FreeEnergy
from ood.lof import LOF
from ood.mahalanobis import Mahalanobis
from ood.knn import KNN
from ood.calculate_metrics import CalculateMetrics


np.random.seed(0)
random.seed(0)

def parse_ood_result(results):
    tnr = int(results['tnr_95'] * 10000) / 100
    auc = int(results['auc'] * 10000) / 100
    acc = int(results['max_bin_acc'] * 10000) / 100
    _aupr = (results['aupr_in'] + results['aupr_out']) / 2
    aupr = int(_aupr * 10000) / 100    
    
    return tnr, auc, acc, aupr


oods = [MaxSoftmax(), MaxLogits(), FreeEnergy(), KNN(), LOF(LOF.EUCLIDEAN), LOF(LOF.COSINE), Mahalanobis()]


for model_full_name in os.listdir("./features_and_scores/"):
    if model_full_name.startswith("."):
        continue
        
    model_full_name = model_full_name.split(".")[0]
        
    print(model_full_name)
    if not os.path.isdir("./features_and_scores/{}/scores/".format(model_full_name)):
        os.mkdir("./features_and_scores/{}/scores".format(model_full_name))
        os.mkdir("./features_and_scores/{}/results".format(model_full_name))
        
    print("START: {}".format(model_full_name))
    
    df_train = pd.read_pickle("./features_and_scores/{}/features/train.pickle".format(model_full_name))
    df_test = pd.read_pickle("./features_and_scores/{}/features/test.pickle".format(model_full_name))
    for ood in oods:
        if os.path.isdir("./features_and_scores/{}/scores/{}/".format(model_full_name, ood.name)):
            print("OK ----- {}".format(ood.name))
            continue
            
        print("START ----- {}".format(ood.name))
        os.mkdir("./features_and_scores/{}/scores/{}/".format(model_full_name, ood.name))
        os.mkdir("./features_and_scores/{}/results/{}/".format(model_full_name, ood.name))
        
        ood.clear()
        ood.fit(df_train)    
        ood.known_out = ood.test(df_test)
        
        score_path = "./features_and_scores/{}/scores/{}/{}.npy".format(model_full_name, ood.name, "test")  
        np.save(score_path, ood.known_out)      
        
        for pickle_file in os.listdir("./features_and_scores/{}/features/".format(model_full_name)):
            if "ood_" not in pickle_file:
                continue
                
            df_ood = pd.read_pickle("./features_and_scores/{}/features/{}".format(model_full_name, pickle_file))
            
            ood.unknown_out = ood.test(df_ood)
            score_path = "./features_and_scores/{}/scores/{}/{}.npy".format(model_full_name, ood.name, pickle_file.split(".")[0])  
            np.save(score_path, ood.unknown_out)      
            
            ood.unknown_out = ood.unknown_out[:len(ood.known_out)]
            results = CalculateMetrics().run(ood)
            tnr, auc, acc, aupr = parse_ood_result(results)
            row = {
                'model': [model_full_name], 'ood_method': [ood.name], 'unknown_dataset': [pickle_file.split(".")[0]],
                'tnr_95': [tnr], 'auc': [auc], 'acc': [acc], 'aupr': [aupr]
            }
            print("\t\t", pickle_file, {'tnr_95': [tnr], 'auc': [auc], 'acc': [acc], 'aupr': [aupr]})

            df_results = pd.DataFrame.from_dict(row)
            df_results.to_csv("./features_and_scores/{}/results/{}/{}.csv".format(model_full_name, ood.name, pickle_file.split(".")[0]))
