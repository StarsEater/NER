import os
from configparser import ConfigParser
from random import shuffle
import random
import sys
import collections
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import logging
from tools import *
random.seed(2200)

def dataset_merge_shuffle(file_path,save_path,split_ratio=[0.9,0.1,0.1]):
    total_dataset = readJson(file_path,lines=True)
    shuffle(total_dataset)

    r1,r2,r3 = split_ratio
    assert r1+r2+r3==1

    n = len(total_dataset)
    trainset = total_dataset[:round(r1*n)]
    devset = total_dataset[round(r1*n):round((r1+r2)*n)]
    testset = total_dataset[round((r1+r2)*n):]

    train_path = os.path.join(save_path, "trainset")
    dev_path = os.path.join(save_path, "devset")
    test_path =os.path.join(save_path, "testset")

    saveJson(trainset,train_path,lines=True)
    saveJson(devset,dev_path,lines=True)
    saveJson(testset,test_path,lines=True)
    print("save it ",train_path)

if __name__ == '__main__':
    config = ConfigParser()
    config_path = "../data/dev/multi_feature.conf"
    config.read(config_path, encoding='utf-8')

    ner_file_path = os.path.join(config["samples_generate"]["sample_save_path"],"text_ann.json")
    ner_save_path = config['dataset_split']['dataset_save_path']
    split_ratio = config['dataset_split']['split_ratio'].split(',')
    split_ratio = [float(item) for item in split_ratio]
    checkFileOMake(ner_save_path)
    dataset_merge_shuffle(ner_file_path,ner_save_path,split_ratio)