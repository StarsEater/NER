import json
import os,sys
import pickle
from functools import reduce

import jsonlines
import pandas as pd
import torch
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)


def readJson(path,lines=False):
    """
    （判断是否）逐行从json文件中读取数据
    :param path:
    :param lines:
    :return:
    """
    if not os.path.exists(path):
        print("no such file ,check the path %s"%path)
    data = []
    if lines:
        with open(path, "r", encoding='utf-8') as f:
            for l in f.readlines():
                data.append(json.loads(l))

    else:
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)

    return data

def saveJson(data,path,lines=False):
    """
    （判断是否）逐行保存数据到json文件
    :param data:
    :param path:
    :param lines:
    :return:
    """
    if lines:
        with open(path,"w",encoding="utf-8") as f:
            for d in data:
                json.dump(d,f,ensure_ascii=False)
                f.write("\n")
    else:
        with open(path,"w",encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False)
def checkFileOMake(path):
    """
    检查文件是否存在，不存在创建文件
    :param path:
    :return:
    """
    if len(path) < 2:
        return
    if not os.path.exists(path):
        os.makedirs(path)

def saveJsonL(data,path):
    """
    写入jsonl文件
    :param data:
    :param path:
    :return:
    """
    jsons = json.loads(data)
    with jsonlines.open(path,mode='a') as f:
        f.write(jsons)

def readJsonL(path):
    """
    读jsonlines文件
    :param path:
    :return:
    """
    if not os.path.exists(path):
        print("no such file ,check the path %s"%path)
    data = []
    with open(path,"r+",encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            data.append(item.strip())
    return data

def dict_list2csv(dicts_list,path,columns=None):
    """
    :param dicts_list:  {'a':[1,2],'b':[2,3]}/[(1,2,3,4),(2,3,4,5)]
    :param path:
    :param columns(optional): 在list保存的情况下的列名 e.g [“v1”,"v2","v3","v4"]
    :return:
    a   b
    1   2
    2   3
    """
    if isinstance(dicts_list,dict):
        pd.DataFrame(dicts_list).to_csv(path,columns=dicts_list.keys(),index=None,encoding="utf_8_sig")
    elif isinstance(dicts_list,list):
        assert columns is not None,print("列名不能为空")
        pd.DataFrame(dicts_list).to_csv(path,columns=columns,index=None,encoding="utf_8_sig")
    else:
        print("数据格式错误")

def pd_read_csv(path,columns=None):
    data = pd.read_csv(path)
    data.fillna("",inplace=True)
    if columns:
        data = list(reduce(lambda x,y:[list(data[x])]+[list(data[y])],columns))
        data = list(zip(*data))
    return data

def convertModelInFormat(choice="Bert",
                         text=None,
                         label=None,
                         one_label=True,
                         label2num={},
                         tokenizer=None):
    if choice=="Bert":
        text = ['[CLS]'] + tokenizer.tokenize(text)[:500] + ['[SEP]']
        text_index = torch.LongTensor(tokenizer.convert_tokens_to_ids(text))
        mask_index = torch.ByteTensor([1] * len(text_index))
        label = [label2num[ll] for ll in label]
        if one_label:
            label_index = torch.LongTensor(list(label)[0:1])
        else:
            label_index = torch.LongTensor([1 if i in label else 0 for i in range(len(label2num))])
    return text_index,mask_index,label_index

def readText(path,lines=False):
    assert os.path.exists(path),print("没有该文件{}".format(path))
    with open(path,"r",encoding="utf-8",newline="") as f:
        if lines:
            res = []
            for line in f.readlines():
                res.append(line.rstrip())
            return res
        else:
            return f.read().rstrip()
def saveText(data,save_path,lines=False):
    with open(save_path,"w",encoding='utf-8') as f:
        if lines:
            for d in data:
                f.write(d.rstrip())
                f.write("\n")
        else:
            f.write(data)
def savePickle(data,path):
    with open(path,"wb") as f:
        pickle.dump(data,f)
def loadPickle(path):
    with open(path,"rb") as f:
        data = pickle.load(f)
    return data
##########      mysql         ####################

if __name__ == '__main__':
    h = pd_read_csv("./data/raws/Pathology.csv",["id","stand_sign"])
    print(h[:3])