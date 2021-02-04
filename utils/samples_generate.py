import logging
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import time
import re
import collections
import pandas as pd
from configparser import ConfigParser
from tools import *

def transfer_from_dir(text_ann_dir,save_json_path,test=False):
    """
    :param text_ann_dir: 存放txt和ann标注文件的路径
    :return: 该路径下的文件名集合
    """
    assert os.path.isdir(text_ann_dir),print(text_ann_dir)
    names = []
    names_list = [x.split(".")[0] for x in os.listdir(text_ann_dir) if x.endswith("txt") or x.endswith("ann")]

    for k,v in collections.Counter(names_list).items():
        if v==2:
            names.append(k)
        else:
            print("no match file for {}".format(k))
    res = []
    names = sorted(names)[:400]
    for name in names:
        txt_from = readText(os.path.join(text_ann_dir,name+".txt"),lines=False)
        if not test:
            ann_from = readText(os.path.join(text_ann_dir,name+".ann"),lines=True)
        else:
            ann_from = []
        entitys = list(filter(lambda x:x[0]=="T",ann_from))
        # rels = list(filter(lambda x:x[0]=="R",ann_from))
        #check valid
        txt_pair_type = [[x,"O"] for x in txt_from]
        for i,entity in enumerate(entitys):
            entity = entity.split("\t")
            type_pos,word = entity[1:]
            en_type,start,end = type_pos.split()
            start,end = int(start),int(end)
            assert txt_from[start:end]==word,\
                print("{0} don't match {1}".format(word,txt_from[start:end]))
            for j in range(start,end):
                txt_pair_type[j][-1]=en_type+"-"+str(i)
        if not test:
            txt_pair_type = [(re.sub(r"\t|\n|\r"," ",x[0]),x[1]) for x in txt_pair_type]
        ins,outs = list(zip(*txt_pair_type))

        from2tos = []
        if outs[0]!='O':
            from2tos.append([0,0,outs[0].split("-")[0],ins[0]])

        for i in  range(1,len(outs)):
            v = outs[i]
            if v!=outs[i-1] and v!="O":
                from2tos.append([i,i,v.split("-")[0],ins[i]])
            elif v==outs[i-1] and v!="O":
                from2tos[-1][1]= i
                from2tos[-1][3] = ''.join(ins[from2tos[-1][0]:from2tos[-1][1]+1])


        #BIO
        BIO = ['O']*len(outs)
        for start,end,label_name,_ in from2tos:
            BIO[start] = 'B-'+label_name
            for i in range(start+1,end+1):
                BIO[i] = 'I-'+label_name

        # BMES
        BMES = ['O']*len(outs)
        for start,end,label_name,_ in from2tos:
            if start==end:
                BMES[start] = "S-"+label_name
                continue
            BMES[start] = 'B-'+label_name
            for i in range(start+1,end):
                BMES[i] = "M-"+label_name
            BMES[end] = "E-"+label_name

        res.append({
            "tokens":txt_pair_type,
            "ins":ins,
            "outs":outs,
            'from2tos':from2tos,
            'BIO':BIO,
            'BMES':BMES
        })
    print("total data num is {}".format(len(res)))
    logging.info("total data num is {}".format(len(res)))
    if not test:
        checkFileOMake(save_json_path)
        save_json_path = os.path.join(save_json_path,"text_ann.json")
        saveJson(res,save_json_path,lines=True)
        return
    saveJson(res, save_json_path,lines=True)
    return names,res





if __name__ == '__main__':
    config = ConfigParser()
    config_path = "../data/dev/multi_feature.conf"
    # config.read(config_path,encoding='utf-8')
    config.read(config_path,encoding='utf-8')
    # print(config.sections())
    ann_dir = config['samples_generate']["ann_dir"]
    sample_save_path = config["samples_generate"]["sample_save_path"]
    log_path = config["samples_generate"]["log"]

    checkFileOMake(log_path)

    logging.basicConfig(level=logging.DEBUG,
                        filemode='a')
    logging.info("NER samples generating !")
    localtime = time.asctime(time.localtime(time.time()))
    logging.info("### start time : %s"%(localtime))
    time_stamp = int(time.time())
    logging.info("time stamp: %d"%(time_stamp))



    # print("原始数据目录: %s" % (ann_dir))
    # print("保存数据路径: %s"%(sample_save_path))
    logging.info("原始数据路径: %s"%(sample_save_path))
    logging.info("保存数据路径: %s"%(sample_save_path))
    transfer_from_dir(ann_dir, sample_save_path)





