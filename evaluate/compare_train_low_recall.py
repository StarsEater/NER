import os

from prettyprinter import cpprint

from tools import readText

gold_dir = "/nlp_data/qinye/raw_data_base/ann_process"
pre_dir  = "/nlp_data/qinye/ner_task/pred"
pre_names = set()
for file in os.listdir(pre_dir):
    if file.startswith("qy_lung_pos"):
        pre_names.add(file.split(".")[0])
wrong_ans = {}
for file in pre_names:
    txt = os.path.join(gold_dir,file+".txt")
    gold_ann = os.path.join(gold_dir,file+".ann")
    pre_ann  = os.path.join(pre_dir,file+".ann")
    gold_ann_lst = set(['\t'.join(x.split("\t")[1:]) for x in readText(gold_ann,lines=True)])
    pre_ann_lst = set(['\t'.join(x.split("\t")[1:]) for x in readText(pre_ann,lines=True)])
    no_in_gold = pre_ann_lst - gold_ann_lst
    no_in_pre  = gold_ann_lst - pre_ann_lst
    if len(no_in_pre) > 0 and len(no_in_gold) > 0:
        print(txt)
        print("no_in_gold",no_in_gold)
        print("no_in_pre",no_in_pre)

