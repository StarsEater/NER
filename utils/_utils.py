import os

import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score,recall_score,precision_score,f1_score

from tools import readJson

NULLKEY = "-null-"
#################     common    ######################
from transformers import BertTokenizer


def readVectors(path,topn):
    """
     读取 topn个词向量
    :param path:
    :param n:
    :return:
    """
    lines_num,dim = 0,0
    vectors,iw,wi = {},{},{}
    with open(path,encoding='utf-8') as f:
        tokens = f.readlines()
        dim =  int(tokens[0].rstrip().split()[1])
        for i in range(1,len(tokens)):
            token = tokens[i].rstrip().split(' ')
            if len(token[0])==1:
                lines_num += 1
                vectors[token[0]] = np.asarray([float(x) for x in token[1:]])
                iw.append(token[0])
            if topn !=0 and lines_num >= topn:
                break
    unk = np.random.rand(dim)
    unk = (unk - 0.5)/100
    vectors['unk'] = unk

    wi = {w: i for i, w in enumerate(iw)}
    print("word vector lens(including unk) is %d"% (len(vectors)))
    return vectors,iw,wi,dim

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word +='0'
        else:
            new_word +=char.lower()
    return new_word

def read_instance(input_file, word_alphabet, biword_alphabet, label_alphabet, number_normalized,
                  max_sent_length, bertpath):
    tokenizer = BertTokenizer.from_pretrained(bertpath, do_lower_case=True)
    instance_texts, instance_Ids = [], []
    for data in readJson(input_file,lines=True):
        # text
        if number_normalized:
            chars_text = [normalize_word(x) for x in data["ins"]]
        else:
            chars_text = data['ins']

        biword_text = list(zip(chars_text,chars_text[1:]+[NULLKEY]))
        biword_text = [''.join(x) for x in biword_text]
        labels = data['BMES']
        instance_texts.append([chars_text,biword_text,labels])

        # ids
        chars_ids = [word_alphabet.get_index(x) for x in chars_text][:max_sent_length]
        biword_ids = [biword_alphabet.get_index(x) for x in biword_text][:max_sent_length]
        bert_ids = tokenizer.convert_tokens_to_ids(
            ['[CLS]']+chars_text[:]+['[SEP]']
        )
        labels_id = [label_alphabet.get_index(x) for x in labels][:max_sent_length]
        instance_Ids.append([chars_ids,biword_ids,labels_id,bert_ids])
    return instance_texts,instance_Ids


def plot_loss(history, save_root, model_name,time_stamp=0,save_mode=True):
    plt.figure()
    plt.plot(history['train_loss'], 'r', history['val_loss'], 'b')
    plt.legend(["train_loss", "val_loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss during training")
    if save_mode:
        plt.savefig(os.path.join(save_root,'loss_'+model_name+'_'+str(time_stamp)+'.jpg'))
    plt.close()

def plot_acc_score(history, save_root,model_name, time_stamp=0,save_mode=True):
    plt.figure()
    plt.plot(history['train_acc'], 'r', history['val_acc'], 'b')
    plt.legend(["train_acc_score", "val_acc_score"])
    plt.xlabel("epoch")
    plt.ylabel("acc_score")
    plt.title("acc_score during training")
    if save_mode:
        plt.savefig(os.path.join(save_root , 'acc_score_'+model_name+'_'+str(time_stamp)+'.jpg'))
    plt.close()

def plot_f1_score(history, save_root, model_name,time_stamp=0,save_mode=True):
    plt.figure()
    plt.plot(history['train_f1'], 'r', history['val_f1'], 'b')
    plt.legend(["train_f1_score", "val_f1_score"])
    plt.xlabel("epoch")
    plt.ylabel("f1_score")
    plt.title("f1_score during training")
    if save_mode:
        plt.savefig(os.path.join(save_root , 'f1_score_'+model_name+'_'+str(time_stamp)+'.jpg'))
    plt.close()




