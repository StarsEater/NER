import numpy as np
from prettyprinter import cpprint

from tools import saveText


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path !=None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale  = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0

    for word,index in word_alphabet.instance2index.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale,scale,[1,embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / word_alphabet.size()))

    return pretrain_emb, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/ root_sum_square



def load_pretrain_emb(embedding_path):
    """
    每一行，第一个为词，之后是词向量，都是以空格隔开
    :param embedding_path:
    :return:
    """
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line)==0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert  (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1,embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

def filerchar_bichar_from_word2vec(path,char_path,bichar_path):
    chars = []
    bichars = []
    first_line = []
    with open(path, "r", encoding='utf-8') as f:
        for rw in f.readlines():
            w = rw.strip().split(" ")
            if len(w) == 2:
                first_line = w
                continue
            if len(w[0]) == 1:
                assert len(w)==int(first_line[-1])+1,print(w[0],w[1],w[2],rw)
                chars.append(rw.rstrip())
            elif len(w[0])==2:
                assert len(w) == int(first_line[-1])+1,print(rw,w[0])
                bichars.append(rw.rstrip())
    clen,bclen= len(chars),len(bichars)
    print("char num {} bichar num {}".format(clen,bclen))
    # chars = [' '.join([str(clen),first_line[-1]])] + chars
    # bichars = [' '.join([str(bclen), first_line[-1]])] + bichars
    saveText(chars,char_path,lines=True)
    saveText(bichars,bichar_path,lines=True)

if __name__ == '__main__':
    pass
    word_path = "/nlp_data/qinye/pretrains/word2vec/sgns.baidubaike.bigram-char"
    char_path =  "/nlp_data/qinye/pretrains/word2vec/sgns.baidubaike.char"
    bichar_path = "/nlp_data/qinye/pretrains/word2vec/sgns.baidubaike.bichar"
    filerchar_bichar_from_word2vec(word_path,char_path,bichar_path)