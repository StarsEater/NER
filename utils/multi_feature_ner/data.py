import sys

from tqdm import tqdm

from tools import readJson
from utils._utils import normalize_word, read_instance
from utils.multi_feature_ner.alphabet import Alphabet
from utils.multi_feature_ner.functions import build_pretrain_embedding
NULLKEY = "-null-"

class Data:
    def __init__(self,min_freq=1,
                      bertpath='',
                      MAX_SENTENCE_LENGTH=1000,
                      word_emb_dim=0,
                      biword_emb_dim=0,
                      number_normalized=True,
                      norm_word_emb =True
                 ):
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
        self.word_emb_dim = word_emb_dim
        self.biword_emb_dim    = biword_emb_dim
        self.number_normalized = number_normalized
        self.norm_word_emb = norm_word_emb
        self.norm_biword_emb = norm_word_emb
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword', min_freq=min_freq)
        self.label_alphabet = Alphabet('label',True)

        self.biword_count = {}

        self.tagScheme = "NoSeg"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.train_split_index = []
        self.dev_split_index = []

        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.label_alphabet_size = 0

        self.bertpath = bertpath

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line) > 2:
                    pairs = line.strip().split()
                    label = pairs[-1]
                    self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False

        for labe,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True

        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

        self.fix_alphabet()
        print("Refresh label alphabet finished: old: %s -> new:%s " % (old_size, self.label_alphabet_size))

    def build_alphabet(self, input_file):

        for data in readJson(input_file,lines=True):
            # text
            chars_text = [normalize_word(x) for x in data["ins"]]
            biword_text = list(zip(chars_text, chars_text[1:] + [NULLKEY]))
            biword_text = [''.join(x) for x in biword_text]
            labels = data['BMES']

            for char in chars_text:
                self.word_alphabet.add(char.lower())
            for bi in biword_text:
                self.biword_alphabet.add(bi.lower())
                self.biword_count[bi] = self.biword_count.get(bi,0) + 1
            for la in labels:
                self.label_alphabet.add(la)
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.label_alphabet.close()
        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()

    def build_word_pretrain_emb(self, emb_path):
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = \
            build_pretrain_embedding(emb_path,self.word_alphabet,
                                     self.word_emb_dim,self.norm_word_emb)
    def build_biword_pretrain_emb(self, emb_path):
        print("build biword pretrain emb")
        self.pretrain_biword_embedding,self.biword_emb_dim = \
            build_pretrain_embedding(emb_path,self.biword_alphabet,
                                     self.biword_emb_dim,self.norm_biword_emb)
    def write_decoded_results(self, output_file, predict_result, name):
        with open(output_file,'w') as fout:
            sent_num = len(predict_result)
            content_list = []
            if name == "raw":
                content_list = self.raw_texts
            elif name == "test":
                content_list = self.test_texts
            elif name == "dev":
                content_list = self.dev_texts
            elif name == "train":
                content_list = self.train_texts
            else:
                print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
            assert sent_num == len(content_list)
            for idx in range(sent_num):
                sent_length = len(predict_result[idx])
                for idy in  range(sent_length):
                    fout.write(content_list[idx][0][idy].encode("utf-8") + " " + predict_result[idx][idy] + "\n")
            fout.write("\n")
        print("Predict %s result has been written into file. %s" % (name, output_file))
    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        assert name in ["train","dev","test","raw"]
        texts, Ids = read_instance(
            input_file,self.word_alphabet,self.biword_alphabet,
            self.label_alphabet,self.number_normalized,self.MAX_SENTENCE_LENGTH,
            self.bertpath
        )
        if name == "train":
            self.train_texts,self.train_Ids = texts,Ids
        elif name == "dev":
            self.dev_texts,self.dev_Ids = texts,Ids
        elif name == "test":
            self.test_texts,self.test_Ids = texts,Ids
        elif name == "raw":
            self.raw_texts,self.raw_Ids = texts,Ids




