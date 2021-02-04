import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

from layers.encodes.ner_layers import NERmodel


class multi_feature_model(nn.Module):
    def __init__(self,data,
                 use_gpu = True,
                 device="cpu",use_bigram=True,
                 hidden_dim=0,word_emb_dim=0,biword_emb_dim=0,
                 bilstm=True,lstm_layer=1,num_layer=1,model_type="lstm",
                 use_bert=True,dropout=0.5,
                 bert1_path=""
                 ):
        super(multi_feature_model, self).__init__()

        self.use_biword = use_bigram
        self.hidden_dim = hidden_dim
        self.word_emb_dim = word_emb_dim
        self.biword_emb_dim = biword_emb_dim
        self.bilstm_flag = bilstm
        self.lstm_layer = lstm_layer
        self.num_layer = num_layer
        self.model_type = model_type
        self.use_bert = use_bert
        self.device = device

        self.word_embedding = nn.Embedding(data.word_alphabet.size(),
                                           self.word_emb_dim,padding_idx=0)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(data.pretrain_word_embedding)
            )

        if self.use_biword:
            self.biword_embdding = nn.Embedding(data.biword_alphabet.size(),
                                                self.biword_emb_dim,padding_idx=0)
            if data.pretrain_biword_embedding is not None:
                self.biword_embdding.weight.data.copy_(
                    torch.from_numpy(data.pretrain_biword_embedding)
                )
        char_feature_dim = self.word_emb_dim
        if self.use_biword:
            char_feature_dim += self.biword_emb_dim

        if self.use_bert:
            char_feature_dim = char_feature_dim + 768
        print("data tags number is {}".format(data.label_alphabet_size))
        print('total char feature_dim is {0} , {1}+{2}+{3}'.
              format(char_feature_dim,self.word_emb_dim,self.biword_emb_dim,768))

        if self.model_type == 'lstm':
            lstm_hidden = self.hidden_dim
            if self.bilstm_flag:
                self.hidden_dim *= 2
            self.NERmodel = NERmodel(model_type='lstm',input_dim=char_feature_dim,
                                     hidden_dim=lstm_hidden,num_layer=self.lstm_layer,biflag=self.bilstm_flag)

            self.hidden2tag = nn.Linear(self.hidden_dim, data.label_alphabet_size) #TODO

        if self.model_type == 'transformer':
            self.NERmodel = NERmodel(model_type='transformer',input_dim=char_feature_dim,
                                     hidden_dim=self.hidden_dim,num_layer=self.num_layer,dropout=dropout)
            self.hidden2tag = nn.Linear(480, data.label_alphabet_size ) # TODO

        self.drop = nn.Dropout(dropout)
        self.crf = CRF(data.label_alphabet_size,batch_first=True)
        if self.use_bert:
            self.bert_encoder_1 = BertModel.from_pretrained(bert1_path)
            for p in self.bert_encoder_1.parameters():
                p.requires_grad = False

        if use_gpu:
            self.word_embedding = self.word_embedding.to(self.device)
            if self.use_biword:
                self.biword_embdding = self.biword_embdding.to(self.device)
            self.NERmodel = self.NERmodel.to(self.device)
            self.hidden2tag = self.hidden2tag.to(self.device)
            self.crf = self.crf.to(device)
            if self.use_bert:
                self.bert_encoder_1 = self.bert_encoder_1.to(self.device)
    def get_tags(self, word_inputs, biword_inputs,mask, batch_bert,bert_mask):
        word_embs = self.word_embedding(word_inputs)

        if self.use_biword:
            biword_embs = self.biword_embdding(biword_inputs)
            word_embs = torch.cat([word_embs, biword_embs], dim = -1)

        if self.model_type != "transformer":
            word_inputs_d = self.drop(word_embs)
        else:
            word_inputs_d = word_embs

        word_input_cat = torch.cat([word_inputs_d],dim=-1) # todo 干啥

        if self.use_bert:
            seg_id = torch.zeros(bert_mask.size()).long().to(self.device)

            outputs_1 = self.bert_encoder_1(batch_bert, bert_mask, seg_id)
            outputs_1 = outputs_1[0][:,1:-1,:]


            word_input_cat = torch.cat([word_input_cat, outputs_1], dim = -1)

        feature_out_id = self.NERmodel(word_input_cat, word_inputs.ne(0))
        tags = self.hidden2tag(feature_out_id)

        return tags

    def neg_log_likelihood_loss(self, word_inputs, biword_inputs,mask,batch_label,
                                batch_bert,bert_mask):
        tags = self.get_tags(word_inputs, biword_inputs, mask, batch_bert,bert_mask)
        # print(tags.shape,batch_label.shape)
        total_loss = -self.crf(tags,batch_label,mask)
        tag_seq = self.crf.decode(tags,mask)
        return total_loss, tag_seq

    def forward(self, word_inputs, biword_inputs, mask,batch_bert,bert_mask):
        tags = self.get_tags(word_inputs,biword_inputs,mask,batch_bert,bert_mask)
        tag_seq = self.crf.decode(tags,mask)
        return tag_seq
