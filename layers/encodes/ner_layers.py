import copy
import math

import torch
import torch.nn as nn
from torch import autograd


class CNNmodel(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_layer, dropout, gpu = True,device="cuda:0"):
        super(CNNmodel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.gpu = gpu

        self.cnn_layer0 = nn.Conv1d(self.input_dim,self.hidden_dim,kernel_size=1,padding=0)
        self.cnn_layers = [nn.Conv1d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=1)
                           for _ in range(self.num_layer - 1)]
        self.dropout = nn.Dropout(dropout)

        if self.gpu:
            self.cnn_layer0 = self.cnn_layer0.to(device)
            for i in range(self.num_layer -1):
                self.cnn_layers[i] =self.cnn_layers[i].to(device)

    def forward(self, input_feature):
        input_feature = input_feature.transpose(2,1).contiguous()
        cnn_output = self.cnn_layer0(input_feature)
        cnn_output = self.dropout(cnn_output)
        cnn_output = torch.tanh(cnn_output)

        for layer in range(self.num_layer - 1):
            cnn_output = self.cnn_layers[layer](cnn_output)
            cnn_output = self.dropout(cnn_output)
            cnn_output = torch.tanh(cnn_output)

        cnn_output = cnn_output.transpose(2,1).contiguous()
        return cnn_output


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):

     def __init__(self,h, d_model, dropout=0.1):
         super(MultiHeadedAttention, self).__init__()
         assert d_model % h ==0
         self.d_k = d_model // h
         self.h = h
         self.linears = clones(nn.Linear(d_model, d_model), 4)
         self.att = None
         self.dropout = nn.Dropout(p = dropout)
     def forward(self, query,key,value,mask=None):
         if mask is not None:
             mask = mask.unsqueeze(1)
         nbatches = query.size(0)
         query,key,value = \
         [l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
          for l,x in zip(self.linears,(query,key,value))]

         x,self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)

         x = x.transpose(1,2).continguous().view(nbatches,-1,self.h * self.d_k)

         return self.linears[-1](x)
class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1,keepdim = True)
        std = x.std(-1,keepdim = True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x , sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x:self.self_attn(x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

def attention(query, key, value, mask=None,dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/\
             math.sqrt(d_k)
    if mask:
        scores = scores.masked_fill(mask==0,-1e9)
    p_attn = torch.softmax(scores,dim=-1)
    if dropout:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value), p_attn


def forward(self,query,key,value,mask=None):

        if mask:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query,key,value = \
           [l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
            for l,x in zip(self.linears,(query,key,value))]

        x,self.attn = attention(query,key,value,mask=mask,
                                dropout=self.dropout)

        x = x.transpose(1,2).contiguous()\
            .view(nbatches,-1,self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(object):
    def __init__(self, d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.x2(self.dropout(torch.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.,d_model, 2)*
                             -(math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self, x):
        x = x + torch.tensor(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)

class AttentionModel(nn.Module):

    def __init__(self, d_input, d_model, d_ff, head, num_layer, dropout):
        super(AttentionModel, self).__init__()
        c = copy.deepcopy

        attn = MultiHeadedAttention(head, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        layer = EncoderLayer(d_model, c(attn),c(ff),dropout)
        self.layers = clones(layer, num_layer)

        self.norm = LayerNorm(layer.size)
        self.posi = PositionalEncoding(d_model, dropout)
        self.input2model = nn.Linear(d_input,d_model)

    def forward(self, x, mask):
        x = self.posi(self.input2model(x))
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)



class NERmodel(nn.Module):

    def __init__(self,model_type, input_dim,
                 hidden_dim,num_layer,
                 dropout=0.5,gpu=True,biflag = True,device="cuda:0"):
        super(NERmodel, self).__init__()
        self.model_type = model_type

        if self.model_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, hidden_dim,
                                num_layers=num_layer,batch_first=True,
                                bidirectional=biflag)
            self.drop = nn.Dropout(dropout)

        if self.model_type == "cnn":
            self.cnn = CNNmodel(input_dim, hidden_dim, num_layer,dropout,gpu,device)

        if self.model_type == "transformer":
            self.attention_model = AttentionModel(
                d_input = input_dim,d_model = hidden_dim,d_ff = 2 * hidden_dim,
                head = 4, num_layer = num_layer,dropout = dropout
            )
            for p in self.attention_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, input,mask=None):
        if self.model_type == 'lstm':
            hidden = None
            feature_out, hidden = self.lstm(input,hidden)
            feature_out_d = self.drop(feature_out)

        elif self.model_type == 'cnn':
            feature_out_d = self.cnn(input)

        elif self.model_type == 'transformer':
            feature_out_d = self.attention_model(input,mask)
        return feature_out_d
