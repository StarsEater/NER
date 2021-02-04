import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader

from tools import loadPickle


class multi_feature_dataset(Dataset):
    def __init__(self,data,mode="train"):
        assert mode in ["train","dev","test"]
        super(multi_feature_dataset, self).__init__()
        raw_data = eval("data."+mode+"_Ids")
        self.process_data = []
        for d in raw_data:
            self.process_data.append(self.convert_to_multi_feature_format(d))
        # [chars_ids,biword_ids,labels_id,bert_ids]

    def __getitem__(self, item):
        return self.process_data[item]

    def __len__(self):
        return len(self.process_data)
    def convert_to_multi_feature_format(self,id_lst):
        chars_ids,biword_ids,labels_id,bert_ids = list(map(lambda x:torch.LongTensor(x),id_lst))
        bert_mask = torch.ByteTensor([1]*len(bert_ids))
        label_mask = torch.ByteTensor([1]*len(chars_ids))
        return chars_ids,biword_ids,bert_ids,bert_mask,labels_id,label_mask

    def _collate_fn(self,batch):
        batch = list(zip(*batch))
        return [pad_sequence(x,batch_first=True,padding_value=0) for x in batch]

    def get_dataloader(self,batch_size,num_workers=0,shuffle=False,pin_memory=False):
        return DataLoader(self,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,
                          collate_fn=self._collate_fn,pin_memory=pin_memory)



