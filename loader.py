# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class MyDataset:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
        self.sentences = []
        self.attribute_schema = json.load(open(config["schema_path"], encoding="utf-8"))
        self.config["num_labels"] = len(self.attribute_schema)
        self.max_length = config["max_length"]
        self.load()
        print("超出设定最大长度的样本数量:%d, 占比:%.3f"%(self.exceed_max_length, self.exceed_max_length/len(self.data)))
        print("由于文本截断，导致实体缺失的样本数量:%d, 占比%.3f"%(self.entity_disapper, self.entity_disapper/len(self.data)))

    def load(self):
        self.text_data = []
        self.data = []
        self.exceed_max_length = 0
        self.entity_disapper = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                context = sample["context"]
                object = sample["object"]
                attribute = sample["attribute"]
                value = sample["value"]

                if object == "" or value == "":
                    continue
                
                if attribute not in self.attribute_schema:
                    attribute = "UNRELATED"

                try:
                    input_id, e1_mask, e2_mask, label = self.process_sentence(context, object, attribute, value)
                except IndexError:
                    self.entity_disapper += 1
                    continue 
                self.data.append([torch.LongTensor(input_id),
                                  torch.LongTensor(e1_mask),
                                  torch.LongTensor(e2_mask),
                                  torch.LongTensor([label])])
        return

    def process_sentence(self, context, object, attribute, value):
        """
        这个函数的作用是对句子进行处理
        param: 
        context 上下文 
        object 头实体
        attribute 属性
        value 尾实体
        return:
        input_id 对context进行bert encode 出来的序列
        e1_mask  头实体的mask,即头实体在context的位置标为1
        e2_mask  尾实体的mask,即尾实体在context的位置标为1
        attribute_label 属性标签
        """
        if len(context) > self.max_length:   # 如果context长度超过最大长度，则截断
            self.exceed_max_length += 1
        object_start = context.index(object) + 1  # 因为bert的分词会在第一位增加[cls]，所以向后移动一位
        value_start = context.index(value) + 1    # 同上

        input_id = self.tokenizer.encode(context, max_length=self.max_length, pad_to_max_length=True)
        attribute_label = self.attribute_schema[attribute] # 关系标签
        
        # 标记头实体
        e1_mask = [0] * len(input_id)
        for index in range(object_start, object_start + len(object)):
            e1_mask[index] = 1
        
        # 标记尾实体
        e2_mask = [0] * len(input_id)
        for index in range(value_start, value_start + len(value)):
            e2_mask[index] = 1

        return input_id, e1_mask, e2_mask, attribute_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)



# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dataset = MyDataset(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    from config import Config
    dg = MyDataset("./triplet_data/triplet_data.json", Config)



