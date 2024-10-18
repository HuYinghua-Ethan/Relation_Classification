# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

"""
复现论文Relation Bert
建立网络模型结构

length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)
1. e_mask != 0:
含义: 这个表达式创建了一个布尔张量，指示 e_mask 中哪些元素不等于 0。
示例: 假设 e_mask 的内容是：
e_mask = [[1, 2, 0, 0],
          [1, 0, 3, 4],
          [0, 0, 0, 0]]
则 e_mask != 0 的结果将是：
[[ True,  True, False, False],
 [ True, False,  True,  True],
 [False, False, False, False]]

2. .sum(dim=1)
含义: 在布尔张量上进行求和操作，dim=1 表示对第二个维度（即序列长度维度）进行求和。这将返回每一行 True 值的数量。
示例: 对上面的布尔张量进行求和，得到：
[2, 3, 0]
这表示第一个样本有 2 个有效元素，第二个样本有 3 个有效元素，第三个样本没有有效元素。
3. .unsqueeze(1)
含义: 在求和的结果上增加一个维度。unsqueeze(1) 会将形状从 [batch_size] 变为 [batch_size, 1]，即在第二个维度（索引为1的位置）增加一个维度。
示例: 对于 [2, 3, 0]，unsqueeze(1) 的结果是：
[[2],
 [3],
 [0]]
最终结果
length_tensor: 最终的 length_tensor 是一个形状为 [batch_size, 1] 的张量，每一行代表对应样本的有效长度（非零元素的数量）。
总结
这段代码的作用是：
计算 e_mask 中每个样本的有效元素个数，并将结果整理为一个二维张量，使其形状适合后续的计算或模型输入。

示例代码
以下是一个简单的完整示例：
import torch

# 假设 e_mask 是如下的张量
e_mask = torch.tensor([[1, 2, 0, 0],  # 第一个样本，有效长度为 2
                       [1, 0, 3, 4],  # 第二个样本，有效长度为 3
                       [0, 0, 0, 0]]) # 第三个样本，有效长度为 0

length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)
print(length_tensor)  # 输出: tensor([[2], [3], [0]])
在这个例子中，length_tensor 的输出 [2], [3], [0] 表示每个样本中有效元素的数量。

"""


class TorchModel(nn.Module):
    def __init__(self, config):
        self.config = config
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        self.hidden_size = self.bert.config.hidden_size  # 768
        self.cls_fc_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.entity_fc_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.num_labels = self.config["num_labels"]  # 最终的分类数量
        self.label_classifier = nn.Linear(self.hidden_size * 3, self.num_labels)
        self.activation = torch.tanh
        self.dropout = nn.Dropout(0.5)

    # entity mask 形如:   [0, 0, 1, 1, 0, 0, ...]
    def entity_average(self, hidden_output, e_mask):
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [batch_size, 1, sentence_length]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1) # 将实体对应的向量提取出来
        avg_vector = sum_vector.float() / length_tensor.float()  # 除以实体词长度，求平均
        return avg_vector


    def forward(self, input_ids, e1_mask, e2_mask, labels=None):
        outputs = self.bert(input_ids)
        sequence_output = outputs[0]  # shape: (batch_size, sequence_length, hidden_size)
        pooled_output = outputs[1]    # [CLS]  shape: (batch_size, hidden_size)

        # 实体向量求平均
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # dropout
        e1_h = self.dropout(e1_h)
        e2_h = self.dropout(e2_h)
        pooled_output = self.dropout(pooled_output)

        #过线性层并激活
        pooled_output = self.activation(self.cls_fc_layer(pooled_output))
        e1_h = self.activation(self.entity_fc_layer(e1_h))
        e2_h = self.activation(self.entity_fc_layer(e2_h))

        # 拼接向量
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)



if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
