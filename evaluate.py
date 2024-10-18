# -*- coding: utf-8 -*-

import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data
from sklearn.metrics import classification_report

"""
模型效果测试

gold += labels.detach().tolist()
labels:
labels 是一个张量（tensor），通常是模型的目标输出或真实标签。在分类任务中，labels 可能包含样本的类标签；在回归任务中，labels 可能包含连续值。
detach():
detach() 是 PyTorch 中的一个方法，调用此方法后返回一个新的张量，它与原始张量共享存储，但不再记录梯度。这意味着在后续的计算中，这个张量不会被用于计算梯度（不会影响反向传播）。通常在需要将张量从计算图中移除时使用，尤其是当你只想保留数据而不需要进行梯度更新时。
tolist():
tolist() 是 PyTorch 张量的方法，将张量转换为一个 Python 列表。这样可以更方便地进行后续操作，例如存储或处理数据。
gold += ...:
gold 是一个 Python 列表，通常用于存储标签或预测结果。+= 操作符用于将右侧的列表元素添加到 gold 列表中。此操作会将 tolist() 返回的列表中的元素逐个添加到 gold 列表中。
下面是一个简单的例子，展示如何在训练过程中使用这段代码：
import torch

# 假设这是一个标签张量
labels = torch.tensor([1, 0, 1, 1, 0])  # 真实标签

# 创建一个空的 gold 列表
gold = []

# 将 labels 添加到 gold 列表中
gold += labels.detach().tolist()

print(gold)  # 输出: [1, 0, 1, 1, 0]

应用场景
模型评估: 在训练过程中，可能会使用 gold 列表来收集真实标签，然后在模型训练结束后进行评估。
监控性能: 可以用于实时监控模型的性能，通过比较 gold 列表与模型预测结果的差异来计算准确率或其他指标。


torch.argmax(batch_pred, dim=-1)
torch.argmax: 这个函数用于返回指定维度上最大值的索引。在分类任务中，输出通常是一个对每个类的得分，使用 argmax 可以找到得分最高的类。
dim=-1: 表示在最后一个维度上进行操作。对于二维张量（如 (batch_size, num_classes)），dim=-1 等同于 dim=1，这意味着函数将在每一行中寻找最大值的索引。
假设我们有一个模型输出如下：

import torch

# 假设这是模型的输出
batch_pred = torch.tensor([[2.0, 1.0, 0.1], 
                            [0.5, 2.5, 1.5], 
                            [1.0, 0.5, 1.0]])

# 使用 argmax 找到每行的最大值索引
predicted_classes = torch.argmax(batch_pred, dim=-1)

print(predicted_classes)

给定上述的 batch_pred：

第一行 [2.0, 1.0, 0.1] 的最大值为 2.0，对应的索引是 0。
第二行 [0.5, 2.5, 1.5] 的最大值为 2.5，对应的索引是 1。
第三行 [1.0, 0.5, 1.0] 的最大值为 1.0，对应的索引是 0。

因此，输出将是：
tensor([0, 1, 0])


np.array(gold) 和 np.array(pred):

这两部分代码将 gold 和 pred 转换为 NumPy 数组。
gold 通常表示真实的标签（即数据的实际分类），而 pred 则表示模型预测的标签。
NumPy 数组使得后续处理更为高效，并兼容 classification_report 函数的输入要求。
classification_report(...):

这个函数来自 sklearn.metrics 模块，用于生成分类报告。报告中包含了各种分类指标，例如精确率（precision）、召回率（recall）、F1 分数等，通常以文本格式输出。
该函数的基本调用形式为 classification_report(y_true, y_pred)，其中 y_true 是真实标签，y_pred 是预测标签。
在这行代码中，np.array(gold) 作为真实标签，np.array(pred) 作为预测标签传递给 classification_report。
.rstrip():

这个方法用于去掉字符串末尾的空白字符（包括空格、换行符等）。
在调用 classification_report 之后，生成的报告是一个字符串，这一步是为了清理输出，确保没有多余的空白字符。
.split("\n"):

这个方法用于将字符串按换行符分割成一个列表，列表中的每个元素对应报告中的一行。
结果是一个包含每一行文本的列表，使得进一步处理（如打印或分析特定行）变得方便。
总结
整段代码的目的是生成一个分类模型的评估报告，并将其整理为一个行列表。具体步骤如下：

将真实标签和预测标签转换为 NumPy 数组。
调用 classification_report 生成分类评估的字符串报告。
去掉报告末尾的空白字符。
将报告按行分割成一个列表，以便后续的处理或展示。

下面是一个使用此代码的简单示例：
import numpy as np
from sklearn.metrics import classification_report

# 假设这是实际标签和模型预测标签
gold = [0, 1, 1, 0, 1]
pred = [0, 1, 0, 0, 1]

# 生成分类报告
report = classification_report(np.array(gold), np.array(pred)).rstrip().split("\n")

# 打印分类报告
for line in report:
    print(line)

输出示例
              precision    recall  f1-score   support

           0       0.67      1.00      0.80         3
           1       1.00      0.60      0.75         2

    accuracy                           0.77         5
   macro avg       0.83      0.80      0.77         5
weighted avg       0.80      0.77      0.77         5




"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.attribute_schema = self.valid_data.dataset.attribute_schema
        self.index_to_label = dict((y, x) for x, y in self.attribute_schema.items())

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"object_acc":0, "attribute_acc": 0, "value_acc": 0, "full_match_acc":0}
        self.model.eval()
        gold = []
        pred = []
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, e1_mask, e2_mask, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            gold += labels.detach().tolist()
            with torch.no_grad():
                batch_pred = self.model(input_id, e1_mask, e2_mask) #不输入labels，使用模型当前参数进行预测
                batch_pred = torch.argmax(batch_pred, dim=-1)
                pred += batch_pred.detach().tolist()
        report = classification_report(np.array(gold), np.array(pred)).rstrip().split("\n")
        self.logger.info(report[0])
        self.logger.info(report[-1])






