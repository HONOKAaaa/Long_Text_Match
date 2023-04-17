import csv
import random
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util, evaluation, SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import pandas as pd


model_name = 'bert-base-chinese'                        # 预训练模型名称
train_batch_size = 8                                    # 训练时每个batch的大小
num_epochs = 10                                         # 遍历整个数据集的次数
num_sen = 2                                             # 选择的关键句个数

# 模型的保存路径，“TRST_multitask_”后的数字表示选择的关键句个数，最后是开始训练时间
model_save_path = 'TRST_model/TRST_multitask_' + str(num_sen) + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 配置Python中的日志系统，完成以下功能
# 设置日志消息的格式；
# 设置日期时间格式；
# 设置日志消息的级别为INFO，只记录INFO及以上级别的日志；
# 配置日志处理器，可以将日志消息写入文件、控制台等。
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

# 初始化一个基于预训练模型的句子嵌入模型
model = SentenceTransformer(model_name)

# 作为距离度量，我们使用余弦距离(cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

# margin是指两个文本向量之间的距离应该超过多少，才能被认为是不相似的，负对之间的距离至少为0.5
margin = 0.5

# 检查目录model_save_path是否存在，如果不存在则创建该目录。
os.makedirs(model_save_path, exist_ok=True)

######### 读取训练数据 ##########

# train_samples_MultipleNegativesRankingLoss 用于存储使用多个负例的排序损失函数训练的样本数据。
# train_samples_ConstrativeLoss 用于存储使用对比损失函数训练的样本数据。
train_samples_MultipleNegativesRankingLoss = []
train_samples_ConstrativeLoss = []

# 读取训练数据并转换为DataFrame格式
df_train = pd.read_csv("C:/Users/PC/PycharmProjects/BS/Long_Text_Match/data/same_event/classification/train.csv")

train_num = len(df_train) // 2        # 读取数据的行数
flag = 0
for idx, item in df_train.iterrows():
    flag += 1
    # 添加特殊标记，所选句子被串联成一个文本序列，以[CLS]标记开始，以[SEP]标记分开
    text1 = "[SEP]".join(item['key_sentences1'].split()[:num_sen])
    text1 = "[CLS]" + text1 + "[SEP]"
    text2 = "[SEP]".join(item['key_sentences2'].split()[:num_sen])
    text2 = "[CLS]" + text2 + "[SEP]"
    train_samples_ConstrativeLoss.append(InputExample(texts=[text1, text2], label=int(item['labels'])))
    if item['labels'] == 1:
        train_samples_MultipleNegativesRankingLoss.append(InputExample(texts=[text1, text2], label=1))
        train_samples_MultipleNegativesRankingLoss.append(InputExample(texts=[text2, text1], label=1))
    if flag > train_num:
        break

# 为MultipleNegativesRankingLoss创建数据加载器和损失
train_dataloader_MultipleNegativesRankingLoss = DataLoader(train_samples_MultipleNegativesRankingLoss, shuffle=True, batch_size=train_batch_size)
train_loss_MultipleNegativesRankingLoss = losses.MultipleNegativesRankingLoss(model)
# 为OnlineContrastiveLoss创建数据加载器和损失
train_dataloader_ConstrativeLoss = DataLoader(train_samples_ConstrativeLoss, shuffle=True, batch_size=train_batch_size)
train_loss_ConstrativeLoss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)


########### 评估器 ##############

# 创建列表保存评估器
evaluators = []

###### 相同事件或不相同事件分类 ######
# 给定(文本1，文本2)，这是否是一个相同事件?
# 评估者将计算两个文本的嵌入向量，然后计算余弦相似度，如果相似度高于一个阈值，就说明是一个相同事件。
dev_sentences1 = []
dev_sentences2 = []
dev_labels = []

df_dev = pd.read_csv("C:/Users/PC/PycharmProjects/BS/Long_Text_Match/data/same_event/classification/dev.csv")
dev_num = len(df_dev) // 2
flag = 0
for idx, item in df_dev.iterrows():
    flag += 1
    text1 = "[SEP]".join(item['key_sentences1'].split()[:num_sen])
    text1 = "[CLS]" + text1 + "[SEP]"
    text2 = "[SEP]".join(item['key_sentences2'].split()[:num_sen])
    text2 = "[CLS]" + text2 + "[SEP]"
    dev_sentences1.append(text1)
    dev_sentences2.append(text2)
    dev_labels.append(int(item['labels']))
    if flag > dev_num:
        break

binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
evaluators.append(binary_acc_evaluator)

# 创建一个SequentialEvaluator。这个SequentialEvaluator按顺序运行所求值器。
# 我们根据上一次评估器的分数(scores[-1])来优化模型
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)


################# 训练模型 #################

model.fit(train_objectives=[(train_dataloader_MultipleNegativesRankingLoss, train_loss_MultipleNegativesRankingLoss),
                            (train_dataloader_ConstrativeLoss, train_loss_ConstrativeLoss)],
          evaluator=seq_evaluator,
          epochs=num_epochs,
          warmup_steps=1000,
          evaluation_steps=250,
          checkpoint_save_steps=250,
          output_path=model_save_path
          )