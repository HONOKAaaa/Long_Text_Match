# import gensim
# import os
# from datetime import datetime
# # 查看CPU核心数量
# # print("CPU核心数为：", os.cpu_count())
#
# # data_dir = "/Users/PC/PycharmProjects/BS/Long_Text_Match/data/"
# # train_file = os.path.join(data_dir, "same_event/train")
# # print(train_file)
#
# model_save_path = "Doc2Vec_model/doc2vec_same_event_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".model"
# print(model_save_path)
from datetime import datetime
#
# vector_size = 300
# window = 10
# min_count = 5
# epochs = 200
# para = "-".join([str(vector_size), str(window), str(min_count), str(epochs)])
# model_save_path = "Doc2Vec_model/doc2vec_same_event_" + para + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".model"
# print(model_save_path)
import pandas as pd
# df = pd.read_csv("/data/same_event/classification/train.csv")
# for idx, item in df.iterrows():
#     text1 = "。".join(item['key_sentences1'].split()[:2])
#     text2 = item['key_sentences2']
#     print(text1)
#     print(text2)
#     break

df = pd.read_csv("../data/same_event/classification/dev.csv")
for idx, item in df.iterrows():
    print(idx)
    print(item)
    break