#加载第三方库
import pandas as pd
import numpy as np
#读取文件
df1 = pd.read_csv("../data/event_and_story/train (1).csv")
df2 = pd.read_csv("../data/event_and_story/train.csv")
#合并
df = pd.concat([df1, df2])
df.drop_duplicates()  #数据去重
#保存合并后的文件
df.to_csv('文件.csv', encoding='utf-8')