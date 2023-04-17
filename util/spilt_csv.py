import pandas as pd

df = pd.read_csv('../data/event_and_story/dev.csv')
data = df.loc[5811:]
data.to_csv('dev.csv', index=False, encoding='utf-8-sig')