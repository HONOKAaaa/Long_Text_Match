from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import logging
import os
import gensim
import jieba
from datetime import datetime
import time



time_start = time.time()
print("开始训练时间：" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_dir = "/Users/PC/PycharmProjects/BS/Long_Text_Match/data/"
train_file = os.path.join(data_dir, "same_event/train.txt")
test_file = os.path.join(data_dir, "same_event/test.txt")

def read_corpus(fname, tokens_only=False):
    with open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            part = line.split('|')
            doc = []
            doc.append(part[5])
            doc.append(part[6])
            for text in doc:
                tokens = gensim.utils.simple_preprocess(text)
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(train_file))
test_corpus = list(read_corpus(test_file, tokens_only=True))

vector_size = 300
window = 10
min_count = 5
epochs = 1
model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)

model.build_vocab(train_corpus)

# print(f"Word 'penalty' appeared {model.wv.get_vecattr('penalty', 'count')} times in the training corpus.")

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

para = "-".join([str(vector_size), str(window), str(min_count), str(epochs)])
model_save_path = "Doc2Vec_model/doc2vec_same_event_" + para + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".model"
model.save(model_save_path)

time_end = time.time()
print("结束训练时间：" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

time_c = time_end - time_start   #运行所花时间
print('time cost', time_c, 's')
