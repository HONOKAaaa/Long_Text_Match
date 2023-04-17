from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import logging
import os
import gensim
import smart_open
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_dir = "/Users/PC/PycharmProjects/BS/Long_Text_Match/data/"
train_file = os.path.join(data_dir, "same_event/train.txt")
test_file = os.path.join(data_dir, "same_event/test.txt")

def read_corpus(fname, tokens_only=False):
    with open(fname, encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f)):
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

model = Doc2Vec.load('Doc2vec_model/doc2vec_same_event_2023-04-03_10-34-32.model')

ranks = []
second_ranks = []
for doc_id in tqdm(range(1, 2)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

import collections
print(ranks)
counter = collections.Counter(ranks)
print(counter)

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


