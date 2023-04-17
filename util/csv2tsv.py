import csv
import random

import pandas as pd

####### Output for duplicate mining #######
# def write_mining_files(name):
#     with open('../data/same_event/duplicate-mining/'+name+'_corpus.tsv', 'w', encoding='utf8') as fOut:
#         fOut.write("did\tkey_sentences\n")
#         df = pd.read_csv("../data/same_event/classification/dev.csv")
#         for idx, item in df.iterrows():
#             fOut.write("{}\t{}\n".format(item["doc_id1"], item["key_sentences1"]))
#             fOut.write("{}\t{}\n".format(item["doc_id2"], item["key_sentences2"]))
#
#     with open('../data/same_event/duplicate-mining/'+name+'_duplicates.tsv', 'w', encoding='utf8') as fOut:
#         fOut.write("did1\tdid2\n")
#         df = pd.read_csv("../data/same_event/classification/dev.csv")
#         for idx, item in df.iterrows():
#             if item["labels"] == 1:
#                 fOut.write("{}\t{}\n".format(item["doc_id1"], item["doc_id2"]))
#
# write_mining_files('dev')

num_dev_queries = 100
dev_ids = set()
df = pd.read_csv("../data/same_event/classification/dev.csv")

# Create dev queries
dev_did, dev_key_sentences, dev_duplicate_dids = [], [], []
corpus_did, corpus_key_sentences = [], []
rnd_dev_ids = [i for i in range(len(df))]
random.shuffle(rnd_dev_ids)

for a in rnd_dev_ids:
    if len(dev_ids) < num_dev_queries:
        dev_ids.add(a)

for idx, item in df.iterrows():
    if idx in dev_ids:
        did, key_sentences1, key_sentences2, duplicate_dids = item["doc_id1"], item["key_sentences1"], item["key_sentences2"], item["doc_id2"]
        dev_did.append(did)
        dev_key_sentences.append(key_sentences1)
        dev_duplicate_dids.append(duplicate_dids)
        corpus_did.append(duplicate_dids)
        corpus_key_sentences.append(key_sentences2)
    else:
        did1, did2, key_sentences1, key_sentences2 = item["doc_id1"], item["doc_id2"], item["key_sentences1"], item["key_sentences2"]
        corpus_did.append(did1)
        corpus_did.append(did2)
        corpus_key_sentences.append(key_sentences1)
        corpus_key_sentences.append(key_sentences2)

dev_queries = pd.DataFrame({"did": dev_did, "key_sentences": dev_key_sentences, "duplicate_dids": dev_duplicate_dids})
corpus = pd.DataFrame({"did": corpus_did, "key_sentences": corpus_key_sentences})

with open('../data/same_event/information-retrieval/corpus.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("did\tkey_sentences\n")
    for idx, item in corpus.iterrows():
        fOut.write("{}\t{}\n".format(item["did"], item["key_sentences"]))

with open('../data/same_event/information-retrieval/dev-queries.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("did\tkey_sentences\tduplicate_dids\n")
    for idx, item in dev_queries.iterrows():
        fOut.write("{}\t{}\t{}\n".format(item["did"], item["key_sentences"], item["duplicate_dids"]))
