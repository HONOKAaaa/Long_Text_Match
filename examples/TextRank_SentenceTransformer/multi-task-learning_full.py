
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

model_name = 'bert-base-chinese'
train_batch_size = 2
num_epochs = 3
num_sen = 2
model_save_path = 'TRST_model/TRST_multitask_' + str(num_sen) + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(format='%(asctime)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

model = SentenceTransformer(model_name)

# As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

#Negative pairs should have a distance of at least 0.5
margin = 0.5

os.makedirs(model_save_path, exist_ok=True)

print("######### Read train data  ##########")
######### Read train data  ##########
# Read train data
train_samples_MultipleNegativesRankingLoss = []
train_samples_ConstrativeLoss = []

df = pd.read_csv("C:/Users/PC/PycharmProjects/BS/Long_Text_Match/data/same_event/classification/train.csv")
train_num = len(df) // 2
flag = 0
for idx, item in df.iterrows():
    flag += 1
    text1 = "。".join(item['key_sentences1'].split()[:num_sen])
    text2 = "。".join(item['key_sentences2'].split()[:num_sen])
    train_samples_ConstrativeLoss.append(InputExample(texts=[text1, text2], label=int(item['labels'])))
    if item['labels'] == 1:
        train_samples_MultipleNegativesRankingLoss.append(InputExample(texts=[text1, text2], label=1))
        train_samples_MultipleNegativesRankingLoss.append(InputExample(texts=[text2, text1], label=1))
    if flag > train_num:
        break

# Create data loader and loss for MultipleNegativesRankingLoss
train_dataloader_MultipleNegativesRankingLoss = DataLoader(train_samples_MultipleNegativesRankingLoss, shuffle=True, batch_size=train_batch_size)
train_loss_MultipleNegativesRankingLoss = losses.MultipleNegativesRankingLoss(model)

# Create data loader and loss for OnlineContrastiveLoss
train_dataloader_ConstrativeLoss = DataLoader(train_samples_ConstrativeLoss, shuffle=True, batch_size=train_batch_size)
train_loss_ConstrativeLoss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

print("################### Development  Evaluators ##################")
# We add 3 evaluators, that evaluate the model on Duplicate Questions pair classification,
# Duplicate Questions Mining, and Duplicate Questions Information Retrieval
evaluators = []

print("###### Classification ######")
# Given (quesiton1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
dev_sentences1 = []
dev_sentences2 = []
dev_labels = []

df = pd.read_csv("C:/Users/PC/PycharmProjects/BS/Long_Text_Match/data/same_event/classification/dev.csv")
dev_num = len(df) // 2
flag = 0
for idx, item in df.iterrows():
    flag += 1
    text1 = "。".join(item['key_sentences1'].split()[:num_sen])
    text2 = "。".join(item['key_sentences2'].split()[:num_sen])
    dev_sentences1.append(text1)
    dev_sentences2.append(text2)
    dev_labels.append(int(item['labels']))
    if flag > dev_num:
        break

binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
evaluators.append(binary_acc_evaluator)

print("###### Duplicate Questions Mining ######")
# Given a large corpus of questions, identify all duplicates in that corpus.

# For faster processing, we limit the development corpus to only 10,000 sentences.
max_dev_samples = 2000
dev_sentences = {}
dev_duplicates = []
dataset_path = "C:/Users/PC/PycharmProjects/BS/Long_Text_Match/data/same_event/"
with open(os.path.join(dataset_path, "duplicate-mining/dev_corpus.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_sentences[row['did']] = "。".join(row['key_sentences'].split()[:num_sen])

        if len(dev_sentences) >= max_dev_samples:
            break

with open(os.path.join(dataset_path, "duplicate-mining/dev_duplicates.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['did1'] in dev_sentences and row['did2'] in dev_sentences:
            dev_duplicates.append([row['did1'], row['did2']])


# The ParaphraseMiningEvaluator computes the cosine similarity between all sentences and
# extracts a list with the pairs that have the highest similarity. Given the duplicate
# information in dev_duplicates, it then computes and F1 score how well our duplicate mining worked
paraphrase_mining_evaluator = evaluation.ParaphraseMiningEvaluator(dev_sentences, dev_duplicates, name='dev')
evaluators.append(paraphrase_mining_evaluator)


print("###### Duplicate Questions Information Retrieval ######")
# Given a question and a large corpus of thousands questions, find the most relevant (i.e. duplicate) question
# in that corpus.

# For faster processing, we limit the development corpus to only 10,000 sentences.
max_corpus_size = 2000

ir_queries = {}             # Our queries (qid => question)
ir_needed_qids = set()      # QIDs we need in the corpus
ir_corpus = {}              # Our corpus (qid => question)
ir_relevant_docs = {}       # Mapping of relevant documents for a given query (qid => set([relevant_question_ids])

with open(os.path.join(dataset_path, 'information-retrieval/dev-queries.tsv'), encoding='utf8') as fIn:
    next(fIn)  # Skip header
    for line in fIn:
        qid, query, duplicate_ids = line.strip().split('\t')
        duplicate_ids = duplicate_ids.split(',')
        ir_queries[qid] = "。".join(query.split()[:num_sen])
        ir_relevant_docs[qid] = set(duplicate_ids)

        for qid in duplicate_ids:
            ir_needed_qids.add(qid)

# First get all needed relevant documents (i.e., we must ensure, that the relevant questions are actually in the corpus
distraction_questions = {}
with open(os.path.join(dataset_path, 'information-retrieval/corpus.tsv'), encoding='utf8') as fIn:
    next(fIn)  # Skip header
    for line in fIn:
        qid, question = line.strip().split('\t')

        if qid in ir_needed_qids:
            ir_corpus[qid] = "。".join(question.split()[:num_sen])
        else:
            distraction_questions[qid] = "。".join(question.split()[:num_sen])

# Now, also add some irrelevant questions to fill our corpus
other_qid_list = list(distraction_questions.keys())
random.shuffle(other_qid_list)

for qid in other_qid_list[0:max(0, max_corpus_size-len(ir_corpus))]:
    ir_corpus[qid] = distraction_questions[qid]

# Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR
# metrices. For our use case MRR@k and Accuracy@k are relevant.
ir_evaluator = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus, ir_relevant_docs)

evaluators.append(ir_evaluator)




# Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
# We optimize the model with respect to the score from the last evaluator (scores[-1])
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])


logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

print("################# Train the model #########################")
# Train the model
model.fit(train_objectives=[(train_dataloader_MultipleNegativesRankingLoss, train_loss_MultipleNegativesRankingLoss), (train_dataloader_ConstrativeLoss, train_loss_ConstrativeLoss)],
          evaluator=seq_evaluator,
          epochs=num_epochs,
          warmup_steps=1000,
          output_path=model_save_path
          )