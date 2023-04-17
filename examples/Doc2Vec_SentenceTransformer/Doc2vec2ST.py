import logging
import os
from datetime import datetime
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from torch.utils.data import DataLoader
from tqdm import tqdm

model_name = 'bert-base-chinese'
train_batch_size = 16
num_epochs = 10
model_save_path = "Doc2vec2ST_model/doc2vec2st_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

time_start = time.time()
print("开始训练时间：" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义SentenceTransformer模型
model = SentenceTransformer(model_name)
# As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
# Negative pairs should have a distance of at least 0.5
margin = 0.5

os.makedirs(model_save_path, exist_ok=True)

# 加载 Doc2Vec 模型
d2v_model = Doc2Vec.load('Doc2Vec_model/doc2vec_same_event_2023-04-03_10-34-32.model')

data_dir = "/Users/PC/PycharmProjects/BS/Long_Text_Match/data/"

############# 加载训练集数据 #############
train_file = os.path.join(data_dir, "same_story/train.txt")
train_samples = []
with open(train_file, encoding="utf-8") as f:
    for i, line in enumerate(f):
        part = line.split('|')
        doc1 = part[5]
        doc2 = part[6]
        text = f'{doc1} {doc2}'
        doc_vec = d2v_model.infer_vector(text.split())
        label = part[0]
        train_samples.append(InputExample(texts=[doc1, doc2], label=int(label), features=doc_vec))

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

################### 评估 ##################
evaluators = []

###### 分类 ######
dev_sentences1 = []
dev_sentences2 = []
dev_labels = []

dev_file = os.path.join(data_dir, "same_story/dev.txt")
with open(dev_file, encoding="utf-8") as f:
    for i, line in enumerate(f):
        part = line.split('|')
        dev_sentences1.append(part[5])
        dev_sentences2.append(part[6])
        dev_labels.append(int(part[0]))

binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
evaluators.append(binary_acc_evaluator)

seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

print("################# Train the model #########################")
# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=seq_evaluator,
          epochs=num_epochs,
          warmup_steps=1000,
          output_path=model_save_path
          )
