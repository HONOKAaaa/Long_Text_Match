# 特征提取并进行csv格式数据集生成，格式为
# [id, doc_id1, doc_id2, key_sentences1, key_sentences2, key_words1, key_words2, key_phrases1, key_phrases2, labels]

from textrank4zh import TextRank4Keyword, TextRank4Sentence
import pandas as pd
from tqdm import tqdm
from textrank_util import keysentences_extraction, keywords_extraction, keyphrases_extraction

ls1 = ["same_event", "same_story"]
ls2 = ["train"]

for i in ls2:
    doc_id1, doc_id2, key_sentences1, key_sentences2, labels, key_words1, key_words2, key_phrases1, key_phrases2 = [], [], [], [], [], [], [], [], []
    for j in ls1:
        data_path = "C:/Users/PC/PycharmProjects/BS/Long_Text_Match/data/" + j + "/" + i + ".txt"
        print(data_path + " Start!")

        for line in tqdm(open(data_path, encoding="utf-8")):
            part = line.split('|')
            label = int(part[0])
            did1 = part[1]
            did2 = part[2]
            title1 = part[3]
            title2 = part[4]
            doc1 = part[5]
            doc2 = part[6]
            if len(doc1) == 0 or len(doc2) == 0:
                print("Error")
                continue
            doc1 = "".join((doc1 + title1).split())
            doc2 = "".join((doc2 + title2).split())
            doc1_key_sentences = keysentences_extraction(doc1)
            doc2_key_sentences = keysentences_extraction(doc2)
            doc1_key_words = keywords_extraction(doc1)
            doc2_key_words = keywords_extraction(doc2)
            doc1_key_phrases = keyphrases_extraction(doc1)
            doc2_key_phrases = keyphrases_extraction(doc2)

            s1, s2 = [], []
            for k in range(len(doc1_key_sentences)):
                s1.append(doc1_key_sentences[k]["sentence"])
            for k in range(len(doc2_key_sentences)):
                s2.append(doc2_key_sentences[k]["sentence"])
            key_sentences1.append(s1)
            key_sentences2.append(s2)
            w1,  w2 = [], []
            for k in range(len(doc1_key_words)):
                w1.append(doc1_key_words[k]["word"])
            for k in range(len(doc2_key_words)):
                w2.append(doc2_key_words[k]["word"])
            key_words1.append(w1)
            key_words2.append(w2)
            key_phrases1.append(doc1_key_phrases)
            key_phrases2.append(doc2_key_phrases)
            labels.append(label)
            doc_id1.append(did1)
            doc_id2.append(did2)


        print(data_path + " Finish!")

    dataframe = pd.DataFrame({'doc_id1': doc_id1, 'doc_id2': doc_id2, 'key_sentences1': key_sentences1, 'key_sentences2': key_sentences2,
                              'key_words1': key_words1, 'key_words2': key_words2, 'key_phrases1': key_phrases1, 'key_phrases2': key_phrases2, 'labels': labels})
    path_save = "C:/Users/PC/PycharmProjects/BS/Long_Text_Match/data/event_and_story/" + i + ".csv"
    dataframe.to_csv(path_save, index=True, sep=',', encoding="utf-8-sig")
