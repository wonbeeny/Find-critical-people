from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd
from math import log
from tqdm import tqdm, tqdm_notebook
from time import time
import heapq
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import inaugural

class Find_methods(object):
    def __init__(self, data, sentence, topn=None):
        self.data = data
        self.sentence = sentence
        self.topn = topn
        
    def TF_sim(self):
        df = self.data
        remove_dup = df['new_p_id'].drop_duplicates()
        remove_dup = list(remove_dup.index)
        df = df.loc[remove_dup].reset_index(drop=True)
        route = df.fr_danger_to_occur_route
        new_route = []
        moving_route = []
        for i in range(len(route)):
            new_route.append(route[i].split('->'))
            moving_route.append(' '.join(new_route[i]))
        def cos_sim(A, B):
            return dot(A, B)/(norm(A)*norm(B))
        vocab = list(set(w for doc in moving_route for w in doc.split()))
        vocab.sort()
        tf_result = []
        for i in tqdm_notebook(range(len(moving_route))):
            tf_result.append([])
            d = moving_route[i]
            for j in range(len(vocab)):
                t = vocab[j]        
                tf_result[-1].append(d.count(t))   
        sentences = [[]]
        for j in range(len(vocab)):
            t = vocab[j]
            sentences[-1].append(self.sentence.count(t))
        sim_rate = []
        for i in range(len(tf_result)):
            sim_value = cos_sim(tf_result[i], sentences[0])
            sim_rate.append(sim_value)
        if self.topn>len(sim_rate):
            print('self.topn이 p_id의 수보다 많습니다. self.topn을 수정해주세요.')
        most_count = heapq.nlargest(self.topn, enumerate(sim_rate), key=lambda x: x[1])
        index = [x[0] for x in most_count]
        similarity = pd.DataFrame([x[1] for x in most_count])
        top_p_id=[]
        for i in range(len(index)):
            top_p_id.append(df['new_p_id'][index[i]])
        data = pd.DataFrame()
        data['p_id'] = top_p_id
        data['유사도'] = similarity
        data.to_csv("TF_sim.csv", mode='w')
        
        return data
    
    def count_rate(self):
        df = self.data
        remove_dup = df['new_p_id'].drop_duplicates()
        remove_dup = list(remove_dup.index)
        df = df.loc[remove_dup].reset_index(drop=True)
        doc_df = df[['new_p_id','fr_danger_to_occur_route']].values.tolist()

        for i in range(0, len(doc_df)):
            doc_df[i][1] = doc_df[i][1].split("->")
        
        sentences = self.sentence
        count_route_all = []

        for i in range(len(doc_df)):    # 사람의 총 수
            count_route = []

            for j in range(len(sentences)):    # 각 사람의 총 이동 수
                p_route = doc_df[i][1]
                one_loc = sentences[j]
                count_route.append(p_route.count(one_loc))
            count_route_all.append(round(sum(count_route)/len(doc_df[i][1]), 3))
        if self.topn>len(count_route_all):
            print('self.topn이 p_id의 수보다 많습니다. self.topn을 수정해주세요.')
        most_count = heapq.nlargest(self.topn, enumerate(count_route_all), key=lambda x: x[1])

        index = [x[0] for x in most_count]
        similarity = pd.DataFrame([x[1] for x in most_count])

        top_p_id=[]
        for i in range(len(index)):
            top_p_id.append(df['new_p_id'][index[i]])

        data = pd.DataFrame()
        data['p_id'] = top_p_id
        data['유사도'] = similarity
        data.to_csv("Count_rate_sim.csv", mode='w')

        return data
    
    def D2V_sim(self, upper_bound, model=None, max_epochs=None, vec_size=None, alpha=None, min_count=None, dm=None, dbow_words=None, workers=None, window=None, negative=None, epochs=None):
        df = self.data
        remove_dup = df['new_p_id'].drop_duplicates()
        remove_dup = list(remove_dup.index)
        df = df.loc[remove_dup].reset_index(drop=True)
        doc_df = df[['new_p_id','fr_danger_to_occur_route']].values.tolist()
        for i in range(0, len(doc_df)):
            doc_df[i][1] = doc_df[i][1].split("->")
        train_data = [TaggedDocument(doc, [P_ID]) for P_ID, doc in doc_df]
        
        import os
        if os.path.isfile(model):
            model= Doc2Vec.load(model)
            print("해당 D2V 모델로 유사성 검정을 시작합니다.")
        else:
            print("모델 학습을 시작합니다.")
            model = Doc2Vec(size=vec_size,
                            alpha=alpha, 
                            min_alpha=alpha/10,
                            min_count=min_count,
                            dm = dm,
                            dbow_words=dbow_words,
                            workers = workers,
                            window = window,
                            seed = 12345,
                            negative=negative)

            model.build_vocab(train_data)
            model.train(train_data,
            total_examples=model.corpus_count,
            epochs=epochs)
            from datetime import datetime
            today = datetime.today().strftime('%Y-%m-%d')
            model.save("d2v_{}".format(today))
            print('model save complete')
            
        def vectorize(location, d2v):
            vocabulary = list(d2v.wv.vocab.keys())
            all_vec = np.array([d2v[t] for t in location if t in vocabulary])
            return all_vec.mean(axis=0)
        infer_vec = [vectorize(doc.words, model) for doc in train_data]
        
        infer_sent = [vectorize(self.sentence, model)]

        cos = cosine_similarity(infer_vec, infer_sent)

        cos_upper = []
        for i in range(len(cos)):
            if cos[i] > upper_bound:
                cos_upper.append(cos[i])

        tag = heapq.nlargest(len(cos_upper), enumerate(cos_upper), key=lambda x: x[1])

        index = [x[0] for x in tag]
        similarity = pd.DataFrame([x[1] for x in tag])

        top_p_id=[]
        for i in range(len(index)):
            top_p_id.append(doc_df[index[i]][0])

        data = pd.DataFrame()
        data['p_id'] = top_p_id
        data['유사도'] = similarity
        data.to_csv("D2V_sim.csv", mode='w')
        
        return data
    
    
class combind_method_count(object):
    def __init__(self, data_1, data_2, data_3):
        self.data_1 = data_1
        self.data_2 = data_2
        self.data_3 = data_3
    
    def count_p_id(self):
        count_p_id = pd.concat([self.data_1, self.data_2, self.data_3])

        count={}

        for i in list(count_p_id['p_id']):
            try: count[i] += 1
            except: count[i]=1

        count = {key: value for key, value in count.items() if value != 1}

        data = pd.DataFrame()
        data['p_id'] = list(count.keys())
        data['count'] = list(count.values())
        data.to_csv("method_count.csv", mode='w')

        return data