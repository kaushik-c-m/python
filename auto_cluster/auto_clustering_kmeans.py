# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:21:44 2019

@author: Kaushik C M
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from kneed import KneeLocator
from gensim.summarization import keywords
#from sklearn.externals import joblib

cwd = os.getcwd()
keywords_op=[]
FileInput = pd.read_csv(cwd+"\9379d3dc-0-dataset.tar\\dataset\\news_demo.csv")
FileInput["club"] = FileInput["headline"].map(str) + FileInput["text"]

englishStemmer2 = SnowballStemmer("english", ignore_stopwords=True)
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(englishStemmer2.stem(word))
        #stem_sentence.append(" ")
    return " ".join(stem_sentence)

for news in FileInput["club"] :
    # FileInput.iloc[0:4,2]:
    stem_news = stemSentence(news)
    keywords_op.append(" ".join(keywords(stem_news, ratio=0.8, split=True)))
    
    
stem_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.5)      
X = stem_vectorizer.fit_transform(keywords_op)
import pickle
vocab=open("stem_feature_30.pkl","wb")
pickle.dump(stem_vectorizer.vocabulary_,vocab)
vocab.close()

def elbow_plot(data, maxK=10, seed_centroids=None):
    """
        parameters:
        - data: pandas DataFrame (data to be fitted)
        - maxK (default = 10): integer (maximum number of clusters with which to run k-means)
        - seed_centroids (default = None ): float (initial value of centroids for k-means)
    """
    sse = {}
    for k in range(1, maxK):
        try:
            print("k: ", k)
            if seed_centroids is not None:
                seeds = seed_centroids.head(k)
                kmeans = KMeans(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
            else:
                kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(data)
            sse[k] = kmeans.inertia_
            print("sse[k]: ", sse[k])
        except Exception as e:
            print(str(e))
            sse[k] = 0
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()
    return sse

ssse=elbow_plot(X) 
kindel = KneeLocator(list(ssse.keys()), list(ssse.values()), curve='convex', direction='decreasing')
print(kindel.knee)

true_k = kindel.knee
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1)
model.fit(X)
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = stem_vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print()

kmeans_model_pkl = open("kmean_model_trained_stem_feature_today.pkl","wb")
pickle.dump(model, kmeans_model_pkl)
kmeans_model_pkl.close()