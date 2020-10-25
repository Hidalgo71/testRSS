# Gerekli Kütüphaneler
import pandas as pd
import numpy as np
import re, io
import os, pkgutil, json, urllib
from urllib.request import urlopen
import warnings
from collections import Counter
import gensim
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scattertext import CorpusFromPandas, produce_scattertext_explorer
import scattertext as st
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pprint import pprint
from scipy.stats import rankdata, hmean, norm
import spacy

warnings.filterwarnings(action='ignore')
import multiprocessing

# data = pd.read_csv("E:/real sets/allEKdata.csv" , engine='c')
# print(data.head())
urldata = pd.read_csv('E:/real sets/allEKdata.csv')
urldata.drop('Unnamed: 0', axis=1, inplace=True)
print(urldata.head())

text_data = urldata[['Sentence', 'NegPos']]
print(text_data.head())

# ax = sns.barplot(data.NegPos.value_counts().index, y=data.Kategori.value_counts(), data=data, palette="rainbow")

corpus = []
for i in range(len(urldata)):
    corpus.append(text_data.iloc[i, 0])
# Corpus ile TF-IDF modelini fit ediyoruz.
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

# Burada transform ediyorum.
X = vectorizer.transform(corpus).toarray()
y = text_data.NegPos
# Burada %25-%75 olmak üzere 2'ye ayırıyorum.
# Stratify=y'de train ve testteki y oranının aynı olmasını sağlıyor.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39, stratify=y)
rf = RandomForestClassifier(class_weight='balanced_subsample')
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(accuracy_score(y_test, y_pred_rf))
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)
