# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:12:29 2021

@author: """

import re
import numpy as np
import pandas as pd
import scipy.sparse as ss
import matplotlib.pyplot as plt
# %matplotlib inline
import graphviz

from corextopic import corextopic as ct
from corextopic import vis_topic as vt # jupyter notebooks will complain matplotlib is being loaded twice

# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv(".../fed_speeches_1996_2020.csv",encoding="latin-1",header=0,index_col=0)

data.drop(['link','text_len'], axis=1, inplace=True)
# data = data[646:1499]

# data['text_lem'] = data['text_lem'].str.replace('\d+', '')
data['text'] = data['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

data['text'] = data['text'].str.replace('[^\w\s]',' ')


banned = ['Governor','Vice','Chairman','Vice Chair','Chair', 'for Supervision',
          'for Supervision and  of the Financial Stability Board',
          'of the Financial Stability Board', 'and']

data['name'] = data['speaker'].replace(dict(zip(banned,['']*len(banned))),regex=True)

data['name'] = data['name'].str.strip()

# data['name'] = data['name'].str.split().str[-1]
# data['name'][820]
# data['name'][893]
data.name.value_counts()

# Transform data into a sparse matrix
vectorizer = CountVectorizer(stop_words='english',lowercase=True,max_features=20000)
doc_word = vectorizer.fit_transform(data.text_lem)
doc_word = ss.csr_matrix(doc_word)

doc_word.shape # n_docs x m_words

# Get words that label the columns (needed to extract readable topics and make anchoring easier)
words = list(np.asarray(vectorizer.get_feature_names()))

not_digit_inds = [ind for ind,word in enumerate(words) if not word.isdigit()]
doc_word = doc_word[:,not_digit_inds]
words = [word for ind,word in enumerate(words) if not word.isdigit()]

doc_word.shape # n_docs x m_words

# Train the CorEx topic model with 10 topics
topic_model = ct.Corex(n_hidden=20, words=words, max_iter=2000, verbose=False, seed=456987321,docs=data.name)
topic_model.fit(doc_word, words=words,docs=data.name);

# results as a datafame
topic_df = pd.DataFrame(
    topic_model.transform(doc_word,details=False), 
    columns=["topic_{}".format(i+1) for i in range(20)]
).astype(float)

topic_df.index = data.index
data = pd.concat([data, topic_df], axis=1)

# Print a single topic from CorEx topic model
topic_model.get_topics(topic=0, n_words=15)

# Print all topics from the CorEx topic model
topics = topic_model.get_topics(n_words=50)

for n,topic in enumerate(topics):
    topic_words,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))
    
topic_model.get_topics(topic=0, n_words=20, print_words=True)

print(topic_model.p_y_given_x.shape)
print(topic_model.labels.shape) # n_docs x k_topics


print(topic_model.clusters)
print(topic_model.clusters.shape) # m_words

# Print a single topic from CorEx topic model
topic_model.get_top_docs(topic=2, n_docs=30, sort_by='log_prob',print_docs=True)

""" Total Correlation and Model Selection """

# overall TC
topic_model.tc

# Topic TC
topic_model.tcs.shape # k_topics

print(np.sum(topic_model.tcs))
print(topic_model.tc)


# Selecting number of topics: one way to choose the number of topics 

# %matplotlib inline

plt.figure(figsize=(10,5))
plt.bar(range(1,topic_model.tcs.shape[0]+1), topic_model.tcs, color='#4e79a7', width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16);

""" Hierarchical Topic Models """

# Train a second layer to the topic model
tm_layer2 = ct.Corex(n_hidden=10)
tm_layer2.fit(topic_model.labels);

# Train a third layer to the topic model
tm_layer3 = ct.Corex(n_hidden=1)
tm_layer3.fit(tm_layer2.labels);

vt.vis_hierarchy([topic_model, tm_layer2, tm_layer3], column_label=words, max_edges=200, prefix='topic-model-example')
