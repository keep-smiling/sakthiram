

import dill
import keras.backend as K
import multiprocessing
import tensorflow as tf

from gensim.models.word2vec import Word2Vec

import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer


np.random.seed(1000)

use_gpu = True

config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(), 
                        inter_op_parallelism_threads=multiprocessing.cpu_count(), 
                        allow_soft_placement=True, 
                        device_count = {'CPU' : 1, 
                                        'GPU' : 1 if use_gpu else 0})

session = tf.Session(config=config)
K.set_session(session)

model_location = './model/'
corpus = []
labels = []

filepath = 'first.txt' 
with open(filepath) as fp:  
   line = fp.readline()
   while line:
        corpus.append(line)
        labels.append(1)
        line = fp.readline()

filepath = 'second.txt' 
with open(filepath) as fp:  
   line = fp.readline()
   while line:
        corpus.append(line)
        labels.append(0)
        line = fp.readline()
        
l1= len(corpus)


filepath = 'first1.txt' 
with open(filepath) as fp:  
   line = fp.readline()
   while line:
        corpus.append(line)
        labels.append(1)
        line = fp.readline()

filepath = 'second1.txt' 
with open(filepath) as fp:  
   line = fp.readline()
   while line:
        corpus.append(line)
        labels.append(0)
        line = fp.readline()
        
l2 = len(corpus)
diff = l2-l1

print(l1,diff)

tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, tweet in enumerate(corpus):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(tweet) if not t.startswith('@')]
    tokenized_corpus.append(tokens)

with open(model_location + 'tokenized_corpus.dill', 'wb') as f:
    dill.dump(tokenized_corpus, f)

with open(model_location + 'tokenized_corpus.dill', 'rb') as f:
    tokenized_corpus = dill.load(f)

vector_size = 512
window_size = 10

word2vec = Word2Vec(sentences=tokenized_corpus,
                    size=vector_size, 
                    window=window_size, 
                    negative=20,
                    iter=50,
                    seed=1000,
                    workers=multiprocessing.cpu_count())

word2vec.save(model_location + 'word2vec.model')


word2vec = Word2Vec.load(model_location + 'word2vec.model')


X_vecs = word2vec.wv

del word2vec
del corpus


train_size = l1
test_size = diff


avg_length = 0.0
max_length = 0

for tweet in tokenized_corpus:
    if len(tweet) > max_length:
        max_length = len(tweet)
    avg_length += float(len(tweet))
    
print('Average tweet length: {}'.format(avg_length / float(len(tokenized_corpus))))
print('Max tweet length: {}'.format(max_length))


max_tweet_length = 15

indexes = np.random.choice(len(tokenized_corpus), train_size + test_size, replace=False)

X_train = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())
Y_train = np.zeros((train_size, 2), dtype=np.int32)
X_test = np.zeros((test_size, max_tweet_length, vector_size), dtype=K.floatx())
Y_test = np.zeros((test_size, 2), dtype=np.int32)

     
#print(X_train)
#print(Y_train)
for i, index in enumerate(indexes):
    for t, token in enumerate(tokenized_corpus[index]):
        if t >= max_tweet_length:
            break
        
        if token not in X_vecs:
            continue
    
        if i < train_size:
            X_train[i, t, :] = X_vecs[token]
        else:
            X_test[i - train_size, t, :] = X_vecs[token]
            
    if i < train_size:
        Y_train[i, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
    else:
        Y_test[i - train_size, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
        
np.save("train_features.npy",X_train)
np.save("train_labels.npy", Y_train)
np.save("test_features.npy",X_test)
np.save("test_labels.npy",Y_test)


print("features Generated")