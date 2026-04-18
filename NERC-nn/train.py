#! /usr/bin/python3

import sys
from contextlib import redirect_stdout

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, Lambda

from dataset import *
from codemaps import *

def build_network(codes) :

   # sizes
   n_words = codes.get_n_words()
   n_lc_words = codes.get_n_lc_words()
   n_sufs = codes.get_n_sufs()
   n_shapes = codes.get_n_shapes()
   n_caps = codes.get_n_caps()
   n_nums = codes.get_n_nums()
   n_dashes = codes.get_n_dashes()
   n_gazetteer = codes.get_n_gazetteer()
   n_labels = codes.get_n_labels()   
   max_len = codes.maxlen

   inptW = Input(shape=(max_len,)) # word input layer & embeddings
   embW = Embedding(input_dim=n_words, output_dim=100,
                    input_length=max_len, mask_zero=True)(inptW)  

   inptLW = Input(shape=(max_len,)) # lowercased word input layer & embeddings
   embLW = Embedding(input_dim=n_lc_words, output_dim=100,
                     input_length=max_len, mask_zero=True)(inptLW)
   
   inptS = Input(shape=(max_len,))  # suf input layer & embeddings
   embS = Embedding(input_dim=n_sufs, output_dim=50,
                    input_length=max_len, mask_zero=True)(inptS) 

   inptC = Input(shape=(max_len,))  # capitalization feature
   embC = Embedding(input_dim=n_caps, output_dim=8,
                    input_length=max_len, mask_zero=True)(inptC)

   inptN = Input(shape=(max_len,))  # number feature
   embN = Embedding(input_dim=n_nums, output_dim=6,
                    input_length=max_len, mask_zero=True)(inptN)

   inptD = Input(shape=(max_len,))  # dash feature
   embD = Embedding(input_dim=n_dashes, output_dim=4,
                    input_length=max_len, mask_zero=True)(inptD)

   inptSh = Input(shape=(max_len,))  # token shape feature
   embSh = Embedding(input_dim=n_shapes, output_dim=20,
                     input_length=max_len, mask_zero=True)(inptSh)

   inptG = Input(shape=(max_len,))  # gazetteer feature
   embG = Embedding(input_dim=n_gazetteer, output_dim=8,
                    input_length=max_len, mask_zero=True)(inptG)

   dropW = Dropout(0.1)(embW)
   dropLW = Dropout(0.1)(embLW)
   dropS = Dropout(0.1)(embS)
   dropC = Dropout(0.05)(embC)
   dropN = Dropout(0.05)(embN)
   dropD = Dropout(0.05)(embD)
   dropSh = Dropout(0.1)(embSh)
   dropG = Dropout(0.05)(embG)
   drops = concatenate([dropW, dropLW, dropS, dropC, dropN, dropD, dropSh, dropG])

   # biLSTM   
   bilstm = Bidirectional(LSTM(units=200, return_sequences=True,
                               dropout=0.1, recurrent_dropout=0.1))(drops) 
   # output softmax layer
   out = TimeDistributed(Dense(n_labels, activation="softmax"))(bilstm)

   # build and compile model
   model = Model([inptW, inptLW, inptS, inptC, inptN, inptD, inptSh, inptG], out)
   model.compile(optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
   
   return model
   


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

# directory with files to process
traindir = sys.argv[1]
validationdir = sys.argv[2]
modelname = sys.argv[3]

# load train and validation data
traindata = Dataset(traindir)
valdata = Dataset(validationdir)

# create indexes from training data
max_len = 150
suf_len = 5
codes  = Codemaps(traindata, max_len, suf_len)

# build network
model = build_network(codes)
with redirect_stdout(sys.stderr) :
   model.summary()

# encode datasets
Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)
Xv = codes.encode_words(valdata)
Yv = codes.encode_labels(valdata)

# train model
with redirect_stdout(sys.stderr) :
   model.fit(Xt, Yt, batch_size=32, epochs=10, validation_data=(Xv,Yv), verbose=1)

# save model and indexs
model.save(modelname)
codes.save(modelname)
