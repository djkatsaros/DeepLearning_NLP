# Script to train deep learning classifiers for NLP classification. 
#
# User is prompted to specify training and testing data, but the training data is used
# only, testing data only asked for in view of a kaggle competition.
# USer is also prompted to give the column names for the text to classify and the classification
# labels. Any easy mod would
# allow dor multiple columns of training text to be given.
#
# Currently the case of a pretrained network is not implemented in this script.
#

# Imports
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Input, Dense, LSTM, GRU, Dropout, Bidirectional, Conv1D, Concatenate
from tensorflow.keras.layers import SpatialDropout1D,GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re


def remove_stops_stem(_str):
    """
    Preprocessing.
    Inputs:
    > _str: a string to preprocess, usually a phrase.
    outputs:
    > string with stopwords removed and the remaining words stemmed using the snowball 
    stemmer
    """
    out = []
    stemmer = SnowballStemmer(language='english')
    stops = set(stopwords.words("english"))
    for w in _str.lower().split():
        w = re.sub("[^a-zA-Z]"," ", w) # delete punctuation.
        if w not in stops:
            out.append(stemmer.stem(w))

    return " ".join(out)

def assemble_model_noConv(activation,loss_fcn,target_length, X_train_pad, X_test_pad, y_train_cat, y_test_cat,
                          embed_dim,
                          vocabulary_size,max_len,
                          lr=0.0, 
                          lr_d=0.0, units=0, spatial_dr=0.0, kernel1=3, dense_units=128, dr=0.1):
    """
    Builds and trains a NN model. Writing this as a function avoids the potential error of retraining the same
    model over and over when tuning parameters or otherwise experimenting.
    Model is roughly described as consisting of
    > an embedding layer for the tokenized phrase,
    > dropout,
    > Bidirectional GRU layer exhibiting some memory and recurrence.
    > A dense layer, followed by another dropout layer
    > A dense output layer.
    The architecture is relatively short but high performing.
    Inputs:
    > Many, including many hyperparameters such as dropout rates and number of hidden layer
    neurons. 
    Also activation function and loss function strings, 
    Also the processed, padded, tokenized test/training data for training and validation.
    """
    model = Sequential()
    model.add(Embedding(vocabulary_size, embed_dim, input_length=max_len))
    model.add(SpatialDropout1D(spatial_dr))
    model.add(Bidirectional(GRU(units)))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dr))
    model.add(Dense(target_length, activation=activation))
    model.compile(loss=loss_fcn, optimizer='adam', metrics=['accuracy'])

    file_path = "best_model.keras"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
    history = model.fit(X_train_pad, y_train_cat, epochs=10, validation_data=(X_test_pad, y_test_cat),
                    batch_size=128, callbacks=[early_stop,check_point])
    model.summary()  # Uncomment to display a description of the architecture.
    return model

def assemble_model_1Conv(activation, loss_fcn, target_length,
        X_train_pad, X_test_pad, y_train_cat, y_test_cat,
        embed_dim, vocabulary_size, max_len,
        lr=0.0, lr_d=0.0, units=0, 
                         spatial_dr=0.0, kernel1=3, dense_units=128, dr=0.1, conv=32):
    """
    Similar to the previous model except with a 1D convolutional layer folllwing the recurrent layer. This has a
    minimal effect on accuracies but adds a little a bit of value, and isn't computationally that expensive to add.
    Importantly, we have to return the sequences from the GRU layer to match the extra dimensinoality of the convolution
    layer."""
    model = Sequential()
    model.add(Embedding(vocabulary_size, embed_dim, input_length=max_len))
    model.add(SpatialDropout1D(spatial_dr))
    model.add(Bidirectional(GRU(units,return_sequences=True))) #Need to return sequences to pass output to a conv layer
    model.add(Conv1D(conv, kernel_size = kernel1, padding='valid', kernel_initializer='he_uniform'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dr))
    model.add(Dense(target_length, activation=activation))
    model.compile(loss=loss_fnc, optimizer='adam', metrics=['accuracy'])

    file_path = "best_model.keras"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
    history = model.fit(X_train_pad, y_train_cat, epochs=10, validation_data=(X_test_pad, y_test_cat),
                    batch_size=128, callbacks=[early_stop,check_point])
    #model.summary()
    return model

def main(df_train, df_test, conv, remove_stops, phrase, toPredict):
    
    """Main function.
    Initializes dataframes based on user input, applies user specified
    preprocessing, and calls the model for training.
    Inputs:
    > Dataframes df_train, df_test of training and testing data.
    > conv: Either the boolean False or a number indicating size of the convolutional layer.
    > remove_stops: Boolean, either False or True. Indicates whether to remove stopwords.
    > phrase, toPredict: Names of columns with the text to classify and the labels. 
    """
    tokenizer = Tokenizer()
    X=df_train[[phrase]]
    y=df_train[[toPredict]]

    if remove_stops:
        # Apply the preprocessing function to the text data
        X[phrase]=X[phrase].apply(remove_stops_stem)
        tokenizer.fit_on_texts(X[phrase].values)
        # max length of 1 row (number of words)
        max_len = max([len(x.split()) for x in df_train[phrase].values])
        # count number of unique words. This is passed into the embedding layer
        vocabulary_size = len(tokenizer.word_index) + 1
        print("Vocab size for tokenizer is: {0}".format(vocabulary_size))
    else:
        # If not removing stopwords, dont call preprocessing function, do everythinf else.
        tokenizer.fit_on_texts(X[phrase].values)
        # max number of words/phrase
        print("Maximum number of words in a phrase in training data is {0}.".format(np.max(df_train['Phrase'].apply(lambda x: len(x.split())))))
        # max length of 1 row (number of words)
        max_len = max([len(x.split()) for x in df_train['Phrase'].values])
        # count number of unique words. This needs to be passed into the network's emnbedding layer
        vocabulary_size = len(tokenizer.word_index) + 1
        print("Vocab size for tokenizer is: {0}".format(vocabulary_size))

    #Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X[phrase].values, y[toPredict].values, test_size=0.1)
    
    #Tokenize
    X_train_tokens = tokenizer.texts_to_sequences(X_train)
    X_test_tokens = tokenizer.texts_to_sequences(X_test)
    
    # ensure every row has same size by padding with zeros
    X_train_pad = pad_sequences(X_train_tokens, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_tokens, maxlen=max_len, padding='post')
    
    # Record max length of phrase for use in architecture.
    max_len = max([len(x.split()) for x in X[phrase].values])
    embed_dim = 300
    
    # Number of classification classes
    num_classes = len(np.unique(y_train))

    if num_classes != 2:
        # If more than 2 classes, make vectors from the class labels and turn the labels 
        # into a matrix of the vectors.
        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)

        target_length = y_train_cat.shape[1] # record length of vectors to pass into model
        print('Original vector size: {}'.format(y_train.shape))
        print('Converted vector size: {}'.format(y_train_cat.shape))

        # Set activation and loss function appropriately for multiclass problem
        activation = 'softmax'
        loss_fcn = 'categorical_crossentropy'
    else:
        # binary classification
        y_test_cat = y_test
        y_train_cat = y_train
        target_length = 2

        # Set activation/loss appropriately
        activation = 'sigmoid'
        loss_fcn = 'binary_crossentropy'

    # Call user decided model structure, either with or without convolutional layer.
    if conv != False:
        model_none = assemble_model_noConv(activation, loss_fcn, target_length,
                X_train_pad, X_test_pad, y_train_cat, y_test_cat,
                embed_dim, vocabulary_size, max_len,
                lr = 1e-3, lr_d = 1e-7, units = 128, 
                spatial_dr = 0.2, kernel1=3, dense_units=128, dr=0.3)
    else:
        model_1conv = assemble_model_1Conv(activation, loss_fcn, target_length,
               X_train_pad, X_test_pad, y_train_cat, y_test_cat,
                embed_dim, vocabulary_size, max_len,
                lr=1e-3, lr_d=1e-7, units=128, spatial_dr=0.3, 
                kernel1=3, dense_units=128, dr=0.1, conv=32)

if __name__ == "__main__":
    
    # Input dataset paths, as well as column titles to predict/train with"
    toPredict = input("Enter column name of class labels: ")
    phrase = input("Enter index of the phrase to classify: ")

    data_name = input("Enter path to training dataset: ")
    test_name = input("Enter path to test data: ")

    try:
        df_train=pd.read_csv(data_name
                             , sep='\t', usecols=[phrase, toPredict])
    except FileNotFoundError or pandas.errors.EmptyDataError:
        data_name = input("Path to data not found or not a csv file. Enter correct file path: ")
        df_train=pd.read_csv(data_name, sep='\t', usecols=[phrase, toPredict])
    try:
        df_test=pd.read_csv(test_name, sep='\t', usecols=[phrase])
    except FileNotFoundError or pandas.errors.EmptyDataError:
        test_name = input("Path to data not found or not a csv file. Enter correct file path: ")
        df_test=pd.read_csv(test_name, sep='\t', usecols=[phrase])
    print("Head of dataframe: \n")
    df_train.head()
    try:
        df_train[toPredict]
    except KeyError:
        toPredict = input("Not a part of dataset, enter a valid index: ")
    try:
        df_train[phrase]
    except KeyError:
        phrase = input("Not a part of dataset, enter a valid index: ")

    conv = False
    remove_stops=False
    stops_ = input('Remove stopwords [Y/N]? ')
    cnv = input('Add optional hidden convolutional layer [Y/N]? ')
    if cnv.lower() == 'y':
        conv = input('Size of convolution layer kernel: ')
    if stops_.lower() == 'y':
        remove_stops = True

    # Call main function. Training the neural nets takes a long time on local machines.
    main(df_train, df_test, conv, remove_stops, phrase, toPredict)
   
