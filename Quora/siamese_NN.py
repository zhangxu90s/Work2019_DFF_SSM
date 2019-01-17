# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Input, TimeDistributed, Lambda,LSTM,GlobalMaxPooling1D,Dense,Activation,subtract,Add,multiply,concatenate,merge,Dropout,BatchNormalization
from keras.models import Model,Sequential
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam,Adadelta
from keras.preprocessing.sequence import pad_sequences
from multi_perspective import MultiPerspective,PredictLayer
import data_helper


input_dim = data_helper.MAX_SEQUENCE_LENGTH
emb_dim = data_helper.EMB_DIM
model_path = './model/siameselstm.hdf5'
tensorboard_path = './model/ensembling'

embedding_matrix = data_helper.load_pickle('embedding_matrix.pkl')

embedding_layer = Embedding(embedding_matrix.shape[0],
                            emb_dim,
                            weights=[embedding_matrix],
                            input_length=input_dim,
                            trainable=False)


def base_network(input_shape):
    input = Input(shape=input_shape)

    x = embedding_layer(input)
    x = TimeDistributed(Dense(300, activation='relu'))(x)
    x = Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,))(x)
   
    y = embedding_layer(input)
    y = TimeDistributed(Dense(300, activation='relu'))(y)
    y = Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,))(y)

    multi_memory_TD = Add()([x,y])

    p = embedding_layer(input)
    p = LSTM(300, return_sequences=False, dropout=0.1, recurrent_dropout=0.1,name='f_input')(p)

    q = embedding_layer(input)
    q = LSTM(300, return_sequences=False, dropout=0.1, recurrent_dropout=0.1,name='re_input')(q)

    multi_memory_lstm = Add()([p,q])

    multi_memory = concatenate([multi_memory_lstm,multi_memory_TD])


    return Model(input, multi_memory, name='DFF')

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision


def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    recall = c1 / c3

    return recall

margin = 0.6
theta = lambda t: (K.sign(t)+1.)/2.
nb_classes = 10
def one_loss(y_true, y_pred):
    loss1 = mse_loss(y_true, y_pred)
    #one_hot
    loss2 = mse_loss(K.ones_like(y_pred)/nb_classes, y_pred)
    return 0.9*loss1+0.1*loss2

def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))
def mse_onehot(y_true, y_pred):
    return mse_loss(K.ones_like(y_pred)/nb_classes, y_pred)
def loss(y_true, y_pred):
    return - (1 - theta(y_true - margin) * theta(y_pred - margin) 
              - theta(1 - margin - y_true) * theta(1 - margin - y_pred)
              ) * (y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8))
def myloss(y_true, y_pred, e=0.15):
    loss1 = mse_loss(y_true, y_pred)
    #one_hot
    loss2 = mse_loss(K.ones_like(y_pred)/nb_classes, y_pred)
    loss3 = loss(y_true, y_pred)
    return (1-2*e)*loss1 + e*loss2 + e*loss3

def siamese_model():
    input_shape = (input_dim,)
    
    # Creating Encoder

    base_net = base_network(input_shape)

    
    # Creating Encoder layer for frist Sentence
    input_q1 = Input(shape=input_shape, dtype='int32', name='sequence1')
    processed_q1 = base_net([input_q1])
    
    
    # Creating Encoder layer for Second Sentence
    input_q2 = Input(shape=input_shape, dtype='int32', name='sequence2')
    processed_q2 = base_net([input_q2])
    
    #doing matching
    abs_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([processed_q1,processed_q2])
    cos_diff = Lambda(lambda x: K.cos(x[0] - x[1]))([processed_q1,processed_q2])
    multi_diff = multiply([processed_q1,processed_q2])
    all_diff = concatenate([abs_diff,cos_diff,multi_diff])

    #DNN
    all_diff = Dropout(0.5)(all_diff)
    similarity = Dense(600)(all_diff)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('relu')(similarity)
    similarity = Dense(600)(similarity)
    similarity = Dropout(0.5)(similarity)
    similarity = Activation('relu')(similarity)
    similarity = Dense(1)(similarity)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('sigmoid')(similarity)
    model = Model([input_q1, input_q2], [similarity])
    #loss:binary_crossentropy_loss;optimizer:adm,Adadelta
    adm = Adam(lr=0.002)
    model.compile(loss=myloss, optimizer=adm, metrics=['accuracy'])
    return model


def train():
    
    data = data_helper.load_pickle('model_data.pkl')

    train_q1 = data['train_q1']
    train_q2 = data['train_q2']
    train_y = data['train_label']

    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_y = data['dev_label']
    
    test_q1 = data['test_q1']
    test_q2 = data['test_q2']
    test_y = data['test_label']
    
    model = siamese_model()
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir=tensorboard_path)    
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max')
    callbackslist = [checkpoint, tensorboard,earlystopping,reduce_lr]

    model.fit([train_q1, train_q2], train_y,
              batch_size=512,
              epochs=200,
              validation_data=([dev_q1, dev_q2], dev_y),
              callbacks=callbackslist)
    '''
    ## Add graphs here
    import matplotlib.pyplot as plt

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])   
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss','train accuracy', 'val accuracy','train precision', 'val precision','train recall', 'val recall','train f1_score', 'val f1_score'], loc=3,
               bbox_to_anchor=(1.05,0),borderaxespad=0)
    pic = plt.gcf()
    pic.savefig ('pic.eps',format = 'eps',dpi=1000)
    plt.show()
    '''
    loss, accuracy= model.evaluate([test_q1, test_q2],test_y,verbose=1,batch_size=256)
    print("Test best model =loss: %.4f, accuracy:%.4f" % (loss, accuracy))

if __name__ == '__main__':
    train()
    
 
