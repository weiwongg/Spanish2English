#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 13:33:39 2018

@author: huangwei
"""
#Connecting to google drive
#from google.colab import drive
#drive.mount('/content/gdrive')
import tensorflow as tf
import sys
sys.path.append('gdrive/My Drive/Spanish2English')
from data_helpers import *
from model import Seq2SeqModel
from tqdm import tqdm
import math
import os
import random

rnn_size = 1024
num_layers = 1
embedding_size = 256
learning_rate = 0.001
model_dir = "./model/"
numEpochs = 15
batch_size = 64
steps_per_checkpoin = 500
model_name = "translation.ckpt"
data = Data()
eng2int = data.eng2int
spa2int = data.spa2int
int2eng = data.int2eng
int2spa = data.int2spa
Samples = data.num_container
#Spliting the dataset into training set validation set and testing set
random.shuffle(Samples)
train_samples = Samples[:23000]
valid_samples = Samples[23000:24000]
test_samples = Samples[24000:]

with tf.Session() as sess:
    model = Seq2SeqModel(rnn_size, num_layers, embedding_size, eng2int, spa2int, learning_rate, use_attention = True, max_gradient_norm=5.0)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())
    current_step = 0
    summary_writer = tf.summary.FileWriter(model_dir, graph=sess.graph)
    for e in range(numEpochs):
        random.shuffle(train_samples)
        print("----- Epoch {}/{} -----".format(e + 1, numEpochs))
        batches = getBatches(train_samples, batch_size)
        for nextBatch in tqdm(batches, desc="Training"):
            train_loss = model.train(sess, nextBatch)
            current_step += 1
            if current_step % steps_per_checkpoin == 0:
                random.shuffle(valid_samples)
                valid_batches = getBatches(valid_samples, batch_size)
                valid_loss, summary = model.evaluation(sess, valid_batches[0])
                tqdm.write("\n----- Step {0}-- Validation Loss --{1} --Training Loss --{2}".format(current_step, valid_loss, train_loss))
                sentence1 = 'esta es mi vida.'
                sentence2 = '¿todavia están en casa?'  
                batch1, input_tags1 = sentence2batch(sentence1, spa2int)
                batch2, input_tags2= sentence2batch(sentence2, spa2int)
                predicted_ids1, alignments1= model.translation(sess, batch1)
                predicted_ids2, alignments2= model.translation(sess, batch2)
                # print(predicted_ids)
                print("\n The translation of the first sentence is:\n")
                output_tags1 = ids_to_words(predicted_ids1, int2eng)
                print("\n The translation of the seoncd sentence is:\n")
                output_tags2 = ids_to_words(predicted_ids2, int2eng)
                summary_writer.add_summary(summary, current_step)
    #Saving the final model:
    checkpoint_path = os.path.join(model_dir, model_name)
    model.saver.save(sess, checkpoint_path, global_step=current_step)

    print("=====Starting Testing=====")
    test_batches = getBatches(test_samples, 6000)
    loss, _ = model.evaluation(sess, test_batches[0])
    print('Test loss is : {0}.'.format(loss))
    print('End of Testing.')
