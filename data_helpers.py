#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:22:01 2018

@author: huangwei
"""

import tensorflow as tf
import os
import unicodedata
import numpy as np
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
def tokenize(s):
    for char in [unicode_to_ascii('?'), unicode_to_ascii('.'), unicode_to_ascii('!'), unicode_to_ascii(','), unicode_to_ascii('¡'), unicode_to_ascii('¿')]:
        s = s.replace(char, ' '+char+' ')
        s = ' '.join(s.split())
    s = s.split(' ')
    return s
#Loading raw data
class Data():
    def __init__(self):
        self.load_data()
        self.transform()
        self.token_container
        self.num_container
        self.eng2int
        self.spa2int
        self.int2eng
        self.int2spa
       
        
    def load_data(self):
        print('========Loading and preprocessing the data...========')
        path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin = 'http://download.tensorflow.org/data/spa-eng.zip',extract = True)
        path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"
        sentences_container = []
        with open(path_to_file, 'r', encoding='UTF-8') as f:
            while True:
                text_line = f.readline()
                #delete the space and \n in the begining and the end of each line
                text_line = text_line.strip(' ')
                text_line = text_line.strip('\n').lower()
                if text_line:
                    sentences_container.append(text_line.split('\t'))
                else:
                    break
        #Sampling the first 30000 sentences
        sentences_container = sentences_container[:30000]
        
        #convert Unicode sentences into the ASCII format
        sentences_container = [[unicode_to_ascii(sentence[0]), unicode_to_ascii(sentence[1])] for sentence in sentences_container]
        #Tokenize each sentence
        self.token_container = [[tokenize(sentence[0]), tokenize(sentence[1])] for sentence in sentences_container]
        #adding <SOS> and <EOS>
        for token in self.token_container:
            token[0].append('<EOS>')
            token[1].append('<EOS>')
            token[0].insert(0,'<SOS>')
            token[1].insert(0,'<SOS>')
        #Statistic information for each word 
        self.eng2int = {'<pad>': 0}
        engindex = 1
        self.spa2int = {'<pad>': 0}
        spaindex = 1
        for token in self.token_container:
            for word in token[0]:
                if word in self.eng2int.keys():
                    pass
                else:
                    self.eng2int[word] = engindex 
                    engindex = engindex + 1
            for word in token[1]:
                if word in self.spa2int.keys():
                    pass
                else:
                    self.spa2int[word] = spaindex 
                    spaindex = spaindex + 1
        print("Sumarry Information:")
        print("The total number of English word is {}".format(len(self.eng2int) - 3))
        print("The total number of Spanish word is {}".format(len(self.spa2int) - 3))
        print("The maximum length of English sentence is {}".format(np.max([len(token[0]) for token in self.token_container])))
        print("The maximum length of Spanish sentence is {}".format(np.max([len(token[1]) for token in self.token_container])))
        self.int2eng = list(self.eng2int.keys())
        self.int2spa = list(self.spa2int.keys())
    def transform(self):
        self.num_container = []
        for token in self.token_container:
            out1 = []
            out2 = []
            for word in token[0]:
                out1.append(self.eng2int[word])
            for word in token[1]:
                out2.append(self.spa2int[word])
            
            self.num_container.append([out1,out2])
        
#Create batch 
class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []              


        
def getBatches(data, batch_size):
    np.random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    def createBatch(samples):
        batch = Batch()
        batch.encoder_inputs_length = [len(sample[1]) for sample in samples]
        batch.decoder_targets_length = [len(sample[0]) for sample in samples]
        
        max_source_length = max(batch.encoder_inputs_length)
        max_target_length = max(batch.decoder_targets_length)
        
        for sample in samples:
            source = list(sample[1])
            pad = [0] * (max_source_length - len(source))
            batch.encoder_inputs.append(source + pad)
            
            target = sample[0]
            pad = [0] * (max_target_length - len(target))
            batch.decoder_targets.append(target + pad)
        
        return batch

    for samples in genNextSamples():
        batch = createBatch(samples)
        batches.append(batch)
    return batches
def createBatch(samples):
    batch = Batch()
    batch.encoder_inputs_length = [len(sample[1]) for sample in samples]
    batch.decoder_targets_length = [len(sample[0]) for sample in samples]
    
    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)
    
    for sample in samples:
        source = list(sample[1])
        pad = [0] * (max_source_length - len(source))
        batch.encoder_inputs.append(source + pad)
        
        target = sample[0]
        pad = [0] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)

    return batch


def sentence2batch(sentence, spa2int):
    if sentence == '':
        return None
    sentence = sentence.lower()
    sentence = unicode_to_ascii(sentence)
    tokens = tokenize(sentence) 
    tokens.append('<EOS>')
    tokens.insert(0,'<SOS>')
    wordIds = []
    start = spa2int['<SOS>']
    end = spa2int['<EOS>']
    for token in tokens:
        wordIds.append(spa2int[token]) 
    batch = createBatch([[[start, start, start, start, start, start, start, start, start, end], wordIds]])
    return batch, tokens


def ids_to_words(predict_ids, int2eng):
    for single_predict in predict_ids:
        predict_list = np.ndarray.tolist(single_predict)
        predict_seq = [int2eng[idx] for idx in predict_list]
        result = " ".join(predict_seq)
        result = result.encode("utf-8")
        print(result.decode('utf-8'))
        return predict_seq
