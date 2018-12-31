#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:39:19 2018

@author: huangwei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:48:34 2018

@author: huangwei
"""

import tensorflow as tf
from data_helpers import *
from model import Seq2SeqModel
import sys
import numpy as np
import matplotlib.pyplot as plt


rnn_size = 1024
num_layers = 1
embedding_size = 256
learning_rate = 0.001
model_dir = "./model/"
graph_dir = "./graph/"
model_name = "translation.ckpt"
data = Data()
eng2int = data.eng2int
int2eng = data.int2eng
spa2int = data.spa2int
int2spa = data.int2spa

def plot_attention(attention_map, input_tags=None, output_tags=None, idx=None):
    attn_len = len(attention_map)
    
    # Plot the attention_map
    plt.clf()
    f = plt.figure(figsize=(15, 10))
    ax = f.add_subplot(1, 1, 1)
    
    # Add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)
    
    # Add labels
    ax.set_yticks(range(attn_len))
    if output_tags is not None:
        ax.set_yticklabels(output_tags[:attn_len])
    
    ax.set_xticks(range(attn_len))
    if input_tags is not None:
        ax.set_xticklabels(input_tags[:attn_len], rotation=45)
    
    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()
    
    # plt.show()
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + "/alignment-{}.png".format(idx), bbox_inches='tight')
    plt.close()

with tf.Session() as sess:
    model = Seq2SeqModel(rnn_size, num_layers, embedding_size, eng2int, spa2int, learning_rate, use_attention = True, max_gradient_norm=5.0)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(model_dir))
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
    
    plot_attention(alignments1[:,0,:], input_tags1, output_tags1, idx=1)
    plot_attention(alignments2[:,0,:], input_tags2, output_tags2, idx=2)
       
