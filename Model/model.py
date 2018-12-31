#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 21:32:12 2018

@author: huangwei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:05:20 2018

@author: huangwei
"""

import tensorflow as tf


class Seq2SeqModel():
    def __init__(self, rnn_size, num_layers, embedding_size, eng2int, spa2int, learning_rate, use_attention, max_gradient_norm=5.0):
        self.learing_rate = learning_rate
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.eng_vocab_size = len(eng2int)
        self.spa_vocab_size = len(spa2int)
        self.use_attention = use_attention
        self.max_gradient_norm = max_gradient_norm
        self.eng2int = eng2int
        self.spa2int = spa2int
        self.build_model()

    def _create_rnn_cell(self):
        def single_rnn_cell():
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            #Implementing dropout
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell
        #wrapping all the cells
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_model(self):
        print('Building model... ...')
        #=================================1, Defining the placeholder
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')

        #=================================2, Defining Encode
        with tf.variable_scope('encoder'):
            #using LSTM to encode input sentence.
            encoder_cell = self._create_rnn_cell()
            #define word embedding, we could also use existing word embedding
            spa_embedding = tf.get_variable('spa_embedding', [self.spa_vocab_size, self.embedding_size])
            encoder_inputs_embedded = tf.nn.embedding_lookup(spa_embedding, self.encoder_inputs)
            # encoder_outputs = [batch_size, encoder_inputs_length, rnn_size]
            # final_encoder_state = [batch_size, rnn_szie]
            encoder_outputs, final_encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                               sequence_length=self.encoder_inputs_length,
                                                               dtype=tf.float32)

        # =================================3, Defining Decode
        with tf.variable_scope('decoder'):
            encoder_inputs_length = self.encoder_inputs_length
            #define word embedding, we could also use existing word embedding
            eng_embedding = tf.get_variable('eng_embedding', [self.eng_vocab_size, self.embedding_size])
            #Standard way to implementing attention model
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs,
                                                                     memory_sequence_length=encoder_inputs_length)
          
            decoder_cell = self._create_rnn_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.rnn_size, alignment_history = True, name='Attention_Wrapper')
            #If we use beam search, we have to change the batch size
            #use final_encoder_state as decoder_initial_state
            decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(cell_state=final_encoder_state)
            output_layer = tf.layers.Dense(self.eng_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            #----------Training decoding part
            # decoder_inputs_embedded的shape为[batch_size, decoder_targets_length, embedding_size]
            #adding the 'SOS' at the begining of sentence and delete the final word 
            ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.eng2int['<SOS>']), ending], 1)
            decoder_inputs_embedded = tf.nn.embedding_lookup(eng_embedding, decoder_input)
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                sequence_length=self.decoder_targets_length,
                                                                time_major=False, name='training_helper')
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                               initial_state=decoder_initial_state, output_layer=output_layer)
            train_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                      impute_finished=True,
                                                                maximum_iterations=self.max_target_sequence_length)
    
            self.decoder_logits_train = tf.identity(train_decoder_outputs.rnn_output)
            self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
  
            self.train_loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                         targets=self.decoder_targets, weights=self.mask)


            optimizer = tf.train.AdamOptimizer(self.learing_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.train_loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
            #-----------Inference decoding part 
            start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.eng2int['<SOS>']
            end_token = self.eng2int['<EOS>']
            decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=eng_embedding,
                                                                       start_tokens=start_tokens, end_token=end_token)
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                                initial_state=decoder_initial_state,
                                                                output_layer=output_layer)
            inference_decoder_outputs, inference_decoder_state, _= tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                      impute_finished=True,
                                                                maximum_iterations=self.max_target_sequence_length)
            
            self.decoder_logits_inference = tf.identity(inference_decoder_outputs.rnn_output)
            
            self.decoder_alignments = tf.identity(inference_decoder_state.alignment_history.stack())
            
            self.padding_decoder_logits_inference = tf.pad(self.decoder_logits_inference, [[0, 0], [0, self.max_target_sequence_length - tf.shape(self.decoder_logits_inference)[1]], [0, 0]], "CONSTANT")
            
            self.decoder_predict_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_pred_inference')

            self.inference_loss = tf.contrib.seq2seq.sequence_loss(logits=self.padding_decoder_logits_inference,
                                                         targets=self.decoder_targets, weights=self.mask)

            # Inference summary for the current batch_loss
            tf.summary.scalar('inference_loss', self.inference_loss)
            self.summary_op = tf.summary.merge_all()
        # =================================4, Save the model
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        _, loss = sess.run([self.train_op, self.train_loss], feed_dict=feed_dict)
        return loss
                      


    def evaluation(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        loss, summary = sess.run([self.inference_loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def translation(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        predict, alignments = sess.run([self.decoder_predict_inference, self.decoder_alignments], feed_dict=feed_dict)
        return predict, alignments
