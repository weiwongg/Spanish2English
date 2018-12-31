# Spanish2English

Implementation of Neural Machine Translation from Spanish to English 

Model: Encoder and Decoder model with Attention

Dataset: tf.keras.utils.get_file('spa-eng.zip', origin = 'http://download.tensorflow.org/data/spa-eng.zip',extract = True)

How to train model in Google Colaboratory 

step 1: upload downloaded folder to Google Drive

step 2: open Colaboratory and do not forget to change running type to GPU

step 3: mount your google drive to Colaboratory

--run

from google.colab import drive

drive.mount('/content/gdrive')


step 4: copy train.py in the running block

step 5: run code

step 6: download trained model from Colaboratory 

step 7: run predict.py
