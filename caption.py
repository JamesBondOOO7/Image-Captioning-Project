#!/usr/bin/env python
# coding: utf-8

# ## Aim : Generate the caption out of the TRAINED MODEL
# - We have already trained the AI Caption Bot
# - The user will give the image
# - We will send it to the server
# - The server will pass the image through ResNet50 model
# - It will generate a (1, 2048) vector
# - This image vector is passed in the model
# - It will generate a caption
# - That caption has to be sent back to the client

# Importing required Libraries
import numpy as np
import keras
import pickle
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add

# Loading the model
model = load_model("./model_29.h5")

# We used ResNet50 Model for encoding the image
model_res = ResNet50(weights="imagenet", input_shape=(224, 224, 3))

# The 2nd last layer of ResNet produces a (1, 2048) vector
# This will generate the required encodings
model_enc = Model(model_res.input, model_res.layers[-2].output)


def preprocess_img(img):
    
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    
    # Converted from 3D to a 4D tensor
    # EX: (224, 224, 3) -- axis=0 --> (1, 224, 224, 3)
    # We can also use reshape
    img = np.expand_dims(img, axis=0)
    
    # We need to feed this image to the ResNet-50 model
    # Normalizing according to how it was trained
    img = preprocess_input(img)
    
    return img

# Encoding an Image
def encode_image(img):
    img = preprocess_img(img)
    # (1 X 2048)
    feature_vector = model_enc.predict(img)
    # 2048
    feature_vector = feature_vector
    return feature_vector

# Loading the dictionaries
with open("./storage/word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)

with open("./storage/idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)


# Predict function
def predict_caption(photo):
    
    # We will feed 2 things to the model
    # 1. Feed the image vector (2048,)
    # 2. Provide the start sequence "startseq" (<s>)
    
    inp_text = "startseq"
    max_len = 35
    for i in range(max_len):
        
        sequence = [word_to_idx[w] for w in inp_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
        
        ypred = model.predict([photo,sequence])
        
        # Greedy Sampling : Word with max probability always
        ypred = ypred.argmax()
        
        # retreiving the word
        word = idx_to_word[ypred]
        
        # adding it to the sequence
        inp_text += (' ' + word)
        
        # If <e>/end sequence is encountered
        if word == "endseq":
            break
            
    # removing <s> and <e>
    final_caption = inp_text.split(' ')[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


def caption_for_the_image(image):
    # obtaining encodings for the image
    enc = encode_image(image)
    caption = predict_caption(enc)
    
    return caption



