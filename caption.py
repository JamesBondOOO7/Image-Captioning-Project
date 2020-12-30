import numpy as np
import keras
import pickle
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences

model = load_model("./model_29.h5")

model_res = ResNet50(weights="imagenet", input_shape=(224, 224, 3))

model_enc = Model(model_res.input, model_res.layers[-2].output)


def preprocess_img(img):
    
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    
    img = np.expand_dims(img, axis=0)
    
    img = preprocess_input(img)
    
    return img

def encode_image(img):
    img = preprocess_img(img)

    feature_vector = model_enc.predict(img)
    
    feature_vector = feature_vector
    return feature_vector


with open("./storage/word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)

with open("./storage/idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)


def predict_caption(photo):
    
    inp_text = "startseq"
    max_len = 35
    for i in range(max_len):
        
        sequence = [word_to_idx[w] for w in inp_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
        
        ypred = model.predict([photo,sequence])
        
        ypred = ypred.argmax()
        
        word = idx_to_word[ypred]
        
        inp_text += (' ' + word)
        
        if word == "endseq":
            break

    final_caption = inp_text.split(' ')[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption

def caption_for_the_image(image):

    enc = encode_image(image)
    caption = predict_caption(enc)
    
    return caption
