import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time


#Additional libraries I will use
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy as np
import json
import pickle
import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model

#Only used if it is not download
#nltk.download()
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

#Here we will start preparing the model and training it for use in the chatbot

words=[]
categories = []
groups = []

excluded = ['!', '?', ',', '.']

#We use the JSON file for the words that we want to rely on in the chatbot
data_file = open('data.json').read()
parts = json.loads(data_file)

#We start to prepare the words
for part in parts['parts']:
    for style in part['styles']:
       
        word = nltk.word_tokenize(style)
        words.extend(word)
      
        groups.append((word, part['tag']))
      
        if part['tag'] not in categories:
            categories.append(part['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in excluded]
words = sorted(list(set(words)))
categories = sorted(list(set(categories)))


#We create files that will help us
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(categories,open('categories.pkl','wb'))


training = []
output_empty = [0] * len(categories)
# bag of words we will use 
for group in groups:
    bag = []
    style_words = group[0]
    style_words = [lemmatizer.lemmatize(word.lower()) for word in style_words]
   
    for word in words:
        bag.append(1) if word in style_words else bag.append(0)
        
    output_row = list(output_empty)
    output_row[categories.index(group[1])] = 1
    training.append([bag, output_row])

    
    
random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])


#The model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

#save model
model.save('training_model.h5', hist)


model = load_model('training_model.h5')
parts = json.loads(open('data.json').read())
words = pickle.load(open('words.pkl','rb'))
categories = pickle.load(open('categories.pkl','rb'))


def pre_sentence(sentence):
 
    sen_words = nltk.word_tokenize(sentence)
    sen_words = [lemmatizer.lemmatize(word.lower()) for word in sen_words]
    return sen_words


def bag_words(sentence, words, show_details=True):
    sen_words = pre_sentence(sentence)
    bag = [0]*len(words)  
    for s in sen_words:
        for i,word in enumerate(words):
            if word == s: 
                bag[i] = 1
                if show_details:
                    print ("we found in bag: %s" % word)
    return(np.array(bag))


def predict_categories(sentence):
    p = bag_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"part": categories[r[0]], "probability": str(r[1])})
    return return_list


def getanswer(ints, data):
    tag = ints[0]['part']
    list_parts = data['parts']
    for i in list_parts:
        if(i['tag']== tag):
            result = random.choice(i['answers'])
            break
    return result


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        response = getanswer(predict_categories(text), parts)
        output.append(response)

    return SimpleText(dict(text=output))
