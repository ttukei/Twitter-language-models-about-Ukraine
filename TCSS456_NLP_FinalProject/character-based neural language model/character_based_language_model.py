import tweepy
import re
import string
from collections import Counter
import nltk
nltk.download('stopwords')
nltk.download('words')
from nltk.corpus import words
from nltk.corpus import stopwords
# Get tweets that contain the hashtag #ukraine
# -is:retweet means I don't want retweets
# lang:en is asking for the tweets to be in english
def get_tweets():
  client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAFHifQEAAAAAl3Rcv1rz5YuBqB7GtR88QfLn82w%3DrS4Na5OJgoaLABrfeNgt0DvKKBdPL83BlqUVlx2c3c6bOOllYq')
  query = '#ukraine -is:retweet lang:en'
  tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100)
  tweet_file = open("tweets.txt", "w")
  for tweet in tweets.data:
      tweet_file.write(tweet.text)

#load doc into memory
def load_doc(filename):
  #open the file as read only
  file = open(filename,'r')
  #read all text
  text = file.read()
  #close the file
  file.close()
  return text

# turn a doc into clean tokens
def clean_doc(doc):
  # split into tokens by white space
  tokens = doc.split()
  # prepare regex for char filtering
  re_punc = re.compile( '[%s]' % re.escape(string.punctuation))
  # remove punctuation from each word
  tokens = [re_punc.sub('' , w) for w in tokens]
  # remove remaining tokens that are not alphabetic
  tokens = [word for word in tokens if word.isalpha()]
  # filter out stop words
  stop_words = set(stopwords.words( 'english' ))
  english_words = set(words.words())
  tokens = [w for w in tokens if (not w in stop_words) and (w in english_words)]
  # filter out short tokens
  tokens = [word for word in tokens if len(word) > 1]
  tokens = ' '.join(tokens)
  return tokens

# Commented out IPython magic to ensure Python compatibility.
# %ls
# %cd drive/MyDrive/TCSS456/LanguageModel1/

# Commented out IPython magic to ensure Python compatibility.
# %pwd
#get_tweets()
raw_text = clean_doc(load_doc('tweets.txt'))
print(raw_text)

#clean text
tokens = raw_text.split()
raw_text = ' '.join(tokens)

#organize into sequences of characters
length = 10 
sequences = list()
for i in range(length, len(raw_text)):
  #select sequence of tokens 
  seq = raw_text[i-length:i+1]
  #store 
  sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

#save tokens to file, one dialog per line 
def save_doc(lines, filename):
  data = '\n'.join(lines)
  file = open(filename, 'w')
  file.write(data)
  file.close()

#save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences,out_filename)

#load sequences
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

#encode sequences
chars = sorted(list(set(raw_text)))
mapping = dict((c,i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
  #integer encode line
  encoded_seq = [mapping[char] for char in line]
  #store 
  sequences.append(encoded_seq)

#vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

#Note: import numpy to use array object 
from numpy import array
#Note: Keras is now fully intregrated into Tensorflow
from tensorflow.keras.utils import to_categorical

#split inputs and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes = vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes = vocab_size)

#Note: make sure to import Sequential, LSTM, Dense from Keras
from keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

#Note: Keras is now fully intregrated into Tensorflow
from tensorflow.keras.utils import plot_model

#define model
def define_model(X):
  model = Sequential()
  model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
  model.add(Dense(vocab_size, activation = 'softmax'))

  #compile model
  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

  #summarize defined model
  model.summary()
  plot_model(model, to_file = 'model.png', show_shapes = True)
  return model

#call to define model
model = define_model(X)

#fit model
model.fit(X, y, epochs = 100, verbose = 2)

#save the model to file
model.save('model.h5')

#Note: import pickle to use dump
import pickle

#Note: use the pickle.dump to save the mapping
pickle.dump(mapping, open('mapping.pk1', 'wb'))

#Note: Keras is now fully intregrated into Tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# generate a sequence of characters with a language model 
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
  in_text = seed_text
  #generate a fixed number of characters 
  for _ in range (n_chars):
    #encode the characters as integers 
    encoded = [mapping[char] for char in in_text]
    
    #truncate sequences to a fixed length 
    encoded = pad_sequences([encoded], maxlen = seq_length, truncating = 'pre')
    
    # one hot encode
    encoded = to_categorical(encoded, num_classes=len(mapping))
 
    # predict character
    #yhat = model.predict(encoded, verbose=0)[0]
    yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1)[0]
    
    #reverse map integer to character
    out_char = ''
    for char,index in mapping.items():
      #print(char,index,yhat)
      if index == yhat:
        out_char = char
        break
    #append to input
    in_text += out_char
  
  return in_text

#load the model
model = load_model('model.h5')

#load mapping
mapping = pickle.load(open('mapping.pk1', 'rb'))

#test start of rhyme
print(generate_seq(model, mapping, 10,  'ukraine' , 1000))