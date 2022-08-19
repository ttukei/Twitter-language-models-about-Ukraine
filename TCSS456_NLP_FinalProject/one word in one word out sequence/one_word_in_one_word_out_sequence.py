# One word in one word out sequence
import tweepy
import numpy as np
from numpy import array
import string
import re
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from collections import Counter
import nltk
nltk.download('words')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import words

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
  tokens = tokens[:200]
  print(tokens)
  tokens = ' '.join(tokens)
  return tokens

# generate a sequence from the model
def generate_seq(model, tokenizer, seed_text, n_words):
  in_text, result = seed_text, seed_text
  # generate a fixed number of words
  for _ in range(n_words):
    # encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    encoded = array(encoded)
    # predict a word in the vocabulary
    yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1)[0]
    # map predicted word index to word
    out_word =' '
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        out_word = word
        break
    # append to input
    in_text, result = out_word, result + ' ' + out_word
  return result

# define the model
def define_model(vocab_size):
  model = Sequential()
  model.add(Embedding(vocab_size, 10, input_length=1))
  model.add(LSTM(50))
  model.add(Dense(vocab_size, activation= 'softmax' ))
  # compile network
  model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
  # summarize defined model
  model.summary()
  plot_model(model, to_file= 'model.png' , show_shapes=True)
  return model

#get_tweets()
data = clean_doc(load_doc("tweets.txt"))
# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print( 'Vocabulary Size: %d' % vocab_size)
# create word -> word sequences
sequences = list()
for i in range(1, len(encoded)):
  sequence = encoded[i-1:i+1]
  sequences.append(sequence)
print( 'Total Sequences: %d' % len(sequences))
# split into X and y elements
sequences = array(sequences)
X, y = sequences[:,0],sequences[:,1]
# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)
# define model
model = define_model(vocab_size)
# fit network
model.fit(X, y, epochs=500, verbose=2)
# evaluate
print(generate_seq(model, tokenizer, 'thing' , 200))