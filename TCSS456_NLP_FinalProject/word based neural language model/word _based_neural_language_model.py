import string
import re
import tweepy
import nltk
nltk.download('stopwords')
nltk.download('words')
from nltk.corpus import stopwords
from nltk.corpus import words
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import numpy as np
from numpy import array

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
  # filter out stop words and non english words
  stop_words = set(stopwords.words( 'english' ))
  english_words = set(words.words())
  tokens = [w for w in tokens if (not w in stop_words) and (w in english_words)]
  # filter out short tokens
  tokens = [word for word in tokens if len(word) > 1]
  # tokens = tokens[:200]
  # tokens = ' '.join(tokens)
  return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
  data = '\n'.join(lines)
  file = open(filename, 'w' )
  file.write(data)
  file.close()

# define the model
def define_model(vocab_size, seq_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 50, input_length=seq_length))
  model.add(LSTM(100, return_sequences=True))
  model.add(LSTM(100))
  model.add(Dense(100, activation= 'relu' ))
  model.add(Dense(vocab_size, activation= 'softmax' ))
  # compile network
  model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
  # summarize defined model
  model.summary()
  plot_model(model, to_file= 'model.png' , show_shapes=True)
  return model

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
  result = list()
  in_text = seed_text
  # generate a fixed number of words
  for _ in range(n_words):
    # encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating= 'pre' )
    # predict probabilities for each word
    yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1)[0]
    # map predicted word index to word
    out_word = ' '
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        out_word = word
        break
      # append to input
      in_text += ' ' + out_word
    result.append(out_word)
  return ' '.join(result)

# get tweets
#get_tweets()
# load document
in_filename = 'tweets.txt'
doc = load_doc(in_filename)
print(doc[:200])
# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print( 'Total Tokens: %d' % len(tokens))
print( 'Unique Tokens: %d' % len(set(tokens)))
# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
  # select sequence of tokens
  seq = tokens[i-length:i]
  # convert into a line
  line = ' '.join(seq)
  # store
  sequences.append(line)
print( 'Total Sequences: %d' % len(sequences))
# save sequences to file
out_filename = 'tweet_sequences.txt'
save_doc(sequences, out_filename)
# load
in_filename = 'tweet_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split( '\n')
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]
# define model
model = define_model(vocab_size, seq_length)
# fit model
model.fit(X, y, batch_size=128, epochs=100)
# save the model to file
model.save( 'model.h5' )
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl' , 'wb' ))
# load cleaned text sequences
in_filename = 'tweet_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split( '\n' )
seq_length = len(lines[0].split()) - 1
# load the model
model = load_model( 'model.h5' )
# load the tokenizer
tokenizer = load(open( 'tokenizer.pkl' , 'rb' ))
# select a seed text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n' )
# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 500)
print(generated)
