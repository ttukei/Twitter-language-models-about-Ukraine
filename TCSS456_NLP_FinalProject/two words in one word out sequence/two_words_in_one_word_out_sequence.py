import re
import string 
import tweepy
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import nltk
nltk.download('stopwords')
nltk.download('words')
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
  tokens = ' '.join(tokens)
  return tokens

# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
  in_text = seed_text
  # generate a fixed number of words
  for _ in range(n_words):
    # encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # pre-pad sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=max_length, padding= 'pre' )
    # predict probabilities for each word
    yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1)
    # map predicted word index to word
    out_word = ' '
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        out_word = word
        break
    # append to input
    in_text += ' ' + out_word
  return in_text
  
# define the model
def define_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 10, input_length=max_length-1))
  model.add(LSTM(50))
  model.add(Dense(vocab_size, activation= 'softmax' ))
  # compile network
  model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
  # summarize defined model
  model.summary()
  plot_model(model, to_file= 'model.png' , show_shapes=True)
  return model


#get_tweets()
data = clean_doc(load_doc('tweets.txt'))
print(data)
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
# retrieve vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print( 'Vocabulary Size: %d' % vocab_size)
# encode 2 words -> 1 word
sequences = list()
for i in range(2, len(encoded)):
  sequence = encoded[i-2:i+1]
  sequences.append(sequence)
print( 'Total Sequences: %d' % len(sequences))
# pad sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding= 'pre' )
print( 'Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
# define model
model = define_model(vocab_size, max_length)
# fit network
model.fit(X, y, epochs=500, verbose=2)
# evaluate model
print(generate_seq(model, tokenizer, max_length-1, 'actress' , 200))
print(generate_seq(model, tokenizer, max_length-1, 'wonderful' , 200))
print(generate_seq(model, tokenizer, max_length-1, 'thing' , 200))
print(generate_seq(model, tokenizer, max_length-1, 'Ukrainian' , 200))