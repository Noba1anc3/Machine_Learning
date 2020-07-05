# Natural Language Processing

## Text to Vector (tokenization)

- Split into word, change every word to vector
- Split into character, change every character to vector
- Extract n-gram of word/character, convert n-gram to vector

### n-gram

n-gram is a set of N-continuous words/characters extracted from a sentence.

'The cat sat on the mat'

- 2-gram : {"The", "The cat", "cat", "cat sat", "sat",
  "sat on", "on", "on the", "the", "the mat", "mat"}  
- 3-gram : {"The", "The cat", "cat", "cat sat", "The cat sat",
  "sat", "sat on", "on", "cat sat on", "on the", "the",
  "sat on the", "the mat", "mat", "on the mat"}  

Bag-of-word is a tokenization which don't save word sequence, abandon overall structure of sentence.  

Usually used for shallow NLP model (like logistic regression, random forest).

### One-hot Encoding

Relate a word to a unique integer index, convert index *i* into a N-vector (binary), only *i*th element is 1.  

Keras built-in Function

- Delete special character
- Only consider N characters most common in the dataset

#### Word-Level

```python
import numpy as np

samples = ['the cat sat on the mat', 'the dog ate my homework']

token_index = {}
for sample in samples:
	for word in sample.split():
		if word not in token_index:
			token_index[word] = len(token_index) + 1

max_length = 10

result = np.zeros(shape=(len(samples), 
                         max_length, 
                         max(token_index.value()) + 1
                        )
                 )
for i, sample in enumerate(samples):
	for j, word in list(enumerate(sample.split()))[:max_length]:
		index = token_index.get(word)
		results[i, j, index] = 1.
```

#### Character Level

```python
import string

samples = ['the cat sat on the mat', 'the dog ate my homework']
characters = string.printable
token_index = dict(zip(range(1, len(characters) + 1), characters))

max_length = 50
results = np.zeros((len(samples), 
                    max_length, 
                    max(token_index.keys()) + 1
                   ))
for i, sample in enumerate(samples):
	for j, character in enumerate(sample):
		index = token_index.get(character)
		results[i, j, index] = 1.
```

#### Keras Built-in Function

```python
from keras.preprocessing.text import Tokenizer

samples = ['the cat sat on the mat', 'the dog ate my homework']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
```

#### One-hot hashing trick

It do not allocate a index and save it in a dictionary for every word.  

It hash encode every word to length-fixed vector.  

Advantage:

- Get avoid of maintain a word index, so as to save memory
- Online encoding

Disadvantage

- Hash Collision (depends on dim of hash space and num of words)

```python
samples = ['the cat sat on the mat', 'the dog ate my homework']

dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
	for j, word in list(enumerate(sample.split()))[:max_length]:
		index = abs(hash(word)) % dimensionality
		results[i, j, index] = 1.
```

### Word Embedding

One-hot Encoding

- Binary
- Sparse
- High-dimensionality
- Hard-Encoding

Word Embedding

- Float
- Dense
- Low-dimensionality
- Learned

Usually 256, 512 or 1024 dims.  



To get word embedding

- While completing the main task (doc classify, emotion predict), learn embedding simultaneously.  

  At the beginning, vector is random. Then, learn the embedding in the same way of weights in NN.

- Pretrained word embedding

#### Learn Embedding

The relation between word vectors should in row with the semantic relation between words.  

For example, synonym should have similar word vector.  

Generally speaking, geometric distance of any two word vectors should corresponding to the semantic distance between these two words.  

Except for the distance, you may expect the direction have some meaning.  



A good embedding space depends on your task largely.  

The best embedding space of emotion analysis and document classify may different, cause the importance of some semantic relation ranges on tasks.



So, the reasonable way is to learn a new embedding space for every new task.

```
Word Index -> Embedding Layer -> Word Embedding
```

```python
from keras.layers import Embedding

embedding_layer = Embedding(1000, 64)
```

The input to embedding layer should be a 2-dim integer tensor, with the shape of (samples, seq_length).  

Shorter sequence should pad with 0, and longer sequence should be cut.  

The output from embedding layer is a 3-dim float tensor, with the shape of (samples, seq_length, emb_dim)  

```python
from keras.datasets import imdb
from keras.layers import preprocessing

max_features = 10000
max_len = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen = maxlen)
```

```python
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
model.add(Embedding(10000, 8, input_length = maxlen))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy', 
              metrics=['acc'])
model.summary()

history = model.fit(x_train, 
                    y_train, 
                    epochs = 10, 
                    batch_size = 32, 
                    validation_split = 0.2)
```

#### Pretrained Embedding

word2vec - Google - Tomas Mikolov - 2013

GloVe - Global vectors for word representation - Stanford - 2014

## Recurrent Neural Network

#### Simple RNN

```
output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
```

```python
from keras.models import Sequential
from keras.layer import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(10000, 32))  # (None, None, 32)
model.add(SimpleRNN(32))         # (None, 32)

or

model.add(SimpleRNN(32, return_sequences=True)) # (None, None, 32)
```

The question in SimpleRNN lies in : theoretically, on timing t, it should remember the information many timesteps before. However, due to the problem of vanishing gradient, it can not learn such a long-term.  

LSTM and GRU are designed to solve such a vanishing gradient problem.

### Long Short-term Memory

LSTM added a method to carry information through many timesteps.  

It save information for further use, get rid of early signal gradually disappear during processing.  

```
output_t = activation(dot(Wo, input_t) + dot(Uo, state_t) + dot(Vo, c_t) + bo)

i_t = activation(dot(Wi, input_t) + dot(Ui, state_t) + bi)
j_t = activation(dot(Wj, input_t) + dot(Uj, state_t) + bj)
k_t = activation(dot(Wk, input_t) + dot(Uk, state_t) + bk)

c_t+1 = i_t + k_t + j_t * c_t
```

```
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation = 'sigmoid'))
```

## Advanced Usage of RNN

It is useful to design a baseline first, usually based on common sense.  

However, it is not easy to beat this baseline, it shows that our common sense comprise of many valuable information, and it is not easy for a model to learn this.  

If you search a solution in a complicated hypothesis space, it is very likely that you can not learn a simple and good performance solution, though it belongs to your hypothesis space technically.  

In a simple way, if learning algorithm is not hard encoded to find a specific simple model, sometimes it can not find a set of parameters to solve a easy problem easily.  

```python
from keras.models import Sequential
from keras import layers

model = Sequential()
model.add(layer.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layer.Dense(1))
```

#### Recurrent Dropout

Use same dropout mask on every timestep, rather than randomly change it with timesteps.  

Network can propagate its loss through time accurately with the same dropout mask.

```python
model.add(layer.GRU(32, 
                    dropout = 0.2, 
                    recurrent_dropout = 0.2, 
                    input_shape = (None, float_data.shape[-1])))
```

#### Recurrent Layer Stacking

Stack recurrent layer in Keras, every hidden layer should return a complete output sequence ( a 3D-tensor), rather than only return last timestep's output, it can be done with ```return_sequence = True ```

```python
model.add(layer.GRU(32, 
                    dropout = 0.1, 
                    recurrent_dropout = 0.5, 
                    return_sequence = True, 
                    input_shape = (None, float_data.shape[-1])))
model.add(layer.GRU(64, 
                    activation = 'relu', 
                    dropout = 0.1, 
                    recurrent_dropout = 0.5))
model.add(layer.Dense(1))
```

#### Bi-directional RNN

In the task of emotion predict, reverse RNN's performance almost the same with RNN. This phenomenon verifies a hypothesis: Although words' sequence order is essential for understanding language, it doesn't matter which order to use.  

Representation learned by reverse RNN differs from representation learned by RNN matters.  

In ML, if a representation of data different than other but useful, it is worth using.  

The greater the difference between this representation and other representation, the better its value.  

It provides a new point of view to see the data, seize something ignored by other method. Thus, it can provide the performance of the model. This is the intuition of ensemble learning.

```python
model.add(layer.Bidirectional(layer.LSTM(32)))
```

## Use CNN deal with sequence

CNN is efficient when dealing with sequence data. Time can be seen as a 1-dim, like height and width in 2-dim image.  

For some seq-related task, 1-dim CNN performs as good as RNN with a much smaller computational cost.  

Useful task:

- Audio Generation
- Machine Translation
- Text Classification
- Time Sequence Prediction

### 1-dim Convolution

1-dim convolution can recognize local pattern in the sequence.  

A pattern learned in one location of the sentence can be recognized in other location later, this make 1-dim convolution has the property of time-translation invariant.  

A character-level 1-dim CNN should be able to learn word formation.  

### 1-dim Pooling

Extract a 1-dim sequence in the input, and output its maximum or average value.

It is used to reduce the sequence length, in other words, sub-sampling.

### Realize a 1-dim CNN

1-dim CNN can use larger convolution kernel.  

```python
from keras.models import Sequential
from keras import layers

max_features = 10000
max_len = 500

model = Sequential()
model.add(layers.EMbedding(max_features, 128, input_length = max_len))
model.add(layers.Conv1D(32, 7, activation = 'relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation = 'relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
```

### CNN + RNN

1-dim CNN insensitive to timestep sequence, which different from RNN.  

To combine the characteristic of CNN's high speed and RNN's sequence sensitivity, use CNN before RNN as a preprocessing. For those extra long sequence, CNN transform them as shorter high-level feature sequence, then send it to RNN for input.  

```python
model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))
```









