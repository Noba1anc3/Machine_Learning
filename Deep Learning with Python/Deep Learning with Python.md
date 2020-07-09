# Deep Learning

## Machine Learning and Deep Learning

### Machine Learning

Machine Learning's technical definition: Utilize the direction of feedback signal to find the representation of input data in the hypothesis space specified before.

### AI Winter

- 1960s : Symbolism AI

- 1980s : Expert System

## Before DL : Machine Learning

### Probabilistic Modeling

Logistic Regression is a classification algorithm

### Kernel Method

Kernel Trick: to find a good decision hyperplane in the new representation space, we do not need to calculate coordinate in the new space directly, the only thing need to do is to calculate the distance between point pairs.  

To utilize kernel function, it can be done easily.  

Kernel function usually selected by human, not learned from data. For SVM, only hyperplane is learned.

### Random Forest

Since Kaggle go online in 2010, random forest is the best choice for competition. After 2014, it is replaced by gradient boost machine. 

### Difference between DL and ML

Deep learning make things easier, it make feature engineering autonomous completely.  

Top-5 Accuracy : Given an image, if top 5 label predicted comprise the groundtruth, it is a right prediction.  

We cannot duplicate shallow method to fulfill deep method, cause the first layer in 3-layer optimized model is not the first layer in 1-layer or 2-layer optimized model.  

The revolutionary change in deep learning lies in : model can learn all the representation layer in the same time, rather than alternate. (It is called Greedy Learning)  

Deep Learning's two basic characteristic : 

- More and more complex representation are formed in a progressive, layer by layer manner
- Learning hidden progressive layers together

## Why DL Why Now

### Algorithm

- Better activation function
- Better weight-initialization scheme
- Better optimization scheme
- Batch Norm, Residual Connect, Deep Dividable Convolution

# Mathematic Bases

## Neural Network

### Compile

- Loss Function
- Optimizer
- Metric

```python
network.compile(
    loss='categorical_crossentropy', 
    optimizer='rmsprop', 
    metrics=['accuracy'])
```

### Data Preparation

```python
from keras.utils import to_categorical

train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
train_labels = to_categorical(train_labels)
```

### Train

```python
network.fit(train_iamges, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
```

## Data Representation

## Tensor

Tensor is the extension of matrix

- Scalar - 0D Tensor
- Vector - 1D Tensor
- Matrix - 2D Tensor

- ...

### Attribute of Tensor

- ndim
  - number of axis in the tensor
- shape
  - dimension size of the tensor along each axis
- dtype

## Tensor Calculation

### Broadcast

When two tensor add up, smaller tensor will be broadcast to match the size of bigger tensor.  

Steps:

- Add axis to smaller tensor, make its ndim equals to bigger tensor
- duplicate smaller tensor along new axis, make its shape equals to bigger tensor

# Neural Network

## Loss Function

- 2-class classify : binary cross_entropy
- multi-class classify : categorical cross_entropy
- regression : mean squared error
- seq learning : CTC (connectionist temporal classification)

## Keras

- Theano - University of Montreal
- TensorFlow - Google
- CNTK - Microsoft

### Define a Model

- Sequential Class
- Functional API

### Activation Function

Without ReLU-like activation function, there are only two linear calculation in dense layer: dot product and addition. ``` output = dot(W, input) + b ```

Thus, dense layer can only learn linear transformation of input data. The hypothesis space is a possible set of linear transformation. It is a relatively limited space, which cannot utilize the advantage of multi representation layer. Though many layers in the network, they are all linear calculation, more layers would not result in extended hypothesis space.  

To get more vast hypothesis space, fully utilize the advantage of multi layer representation, you need to add non-linear to the function.

### Train your model

```python
history = model.fit(x_train, 
                    y_train, 
                    epochs = 20, 
                    batch_size = 512, 
                    validation_data = (x_val, y_val)
```

```model.fit()``` return a History object, it has a dictionary member ```history```, contains all data during training  

```python
history_dict = history.history
history_dict.keys()
dict_keys(['val_acc', 'acc', 'val_loss', 'loss'])
```

### Draw Training Loss and Validation Loss

```python
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### Summary

#### 2-class Classify

- A stacking of dense layer with ReLU activate can solve many problems
- For 2-class classify problem, the last layer should have only 1 unit, use a dense layer with sigmoid activation
- For 2-class classify problem, you should use ```binary_crossentropy``` as loss function

#### N-class Classify

- For n-class classify, the last layer should be a dense layer of size N
- The last layer should use softmax activation function
- Its loss function should be ```categorical_crossentropy```
- Solve the label of multi class classify
  - Use one-hot encoding : ```categorical_crossentropy```
  - Use integer encoding : ```sparse_categorical_crossentropy```

- Avoid using too small hidden layer, in case of information bottleneck

 #### Regression

- You should not use anything calculated from testing data, even data normalization, you should use the mean and std of training data

- The partition approach of validating data may lead to large variance on validation score 

- To smooth your curve :

  ```python
  def smooth_curve(points, factor = 0.9):
  	smoothed_points = []
  	for point in points:
  		if smoothed_points:
  			previous = smoothed_points[-1]
  			smoothed_points.append(previous * factor + point * (1 - factor))
          else:
          	smoothed_points.append(point)
      return smoothed_points
  ```

- Loss function in regression should be ```mse```

- Evaluate metric also different with classify task. Obviously, accuracy do not suit for regression. The common metric used in regression task is ```MAE``` (Mean Average Error)

- If features of input data in different range, you should scale every feature separately

- If there are little usable data, K-fold validation helps evaluate your model efficiently

- If there are little training data, you'd better use a little network with fewer hidden layer, in case of serious over-fitting

# Machine Learning Basis

Self-supervised Learning: Supervised learning without label

## Evaluation

### Training, Validating Testing Set

Information Leak

- Many configuration need to configure (hyper-parameter)

- This configure process is conducted by the feedback of performance on validating set.

- If configuration is set based on performance on validating set, it will over-fitting on validating set.

- Even you didn't train your model directly on validating set.

- Every time you tune hyper-parameter based on validating set, some info about validating data leaks.

  

Hold-out Validation

```python
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

model = get_model()
model.train(np.concatenate([training_data, validation_data]))
test_score = model.evaluate(test_data)
```



Iterated K-fold validation with shuffling

- Many times of K-fold validation
- Shuffle data before K-split



### Notes on data

Data Representativeness

- Shuffle

The Arrow of Time

- If you want to predict future based on past, you should not shuffle data before split
- It will result in temporal leak, your model will be trained on future data
- Every data in validation set need later than every data in training set

Redundancy in data

- Make sure there is no intersection between training and validating set



## Data Pre-processing

- Vectorization
- Normalization
  - Small value : Range in [0, 1]
  - Homogeneous : All features should range in the same period
  - It's unsafe to send big value or heterogeneous data to nn, result in large gradient update

- Missing Value
  - Usually, it is safe to set 0 for missing value, as long as it is not a meaningful value
  - NN can learn from data that 0 is a missing value, and ignore it

## Feature Engineering

- Describe in a more easy way
- Need more understanding of the question
- It is important before deep learning, cause classical shallow algorithm can not provide big enough hypothesis space to learn useful representation. The way data present to the algorithm is important for solving the question
- Good feature can help you solve the question with less resources and more graceful
- Good feature can help you solve the question with less data

## Under Fitting & Over Fitting

The radical problem in ML is the opposition between optimization and generalization.

### More training data

### Reduce the size of Neural Network (Learnable Parameter)

Model with more parameters have larger memorization capacity.  

It can learn a diction-like mapping between sample and target easily, without any generalization.  

Usually, DL model is good at fitting training data. But the real challenge lies in generalization not fitting.



On the other hand, if the memorization capacity is limited, it is not easy to learn this mapping.



To find a suitable model size, start with a smaller model, then add layers gradually, until little inference to validation performance.



Smaller model's over-fitting timing comes later than original model, and its performance dropping speed is also slower.

Bigger model's over-fitting timing comes only after 1 epoch, and its performance drops drastically, training loss drop to zero in a fast speed.



### Weight Regularization

Occam's Razor : If there are two explanation to one thing, the most possible one is the simplest one.



Simple means less entropy of parameter's distribution in the model.

The distribution of weights should be regular.

The way to realize it is to add cost related to bigger weight.



L2 Regularization also names weight decay.



The penalty term only add when training, lead to bigger training loss than testing loss.



### Dropout

Hinton : "I went to the bank to handle some business, and I saw the counter employee also change, so I asked one person why, he answered he doesn't know, but they often change their position. I guess, if bank employees want to cheat bank successfully, they need to cooperate each other. It imply me that, delete some neuron randomly during training every sample can deter them from their conspiracy, thus reduce over-fitting." 



## General Process in ML

### Define your problem and gather data

### Choose a metric to measure your success

The metric of success can lead you choose loss function

- Balanced Classification : Accuracy & ROC AUC
- Unbalanced Classification : Precision & Recall
- Sort & Multi-label classification : mAP

### Determine validation scheme

- Hold-out Validation
- K-fold Cross Validation
- Iterated K-fold Validation

### Prepare data

- Format data to tensor
- Scale to lower range
- Homogeneous your data
- Feature Engineering

### Develop a model better than baseline

The target in this phase is to get statistical power : It can beat dumb baseline

Hypothesis:

- Output can be predicted based on input
- Enough info in data to learn the mapping from input to output

To construct the model, there more configures:

- Activation function in the last layer
- Loss Function
- Optimizer

Something about loss function:  

- It not feasible to optimize the metric of success, that is, it is hard to change the metric to loss function

- Loss function need to be calculate on small batch of data (in ideal situation : one sample)

- Loss function need to be differentiable, unless back-propagation can not be used

### Enlarge the model, develop a over-fitting model

The ideal model stands on the boundary line of under fitting and over fitting, of deficient capacity and sufficient capacity. To find this line, you need to cross it.  

To train a over-fitting model:

- Add more layers
- Bigger every layer
- Train more epochs

### Regularization and adjust hyper-parameter

- Dropout
- Different architecture : more or less layers
- L1 / L2 Regularization
- Different Hyper-parameter
- Feature Engineering
  - Add new feature
  - Delete useless feature

Once a satisfying model is developed. You can train a final model by using training and validating data, and evaluate its performance on the testing data.

# Deep Learning in Computer Vision

## Convolutional Neural Network

```
conv -> maxpooling -> ... -> flatten -> dense
```

### Convolution

The radical difference between dense connection and convolution lies in dense layer learn on overall mode, and convolution layer learn on partial mode.

Two properties of CNN:

- Pattern learned by CNN is translation invariant （平移不变性）
  - Patterns can be anywhere in the image
  - Visual Sense is also translation invariant
- CNN can learn spatial hierarchies of patterns
  - CNN can learn more complicated and more abstract visual concept efficiently
  - Visual Sense also has spatial hierarchies

Every dimension on depth axis of processed image is a feature / filter.

Two key params of Convolution:

- Kernel Size: usually 3x3 or 5x5
- Filter Num ( Output Depth )

Padding:

- Add rows and columns on each edge of the input image, so that every pixel can be the center of the conv window.
- Valid Padding : without padding (default)
- Same Padding : output size equals to input size

Max-Pooling:

- Usually with kernel size 2x2 and stride 2
- Reasons:
  - Reduce num of element need to be processed (time complexity)
  - Result in bigger window of continuous convolution layer (ratio of window size and origin input gets bigger) -> spatial hierarchies

## Train a CNN

Big data is a relative conception, compared with the size and depth of the network you want to train.

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    traget_size=(150,150), 
                                                    batch_size=20, 
                                                    class_mode='binary')
```

Python Generator

- Object like iterator
- Use 'for' or 'in' ... together
- 'yield'
- break manually

```python
history = model.fit_generator(train_generator, 
                              steps_per_epoch=100, 
                              epochs = 30, 
                              validation_data = validation_generator, 
                              validation_steps = 50)
```

Do not augment validation data.

## Pretrained CNN

### Feature Extraction

Feature Extraction extract feature from new sample by the presentation learned by previous neural network.  
Then, feature sent to a new classifier, and train from scratch.

Usually there are two parts in the CNN for image classification: convolutional base & dense connected classifier.  
Usually only the convolutional base is re-used.

The generalization of the representation extracted by a conv-layer based on its depth in the model.  
Layers near the bottom -> partial, general (edge, color, texture)  
Layers near the top -> whole, abstract (cat's ear or dog's eye)  

If new data ranges a lot from previous data, you'd better only use layers close to the bottom for feature extraction not whole conv-base.

#### VGG16

- VGG16

  - ( conv -> conv -> pool ) * 2 -> ( conv -> conv -> conv -> pool) * 3
  - Conv-Head : 14714688 params 
  - froze conv-head when training pretrained model based on VGG-16's conv-head + self-defined dense_connected_classifier, cause randomly initialized dense layer.
    '''
    conv_base.trainable = False
    '''

- VGG16 in Keras

  ```python
  from keras.applications import VGG16
  conv_base = VGG16(weights='imagenet', include_top=False)
  ```

- Architecture of VGG16

  ```python
  conv_base.summary()
  ```

### Finetune the model

Steps:

- Add self-defined network on top of trained base network
- Freeze base network
- Train added network
- Unfreeze some layer in base network (usually near top in image classify task)
- Train these unfrozen layers and added network simultaneously

Reasons:

- Layers more close to bottom extract more general and reusable features
- Layers more close to top extract more specified features
- It is more useful to finetune these specified features, cause they need to change their usage on your own dataset
- Less reward on finetune layers  bottom
- More trainable params, more risk of over-fitting, there are 15 millions of params in the conv-base, it is not easy to train on your little dataset

## Visualize a CNN

- Visualize bottleneck output
- Visualize filter
- Visualize thermodynamic diagram of class-activation

### Visualize bottleneck output

```python
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(imputs = model.input, outputs = layer_outputs)
activations = activation_model.predict(img_tensor)
```

- First layer is a collection of various edge detectors
- With deeper and deeper, activation gets more and more abstract, and difficult to understand intuitively
- With layer more deep, less visual content and more class information
- The sparsity of activation grows with layer gets deeper

Deep NN can be used as information distillation pipeline. Input origin data, transform many times, filter irrelavant and magnify useful information.

### Visualize filter

```python
from kears.application import VGG16
from keras import backend as K
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1)
    
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x

def generate_pattern(layer_name, filter_index):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradient(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])
    loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

    input_img_data = np.random.random((1, 150, 150, 3) * 20 + 128.)

    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    
    return deprocess_image(img)

plt.show(generate_pattern('block3_conv1', 0))
```

### Visualize thermodynamic diagram of class-activation

Method : Grad-CAM : visual explanations from deep networks via gradient-based localization

Given an image, for a feature-map output by one conv-layer, it calculate a weighted mean of class-channel gradient in every channel.

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

# Advanced Deep Learning Practices

## Sequential -> Keras Functional API

### Drawback of Sequential

In Sequential, there is only 1 input and 1 output, all layers are linearly stacked.  

- Some task need multimodal input

- Some task need predict multi attribute of the input
- Many latest nn architecture demand nonlinear topological structure, as a DAG  (Inception, ResNet)

### Keras Functional API

Use layer as a function, receive and return a tensor.  

```python
from keras.models import Sequential, Model
from keras import layers, Input

# Sequential
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation = 'relu', input_shape = (64,)))
seq_model.add(layers.Dense(32, activation = 'relu'))
seq_model.add(layers.Dense(10, activation = 'softmax'))

# Functional API
input_tensor = Input(shape = (64,))
x = layers.Dense(32, activation = 'relu')(input_tensor)
x = layers.Dense(32, activation = 'relu')(x)
output_tensor = layers.Dense(10, activation = 'softmax')(x)

# Model Class transform the i-o tensor to a model
model = Model(input_tensor, output_tensor)
```

### Multi Input Model

```python
from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape = (None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate(
    [encoded_text, encoded_question],axis=-1)

answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
```

```python
import numpy as np

num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size,
                         size=(num_samples, max_length))

question = np.random.randint(1, question_vocabulary_size,
                             size=(num_samples, max_length))

answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
answers = keras.utils.to_categorical(answers, answer_vocabulary_size)

# Use a list of input
model.fit([text, question], answers, epochs=10, batch_size=128)

# Use a dict of input
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)
```

### Multi Output Model

```python
from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)

x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, 
                                 activation='softmax', 
                                 name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])
```

We need specify different loss function for different head.  

For example, age prediction is a scalar regression task. Gender prediction is a 2-class classification task. They need different training process. However, gradient descent demand minimum of a scalar, so as to train the model, we need to mix these losses to one scalar. The simplest way is to add them up.  

```python
model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])

# Equivalent to the usage above
model.compile(optimizer='rmsprop',
              loss={'age': 'mse',
                    'income': 'categorical_crossentropy',
                    'gender': 'binary_crossentropy'})
```

Attention : serious imbalance of loss contribution will lead to the direction of optimization point to the task with biggest loss. To solve this, we need to allocate different quota to different loss.

```python
model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])

model.compile(optimizer='rmsprop',
             loss={'age': 'mse',
                   'income': 'categorical_crossentropy',
                   'gender': 'binary_crossentropy'},
             loss_weights={'age': 0.25,
                           'income': 1.,
                           'gender': 10.})
```

```python
model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)

model.fit(posts, {'age': age_targets,
                  'income': income_targets,
                  'gender': gender_targets},
          epochs=10, batch_size=64)
```

### Directed Acyclic Graph

#### Inception

Inception is a CNN based architecture, it is a stacking of modules, these modules look like a tiny independent network, divided into multi branches.  

The basic inception have 3-4 branches, first 1 x 1 conv, then 3 x 3 conv, finally add all features together.  

This kind of design help network to learn spatial and channel-wise feature separately.  

##### 1 x 1 Convolution ( Pointwise Convolution )

- Mix different channel in the input tensor up
- Would not mix up trans-space information (cause it look into one pixel once)
- Contribute to separating channel feature learning and spatial feature learning

```python
from keras import layers

branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)

branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)

branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)

branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
```

Keras has a built-in implementation of InceptionV3

```
keras.application.inception_v3.InceptionV3

```

#### Xception

Xception = Extreme Inception  

Xception push learn channel feature and spatial feature separately to its extreme. Replace Inception module with deep separable convolution, the spatial feature and channel feature are completely separated.      

Xception's parameters' num roughly equals to Inception V3. Because of its more efficient use of parameters, it have better performance on ImageNet and other large scale datasets.

#### Residual Connection

Residual connection solved two common issue in large scale deep learning model

-  Gradient Vanishing
-  Representation Bottleneck

The output of former layer add to later layer's activation. If they are of different shape, it use a linear transform to change former output's shape equal to the target shape.

- Identity Residual Connection

  ```python
  from keras import layers
  x = ...
  y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
  y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
  y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
  y = layers.add([y, x])
  
  ```

- Linear Residual Connection

  ```python
  from keras import layers
  x = ...
  y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
  y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
  y = layers.MaxPooling2D(2, strides=2)(y)
  # Use 1 x 1 conv, subsampling x to the shape of y
  residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
  y = layers.add([y, residual])
  
  ```

##### Representation Bottleneck

In Sequential model, every continuous representation layer is constructed on the former layer, that is, it can only get the information activated in the former layer. If one layer is too small, the model is limited to how much information can be held in this layer.  

Any loss of information is permanent, residual connection can resend earlier information to later layer.  

##### Gradient Vanishing

Backpropagation propagates loss in the output to the layer in the bottom. If the feedback signal need to go through many layers, it may be very weak, even lost, result to untrainable network.  

Both deep network and recurrent network for sequence processing have this problem.   

We already know how LSTM solve this problem: LSTM introduce a carry track, which propagate information in the track parallel to the main track.  

Residual connection's principle similar to this, and more simple: It introduce a linear track to carry information, parallel with the direction of main stacking layers, which helps propagate gradient through deep layers.  

### Share Weight

We can reuse a layer instance in functional API. If you call a layer instance twice, rather than instantiation a new layer every time, you can reuse the same weight. Thus, construct a model with shared branch.  

For example, to evaluate semantic similarity of two sentence, the model have two inputs.  

It has no sense to learn two separate model to process two input sentence, we only use one LSTM layer to handle the two sentence. We call it Siamese LSTM, or shared LSTM model.

```python
from keras import layers
from keras import Input
from keras.models import Model

lstm = layers.LSTM(32)

left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)

model = Model([left_input, right_input], predictions)
model.fit([left_data, right_data], targets)

```

### Model As Layer

Call a model on an input tensor, it will return a output tensor.  

Call an instance, whether layer or model instance, it will reuse the representation the instance has learned.  

For example: two camera for depth perception

```python
from keras import layers
from keras import applications
from keras import Input

xception_base = applications.Xception(weights=None, include_top=False)

left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))

left_features = xception_base(left_input)
right_input = xception_base(right_input)

merged_features = layers.concatenate([left_features, right_input], axis=-1)

```

## Callback Function & TensorBoard

### Callback Function

Callback is an object sent to model when calling fit function, it will be called by the model at different time in the training process. It can access all available data about the state and performance of the model, and do something like: interrupt training, save model, load different weights or change the state of the model.  

Some Usage :

- Model checkpoint : Save current weights on different timing
- Early stopping 
- Tuning some parameter during training
- Record training and validating result, or visualize the representation in real time

```python
keras.callbacks.ModelCheckpoint
keras.callbacks.EarlyStopping
keras.callbacks.LearningRateScheduler
keras.callbacks.ReduceLROnPlateau
keras.callbacks.CSVLogger

```

#### EarlyStopping & ModelCheckpoint

```python
import keras

callbacks_list = [
	keras.callbacks.EarlyStopping(
		monitor='acc',
		patience=1,
    ),
	keras.callbacks.ModelCheckpoint(
		filepath='my_model.h5',
		monitor='val_loss',
		save_best_only=True,
	)
]

model.compile(
    optimizer='rmsprop',
	loss='binary_crossentropy',
	metrics=['acc'])

model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))

```

#### ReduceLROnPlateau

If val loss do not reduce any more, you can use this callback function to reduce learning rate.  

If loss plateau appears, magnify or reduce lr is an efficient strategy of jumping out local minimum.  

```python
callbacks_list = [
	keras.callbacks.ReduceLROnPlateau(
		monitor='val_loss'
		factor=0.1,
		patience=10,
	)
]

model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))

```

#### Write Customized Callback Function

Setup a subclass of class ```keras.callbacks.Callback```, then you can implement such methods :

- on_epoch_begin
- on_epoch_end
- on_batch_begin
- on_batch_end
- on_train_begin
- on_train_end

Callback Function and access such attributes :

- self.model
- self.validation_data

Below is an example of customized callback function, it save every layer's activation to hard disk after every epoch, the activation is calculated on the first sample of validation data.

```python
import keras
import numpy as np

class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input,
                                                    layer_outputs)
        
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
        	raise RuntimeError('Requires validation_data.')
            
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()

```

### TensorBoard

Loop of Improvement : ``` Idea -> Experiment -> Result -> Idea```  

TensorBoard is a visualization tool based on browser built in TensorFlow.  

Here are some function of TensorBoard :

- Monitoring on indicators with visualization during training
- Visualize architecture of model
- Visualize histogram of activation and gradient
- Research embedding with 3-dim visualization

```python
callbacks = [
	keras.callbacks.TensorBoard(
		log_dir='my_log_dir',
		histogram_freq=1,
		embeddings_freq=1,
	)
]

history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)

```

``` $ tensorboard --logdir=my_log_dir```  

``` http://localhost:6006```  

- SCALARS
  - acc
  - loss

- HISTOGRAM

- EMBEDDING
  - PCA
  - t-SNE

- GRAPHS

  - Visualization of Bottom TensorFlow Operation Graph

  - A more concise way  ```keras.utils.plot_model``` 

    - Layer Graph rather than Operation Graph

    - ``` pip install pydot, pydot-ng, graphviz```
    - ```plot_model(model, show_shapes = True, to_file = 'model.png')```

## To Model's Extreme

### Advanced Architecture

#### Batch Normalization

Normalization makes different sample similar to each other, it helps better learning and generalization.  

``` normalized_data = (data - np.mean(data, axis = ...)) / np.std(data, axis = ...)```  

We need consider normalization after every transformation in the network. Though data sent to dense or conv2d with mean 0 and std 1, it is not sure the output of network also conform to this.  

Principle of Batch Normalization : It save the exponential shift average value of mean and variance of every batch of data during training inside the network.  

Effect of Batch Normalization : It helps gradient propagation, avoid of gradient vanishing, allow for deeper network.  

Batch Normalization often use after conv layer or dense layer.  

BN receive a axis parameter, specify an axis to normalize, default is -1, means the last axis of input tensor.  

##### Batch Renormalization

Compared with BN, it has obvious advantage with cost increased slightly.

##### Self-normalizing Neural Network

It use special activation function (selu) and special initializer (lecun_normal), with the ability to keep data normalized after passing through any dense layer.

#### Depthwise Separable Convolution

- Lighter - Less trainable weights
- Faster - Less float calculation
- Better - More performance

Execute spatial convolution on every channel, and mix them up by 1 x 1 convolution.  

If your data's spatial location is highly correlated, and channels are relatively independent, it is better to do separable convolution.  

``` layers.SeparableConv2D()```  

For larger model, depthwise separable convolution is the basis of Xception.

### Hyper-parameter Optimization

Steps :

- Select a set of hyper-parameters automatically
- Construct corresponding model
- Fit model on training data, measure its final performance on validating data
- Select next group of hyper-parameters automatically
- Repeat the process above
- Finally, measure the performance on testing data

Methods :

- Bayesian Optimization
- Genetic Algorithm
- Simple Random Search

We cannot do gradient descent in hyper-parameter space.  

Usually random search is a good solution. There is a tool actually better than random search, it is a python lib named Hyperopt, use a tree of Parzen estimator to predict which set of hyper-parameter will get good result.  

Hyperas : Hyperopt + Keras  

Attention : You are tuning your hyper-parameter based on validation data. So your model will over-fitting on validation set at some point.  

### Ensemble Learning

Ensembling based on such a hypothesis : For different good model trained independently, their good performance may from different reason, every model make decision from slightly different perspective, get a part of 'truth', not all the truth. Gather their decision together, you can get a more precise description.  

Ensemble Learning do weighted average on different model. The weight is calculated based on the performance on validating data. Usually, a more accurate model gets a bigger weight.  

To guarantee a good ensembled model, the key is the diversity of a set of models. If all the models' bias on the same direction, ensemble learning also has this bias. However, if all the models' bias on different direction, the ensembled model will have more precise and stable performance.  

All the models should both good and different. Usually, this means use different architecture, even different machine learning methods.  

A effective way : tree-based method (random forest, gbdt ... ) + deep learning.

Author's empirical experience : during his Kaggle competition, the model with lowest score derived from a method which different from all other models in ensemble learning (regularized greedy forest) was given a very small weight. However, the result was unexpected, because of its difference with other models, it provided information that other models can not get, improved the result tremendously.  

Recently, a successful way to ensemble is "wide and deep", which means a collection of deep learning and shallow learning.  

# Generative Deep Learning

- Music Writing
- Dialogue 
- Image Generation
- Speech Synthesis
- Molecular Design
- Script Design
- ...

## Text Generating from LSTM

- Text
- Music
- Drawing

### A Brief History of Generative Recurrent Network

- Until end of 2014, almost nobody know LSTM
- Get into main area since 2016
- Earliest - 1997
- First apply in music generation, and get a satisfying  - 2002
- Apply recurrent mix density network on generating human handwriting - 2013

### How to generate sequence data

General Method : Use former tokens input, train a RNN / CNN to predict next one (or more) token(s).  

#### Language Model

Given former tokens, any network able to modeling the probability of next token.  

Language model can capture the latent space of a language, aka, the statistic structure of a language.  

#### Sample

Given a language model, generate a new sequence.

### The Importance of Sample Strategy

#### Greedy Sampling

- Always select the character with max probability

- Generate a repetitive, predictable string
- Do not look like coherent language

#### Stochastic Sampling

- Introduce randomness during sampling
- If the next character is 'e''s probability is 0.3, then model will choose it with a probability of 30%

Whether minimum-entropy selection (greedy sampling) or maximum-entropy selection (completely random sampling) would not generate anything interesting.  

Sampling in the 'real' probability distribution (softmax) is a compromise.  

- Bigger Entropy : Creativity
- Smaller Entropy : Authentic

To control randomness during sampling, we introduce a parameter called 'softmax temperature' to indicate the entropy of sample probability distribution, aka, indicates how unexpected or predictable the next character you select.  

```python
import numpy as np

def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)
```

### Character-Level Text Generation by LSTM

#### Seq2Vector

```python
maxlen = 60
step = 3

sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
	sentences.append(text[i: i + maxlen])
	next_chars.append(text[i + maxlen])
	print('Number of sequences:', len(sentences))
    
	chars = sorted(list(set(text)))
	print('Unique characters:', len(chars))
	char_indices = dict((char, chars.index(char)) for char in chars)

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		x[i, t, char_indices[char]] = 1
		y[i, char_indices[next_chars[i]]] = 1
```

#### Construct LSTM

```python
from keras import layers

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

#### Train a language model and sample from it

Given a trained model and a seed text fragment, replicate following instructions to generation sequence :

- Given generated text, get the probability distribution of next character from the model
- Re-weight distribution by a temperature
- Randomly sample next character according to the distribution re-weighted
- Add new character to the end of the sequence

``` python
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = reweight_distribution(preds, temperature)
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)
```

```python
import random
import sys

for epoch in range(1, 60):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')
    
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)
        
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
            	sampled[0, t, char_indices[char]] = 1.
            
            preds = model.predict(sampled, verbose=0)[0]
            
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            
            generated_text += next_char
            generated_text = generated_text[1:]
            
            sys.stdout.write(next_char)
```

## DeepDream

DeepDream is an artistic technology for image modification. It is published in 2015 by Google.  

It runs a CNN reversely, do gradient ascend to the input, so as to maximum the activation of a filter.  

- Maximum the activation of many layers - Mix up the visualization of all features

- Start from a existed image - The result can seize the visual pattern already exists in the image
- Process on different scale (called octave) - Improve the quality of visualization

### Keras Implementation

Load pretrained Inception V3 model

```python
from keras.applications import inception_v3
from keras import backend as K

# Ban all the operation related to training
K.set_learning_phase(0)

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
```

Set DeepDream Configuration

```python
layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
}
```

We need to cal the loss, aka, the value need to be maximum during gradient ascend.  

To maximum the activation, we cal a weighted sum of L2-norm of a set of layers, and maximize it.  

The layer we choose matters, lower layers generate geometric pattern, and higher layers generate something which can be recognized as some class of ImageNet.  

Define the loss to be maximized

```python
layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)

for layer_name in layer_contributions:
	coeff = layer_contributions[layer_name]
	activation = layer_dict[layer_name].output
    
	scaling = K.prod(K.cast(K.shape(activation), 'float32'))
	loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling
```

Gradient Ascend

```python
dream = model.input

# calculate the gradient of loss to dream image
grads = K.gradients(loss, dream)[0]
# normalization
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
    	loss_value, grad_values = eval_loss_and_grads(x)
    	if max_loss is not None and loss_value > max_loss:
    		break
    	print('...Loss value at', i, ':', loss_value)
    	x += step * grad_values
    return x

```

Finally, it comes to DeepDream.  

First, we define a list, containing the scale of the image to be process. Every continuous scale is the 1.4 times of the previous one.  

For every scale, from the smallest to the biggest, we need do gradient ascend to maximum the loss. After every gradient ascend, we scale up 40% of the image.  

To avoid losing many details of the image, we inject the loss of origin image to the zoomed in image.

```python
import numpy as np

step = 0.01        # learning rate of gradient ascend
num_octave = 3     # num of scale
octave_scale = 1.4 # zoom in ratio
iterations = 20    # how many times we do gradient ascend on every scale

# we need to interrupt the process of gradient ascend when loss > 10
# to avoid generating an ugly image
max_loss = 10.

base_image_path = '...'

img = preprocess_image(base_image_path)

original_shape = img.shape[1:3]
successive_shapes = [original_shape]

for i in range(1, num_octave):
	shape = tuple([int(dim / (octave_scale ** i))
		for dim in original_shape])
	successive_shapes.append(shape)

# reverse the shape list to the ascending order
successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    
    img += lost_detail
    
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')

save_img(img, fname='final_dream.png')

```

Auxiliary methods

```python
import scipy
from keras.preprocessing import image

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
    	x = x.reshape((3, x.shape[2], x.shape[3]))
    	x = x.transpose((1, 2, 0))
    else:
    	x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

```

## Neural Style Transfer

Style : Texture, color and visual pattern in different spatial scale of the image  

Content : High-level macroscopic structure of the image  

Core of all the deep learning algorithms : Define a loss function to specify the target we want to realize, and minimize it.  

If we can give a definition of style and content in mathematics, we will have a proper loss function :

```
loss = distance(style(reference_image) - style(generated_image)) + 
       distance(content(original_image) - content(generated_image))

```

### Content Loss

The content of the image is abstract and global, it can be catch by the layers close to the top of CNN.  

The good candidate for content loss it the L2-norm of two activations. One activation is the output from the layer close to the top by processing the origin image, another activation is generated by processing the generated image.  

This can ensure generated and origin image looks similar from the layer close to the top of CNN.

### Style Loss

We want to catch the appearance extracted by CNN from style image in all spatial scales. Gatys et, al. used Gram matrix of layer activation, aka, the inner product of a feature map of any layer.

The inner product can be understood as the mapping of the relationship between the features of the layer.  

So, the goal of style loss is to save similar relationship between the style image and the generated image about the activation of different layers.

### Implementation

```python
from keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)

print('Model loaded.')

```

Content Loss

```python
def content_loss(base, combination):
	return K.sum(K.square(combination - base))

```

Style Loss

```python
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    
    channels = 3
    size = img_height * img_width
    
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

```

Total Variation Loss : Impel generated image with spatial continuous, avoid excessive pixelation of result.

```python
def total_variation_loss(x):
    a = K.square(
    	x[:, :img_height - 1, :img_width - 1, :] -
    	x[:, 1:, :img_width - 1, :])
    b = K.square(
    	x[:, :img_height - 1, :img_width - 1, :] -
    	x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

```

Define the ultimate loss

```python
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_image)

```

L-BFGS Algorithm for optimization  

Two restriction of L-BFGS implemented in SciPy

- Loss and gradient as two unique parameter
- Only apply on flatted vector

Set the process of gradient descend

```python
grads = K.gradients(loss, combination_image)[0]

fetch_loss_and_grads = K.function([combination_image], [loss, grads])

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        
        self.loss_value = loss_value
        self.grad_values = grad_values
        
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        
        grad_values = np.copy(self.grad_values)
        
        self.loss_value = None
        self.grad_values = None
        
        return grad_values

evaluator = Evaluator()

```

Style Transfer

```python
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

result_prefix = 'my_result'
iterations = 20

x = preprocess_image(target_image_path)
x = x.flatten()

for i in range(iterations):
	print('Start of iteration', i)
	start_time = time.time()
    
	x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                 	x,
                                 	fprime=evaluator.grads,
                                 	maxfun=20)
	print('Current loss value:', min_val)
    
	img = x.copy().reshape((img_height, img_width, 3))
	img = deprocess_image(img)
    
	fname = result_prefix + '_at_iteration_%d.png' % i
	imsave(fname, img)
	print('Image saved as', fname)
	end_time = time.time()
	print('Iteration %d completed in %ds' % (i, end_time - start_time))

```

## VAE & GAN

### Latent Space

Key point of image generation is to find a low-dim representation latent space. In the latent space, every point can be mapped as a realistic image. The module that can fulfill such a mapping called generator (in GAN), or decoder (in VAE).  

- VAE
  - Able to learn a latent space with good structure
  - A specific direction indicate a meaningful axis of change in data

- GAN
  - Latent space does not have a good structure
  - No enough continuous

### Variational Auto Encoder

VAE fuse the idea of deep learning with Bayesian inference.  

Rather than zip input image into an encoding in the latent space, VAE transform image into the parameter of statistical distribution, aka, mean and variance. Then, VAE randomly sample an element in the distribution use the mean and the variance, and decode the element to the origin input.  

The randomness of this process increased its stability, impel all the points in the latent space corresponding to a meaningful representation.  

Principle of VAE :

- ```input_img``` transform into two parameter ```z_mean``` and ```z_log_variance``` in the latent space
- We assume normal distribution can generate the image, and randomly sample a point in the distribution ```z``` : ```z = z_mean + exp(z_log_variance) * epsilon``` , ```epsilon``` is a very small random number
- A decoder map this point in the latent space to the origin input

 Any two adjacent points in the latent space can be decode as two highly similar images.  

Continuous and low-dimension of latent space impel every direction in the latent space corresponding to a meaningful variation axis. This makes latent space has a good structure.  

VAE Loss Function :

- Reconstruction Loss : make output similar to input
- Regularization Loss : Contribute to learn a latent space with good structure, reduce over-fitting

VAE Encoder

```python
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2

input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3,
                  padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu',strides=(2, 2))(x)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

```

Sample in latent space

```python
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

```

VAE Decoder

```python
decoder_input = layers.Input(K.int_shape(z)[1:])

# up-sampling
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)

# change to feature map before flatten
x = layers.Reshape(shape_before_flattening[1:])(x)

x = layers.Conv2DTranspose(32, 3,
                           padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(1, 3,
                  padding='same', activation='sigmoid')(x)

decoder = Model(decoder_input, x)

z_decoded = decoder(z)

```

VAE Loss

```python
class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
        	1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
    	return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x
    
y = CustomVariationalLayer()([input_img, z_decoded])

```

Train VAE

```python
from keras.datasets import mnist

vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))

vae.fit(x=x_train, y=None,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_data=(x_test, None))

```

Decode a point to image

```python
import matplotlib.pyplot as plt
from scipy.stats import norm

n = 15
digit_size = 28

figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
	for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit
       
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

```

### Generative Adversarial Network

The training process of GAN different from any other methods, its optimal value is not a fixed point.  

The optimization process is not to find a minimum, it is a balance of two power.  

#### Tricks

- Use tanh as the activation of the last layer in Generator
- Sample in latent space by Gaussian distribution
- Randomness can rise robustness
  - Use dropout in Discriminator
  - Add random noise in the label

- Sparse gradient can hinder training (max-pooling and ReLU)
  - Max-pooling to Strided Convolution
  - ReLU to LeakyReLU

- When chessboard artifact appears in generated image, make sure stride can divide kernel size

#### Generator

```python
import keras
from keras import layers
import numpy as np

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# up-sample to 32 x 32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

generator = keras.models.Model(generator_input, x)
generator.summary()

```

#### Discriminator

```python
discriminator_input = layers.Input(shape=(height, width, channels))

x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

# grad clip to restrict the range of gradient
# use decay to stabilize training process
discriminator_optimizer = keras.optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy')

```

#### GAN

```python
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))

gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

```

#### DCGAN

- Select a point in latent space randomly
- Feed it to generator for generating an image
- Blend generated image with real image
- Use blended image and corresponding label train discriminator
- Select a new point in latent space randomly
- Use all true label train gan, it will update generator's weight

```python
import os
from keras.preprocessing import image

(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 6]
x_train = x_train.reshape(
    (x_train.shape[0],) +
    (height, width, channels)).astype('float32') / 255.

iterations = 10000
batch_size = 20
save_dir = 'your_dir'

start = 0
for step in range(iterations):
	random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
	generated_images = generator.predict(random_latent_vectors)

    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)
    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # lie all are true images
    # the update direction of generator is to make discriminator predict generated image
    # as true image
	misleading_targets = np.zeros((batch_size, 1))
    #train generator through gan model (freeze discriminator's weight)
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    
    if step % 100 == 0:
        gan.save_weights('gan.h5')
        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir,
                              'generated_frog' + str(step) + '.png'))
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir,
                              'real_frog' + str(step) + '.png'))

```

# Summary

## How to view deep learning

In deep learning, everything is vector, everything is point in the geometric space.  

Core ideology : meaning comes from pairwise relationship between objects, and there relationship can be represented as distance function.  

Neural network first origin from encoding meaning by a Graph, that is why it is called neural network. We still call it neural network purely for historical reasons, it is a misleading name, the better name is :

- Layered Representations Learning
- Hierarchical Representations Learning
- Deep Differentiable Model
- Chained Geometric Transform

## Potential Space

- Vector - Vector
  - Medical Effect Prediction
  - Behavior Orientation
  - Product Quality Control

- Image - Vector
  - Doctor Assistance
  - Auto Driving
  - Go
  - Dietary Assistant
  - Age Prediction

- Time Sequence - Vector
  - Weather Forecast
  - Brain Computer Interface
  - Behavior Orientation

- Text - Text
  - Intelligent Reply
  - Question Answering
  - Abstract Generation

- Image - Text
  - Image Description

- Text - Image
  - Conditional Image Generation
  - Sign Making

- Image - Image
  - Super Resolution
  - Visual Depth Perception

- Image + Text - Text
  - Image Question Answering

- Video + Text - Text
  - Video Question Answering

## Restriction of Deep Learning

Deep Learning model is just a simple and continuous geometric transform chain, map one vector space to another vector space. The only thing it can do is to map a manifold data X to another manifold data Y, if there is a learnable continuous transformation from X to Y.

## Partial Generalization and Extreme Generalization

Fundamental difference between human and machine learning :  

Human learn from embodied experience, not from explicit training samples.

### Extreme Generalization

We use a complicated abstract model for now, for ourselves, for others, and we can set long-term plan from many possible future imagined by ourselves. We can fuse known conception to describe unexperienced things. This ability for processing imaginary situation extend our mind far behind we can experience, make us able to abstract and reasoning. We call it extreme generalization.

### Partial Generalization

Extreme generalization is in sharp contrast to deep learning's generalization.  

Deep Learning map input to output, if new input not quite the same with network has seen, the mapping become meaningless immediately.

## Deep Learning's Future

- Model close to general computer program
  - Inference and abstract

- Make above point become new learning form
  - Abandon differentiable calculation

- Less need for human engineering on model adjusting

- Better, systematic reuse learned feature and architecture

### Model As Program

Evolve of DL in the future : abandon the model can only process pure pattern recognition, fulfill partial generalization, study the model can abstract and inference, fulfill extreme generalization.  

Imagine a nn, it is reinforced by programing primitives, it can operate these primitives to extend its function, like if branch, while sentence, create variable, use long-term disk memory, sort operation, high-level data structure ...   

In the future, we do not use hard encoded algorithm intelligence, and we do not use learned geometric intelligence anymore. We fuse real algorithm and real geometric modules, former can provide the ability of inference and abstract, latter can provide informal intuition and pattern recognition. The study process of the system only need few people, even no people.

### Beyond Back Propagation and Differentiable Layers

If machine learning more like a program, it is not differentiable anymore. The model use continuous differentiable geometric layer as its subprogram as usual, but the model is not differentiable.  

We need to find a method which can train undifferentiable system efficiently.  

Foreseeable future : overall model is undifferentiable, but some module is differentiable. We use a efficient search process to train the model, and use a more efficient version of bp get the gradient to train the differentiable part.

### Autonomous Machine Learning

In the future, model architecture is learned not designed.  

Use AutoML to adjust hyper parameters.  

Another important auto ml direction is learn model architecture and weight simultaneously. Every time we try a different architecture, we need to train a new model from the beginning, it is inefficient. The real powerful automl should be able to adjust the architecture when adjusting weights by back propagation.

### Lifelong Learning and Reuse of Modularized Program Subroutine

Information in many datasets is not sufficient to train a complicated new model from beginning, it is necessary to use the information from previous dataset.  

Train a model which can complete several almost no relation tasks will increase the performance on every task. For example, we train a machine translation model for English-German and French-Italian, this model perform better on both two task. Another example, unify train a image classification model and a image segmentation model, they share the same convolution base, the trained model perform better on two task.  

Intuitive reason : there are some information gap between seemingly irrelevant tasks. Unified model can get more information about every task compared with only train on specific task.

As model more and more like program, we start to reuse the program subroutine, like reuse the function and class in human coding language.

In the future, meta learning can screen from high level reusable module library, and combine a new program. If the system aware of it developed similar subroutine for several tasks, it can extract a abstract, reusable version of these subroutines, and store it in the lib. This process can fulfill abstract, abstract is the necessary component for extreme generalization. 

### Long-term Vision

- Model more like program, it will far beyond continuous geometric transformation.
- Model will fuse algorithm module and geometric module, former provide reasoning, search and abstract ability, latter provide informal intuition and the ability of pattern recognition.
- By use modularized component stored in the reusable subroutine library, model can grow automatically, without human hard-encoding.
- The lib and the corresponding model growth system can fulfill some form of like-human extreme generalization : given a new task or new situation, system can combine a new efficient model suitable for the task by using little data. This should credit to sufficient primitives, they have good generalization, and also need to credit to much experience learned by similar tasks.
- This kind of eternal learning model growth system can be seen as an artificial general intelligence.