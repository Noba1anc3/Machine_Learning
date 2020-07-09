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