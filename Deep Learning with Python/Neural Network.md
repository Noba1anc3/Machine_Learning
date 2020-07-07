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