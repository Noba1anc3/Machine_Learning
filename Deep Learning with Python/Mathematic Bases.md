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

