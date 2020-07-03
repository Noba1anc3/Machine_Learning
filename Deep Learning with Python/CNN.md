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
activations = activation_model.predict(img_tensor)from keras import models
```

- First layer is a collection of various edge detectors
- With deeper and deeper, activation gets more and more abstract, and difficult to understand intuitively
- With layer more deep, less visual content and more class information
- The sparsity of activation grows with layer gets deeper

Deep NN can be used as information distillation pipeline. Input origin data, transform many times, filter irrelavant and magnify useful information.

### Visualize filter
