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

