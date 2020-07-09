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