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

