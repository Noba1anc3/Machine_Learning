# AlexNet

Winner Model in ILSVRC 2012, reached Top-5 Error Rate : 15.32%.

Paper came from the group of Hinton. The first author is Hinton's student Alex Krizhevsky 

Paper : [ImageNet Classification with Deep Convolutional Neural Networks]( http://117.128.6.17/cache/papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf?ich_args2=471-10210609042273_2c33356b33e836590b1305ee534853fd_10001002_9c896d28d4cbf7d5963c518939a83798_923622c10d2856fbe84390527e9c1b0e )

Widely Used Today :

- ReLU
- Dropout
- Local Response Normalization (like BN)

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/06/27/AlexNet/AlexNet_Andrew_Ng_20200627.png)

## Structure

- Convolution - LRN - Max Pooling - ReLU

- Convolution - LRN - Max Pooling - ReLU
- Convolution - ReLU
- Convolution - ReLU

- Convolution - Max Pooling - ReLU - Flatten
- Full Connection - ReLU - Dropout
- Full Connection - ReLU - Dropout
- Full Connection - Softmax

## Useful Components

### ReLU

​	Previously, we used sigmoid or tanh as the activation function of the model, but for gradient descent, unsaturated nonlinear functions (such as relu) are much faster than saturated nonlinear functions. The following figure shows the author's experiment on cifar-10 dataset, using a four layer convolution network structure, where the solid line represents the convergence process of relu activation function, and the dotted line represents the convergence process of tanh activation function. The training error rate reaching 0.25 is 6 times faster than tanh. The author also points out that the experimental results may vary with the network structure, but relu is always several times faster than saturated linear function.

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/06/27/AlexNet/ReLU_20200627.png)

Note : For function y = f(x), when derivative of function f(x) tends to 0 when x tends to positive infinity, called right saturate. When derivative of function f(x) tends to 0 when x tends to negative infinity, called left saturate. When f (x) satisfies both left saturation and right saturation, f (x) is called saturation function.

### Overlapping pooling

​	Different from traditional pooling, the step size of pooling layer used in alexnet is smaller than that of pooling window, which makes pooling window overlap. This method reduces the error rate of top-1 0.4% and top-5 0.3%, but overlapping pooling may be difficult to fit.

### Appendix : Network Structure in the Paper

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/06/27/AlexNet/AlexNet_20200627.png)

​	The network structure in this paper is roughly similar to the previous network structure, but some details are different. Moreover, due to the use of double GPU for calculation, some parts of the network structure are also for parallel operation, so the picture looks a little complicated. The missing part above is not the lack of screenshots, but the original picture in the paper.