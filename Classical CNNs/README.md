# LeNet-5

LeNet-5 comes from < Gradient-Based Learning Applied to Document Recognition > [1998]

![](https://cuijiahua.com/wp-content/uploads/2018/01/dl_3_4.jpg)

1. Input Layer
   - 32 x 32

2. C1 Layer
   - Input Size : 32 x 32 x 1
   - Kernel Size : 5 x 5
   - Kernel Num : 6
   - Kernel Stride : 1
   - Kernel Padding : Valid
   - Output Size : 28 x 28 x  6
   - Trainable Parameters : (5 * 5 + 1) * 6
   
3. S2 Layer
   - Input Size : 28 x 28 x 6
   - Kernel Size : 2 x 2
   - Kernel Stride : 2
   - Output Size : 14 x 14 x 6
   
4. C3 Layer
   - Input Size : 14 x 14 x 6
   - Kernel Size : 5 x 5
   - Kernel Num : 16 ( 6 + 6 + 3 + 1)
   - Kernel Stride : 1
   - Kernel Padding : Valid
   - Output Size : 10 x 10 x 16
   - Trainable Parameters : 6 * (3 * 5 * 5 + 1) + 6 * (4 * 5 * 5 + 1) + 3 * (4 * 5 * 5 + 1) + 1 * (6 * 5 * 5 + 1) 

![](https://cuijiahua.com/wp-content/uploads/2018/01/dl_3_5.png)

5. S4 Layer
   - Input Size : 10 x 10 x 16
   - Kernel Size : 2 x 2
   - Kernel Stride : 2
   - Output Size : 5 x 5 x 16
   
6. C5 Layer
   - Input Size : 5 x 5 x 16
   - Kernel Size : 5 x 5
   - Kernel Num : 120
   - Output Size : 1 x 1 x 120
   - Trainable Parameters : 120 * (16 * 5 * 5 + 1)

7. F6 Layer
   - Input Size : 1 x 1 x 120
   - Output Size : 1 x 1 x 84

8. Output Layer
   - Input Size : 1 x 1 x 84
   - Output Size : 1 x 1 x 10

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

# VGGNet

​	VGGNet is a convolutional neural network jointly developed by the Visual Geometry Group of Oxford University and Google Deepmind. In the ILSVRC-2014 classification competition, VGGNet achieved a 7.32% error rate of top-5 and won the second place. Although it ranked second, the gap was very small, and VGGNet performed better than the latter in various transfer learning tasks.

​	Since it is the same year as GoogLeNet, it is not surprising that there are some similar ideas in the design. For example, they are all designed to study the influence of depth on neural networks. They are all based on the structure of AlexNet for model construction, and some ideas of network in network are used.

Paper : [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

## Network Structure

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/07/02/VGG/VGG-16_20200702.png)

- VGGNet uses 3 × 3 kernel size and 2 × 2 pooling windows. Different from AlexNet, all kernels in VGG uses 3 × 3 convolution kernel. It is not difficult to find that the receptive field obtained by 5 × 5 convolution is the same as that of two 3 × 3 convolutions, and the receptive field of 7 × 7 convolution is the same as that of three 3 × 3 convolutions. However, compared with the large convolution, the advantage of 3 × 3 convolution is that: 
  - The num of  parameters is small : assuming that the input and output are all C channels, the convolution of 7 × 7 requires 7 × 7 × C × C = 49C^2 parameters, while the 3 × 3 convolution of three stacks only needs (3 × 3 × C × C) × 3 = 27C^2 parameters.

- Removed LRN used in AlexNet and GoogLeNet, the author finds that LRN does not improve the accuracy, but increases the time and memory consumption

- According to different depths, the author has carried out several groups of comparative experiments. The final result is that VGG-16 and VGG-19 are obviously better than other networks. Therefore, what we usually call VGG or VGGNet generally refers to VGG-16 or VGG-19. The specific network structure is shown in the figure below. The same thing with GoogLeNet is that the author also tried 1 × 1 convolution, and the conclusion is: 1 × 1 convolution is effective, but the effect is not as good as 3 × 3 convolution.

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/07/02/VGG/VGGNet_20200702.png)

## Summary

- Compared with previous AlexNet and GoogLeNet, VGG has an obvious advantage: simple structure. 

  VGG gives two conclusions :

  - It is better to replace 5 × 5 or 7 × 7 convolution with 3 × 3 convolution kernel
  - It is verified that increasing depth can improve network performance

- The disadvantage of VGGNet is that the number of parameters is too large. VGG-16 contains 138M parameters, the full connection layer contains ~124M parameters, and the full connection layer of the first layer contains 7 × 7 × 512 × 4096 ~103M parameters. That is to say, most of the parameters are generated in the first layer of full connection layer. Therefore, in order to reduce the parameters of VGG, we must start from full connection layer.

- One interesting thing about VGG is that with the deepening of the network, the size will be halved and the depth will be doubled.

# GoogLeNet

​	The most direct way to improve network performance is to build deeper and larger models, but directly increasing the depth and width of the model will often lead to the following problems :

- Over Fitting
- Calculation Complexity
- Gradient Vanishing

​    The basic idea to solve the above problem is to use sparse connection instead of dense connection, but the computing of sparse connection is very inefficient (the cost of searching and caching is high). The solution to this problem is to cluster the sparse matrix into several dense sub matrices for calculation. Under the guidance of this idea, GoogLeNet was born, and won the ILSVRC-2014 category competition champion with an error rate of 6.67% in top-5.

​	The name of GoogLeNet comes from Google on the one hand, and it is also a tribute to LeNet-5. In order to solve the problem of over fitting and cost calculation, the model proposed a very creative structure, Inception. In the paper, the author also cites the image of "we need to go deep" from the movie inception of the same name, which also shows that the starting point of GoogLeNet and Inception is to build a deeper and larger network (of course, under the premise of considering the feasibility).

## Network Structure

### Inception v1

#### Original Inception

​	Here is a basic structure of the original Inception :

![](https://static.oschina.net/uploads/space/2018/0317/141510_fIWh_876354.png)

​	The structure stacks common convolution (1 x 1, 3 x 3, 5 x 5) and pooling (3 x 3) in CNN. On the one hand, it increases the width of the network, on the other hand, it also increases the adaptability of the network to scale. Above these layers, a ReLU operation must be done after each convolution layer to increase the nonlinear characteristics of the network.

​	The author thinks that max pooling can also play a role in feature extraction, and the stride of max pooling layer here is 1, which ensures the same width and height before and after pooling.

​	Advantage of the structure :

- Multi-scale convolution can extract features of different scales

- Sparse matrix is decomposed into dense matrix to accelerate convergence

  - In traditional convolution, only one size of convolution (such as 3 × 3, 256 channel number) is used, so the 256 output features are all extracted from the size of 3 × 3, they are sparse. 

    After using the Inception structure, the final output will aggregate the results of 1 × 1 convolution, 3 × 3 convolution and 5 × 5 convolution together, this is equivalent to a number of densely distributed subsets. Features with strong correlation are gathered together, irrelevant features will be weakened, so the information with higher "purity" can be extracted, and the convergence speed will naturally be faster.

- Herbert's theory, a biological concept, roughly means that if two neurons are always excited at the same time, they will form a "combination" to promote each other.

#### 1 x 1 Convolution

​	Imagine such a structure, the input size is 28 × 28 × 192, and the output size is 28 × 28 × 32. Through the 5 × 5 convolution in the Inception, the number of times of multiplication is calculated as follow : 28 × 28 × 32 feature points are output, so a total of 28 × 28 × 32 convolution calculations are needed. Each convolution calculation requires 5 × 5 × 192 times of multiplication. Therefore, a total of 28 × 28 × 32 × 5 × 5 × 192 ~ 120 million multiplication calculations are needed.

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/06/28/GoogLeNet/computational_cost_20200628.png)

​	Only one convolution requires hundreds of millions of multiplications. Even today, the amount of computation is still unacceptable. In order to solve this problem, it is necessary to reduce the dimension of input and then convolute. In the above example, a 1 × 1 convolution is added, as shown in the following figure. A total of 28 × 28 × 16 × 1 × 1 × 192 + 28 × 28 × 32 × 5 × 5 × 16 ~ 12 million times of multiplication calculation are needed, which greatly reduces the amount of calculation.

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/06/28/GoogLeNet/computational_cost_with_dimensionality_reduction_20200628.png)

​	Using the above method to optimize the Inception structure, a 1 × 1 convolution is added before the 3 × 3 and 5 × 5 convolutions, and a 1 × 1 convolution is added after the max pooling layer. For the same size of receptive field, adding more convolutions can get more abundant features, which is another advantage of 1 × 1 convolution besides dimension reduction. The 1 × 1 convolution is after the maximum pooling layer, not before, because the pooling layer is to extract the original features of the image, once it is connected after the 1 × 1 convolution, it will lose its original intention.

Advantage of 1 x 1 Convolution :

- Reduce Dimension
- Get More Features

Some thinking : If the feature map is reduced from high dimension to low dimension, there will be information loss. The author explained this point in this paper : "This is based on the success of embeddings: even low dimensional embeddings might contain a lot of information about a relatively large image patch.” Before dimensionality reduction, the feature map is large, but there are also many "useless information" or "weak feature information". This "bottleneck" structure is equivalent to a kind of information compression, in which a small amount of "weak information" is lost in exchange for a large amount of calculation. From another point of view, 1 × 1 convolution is equivalent to making a full connection to the number of different channels in each position of the feature map. This full connection will reduce the number of neurons, thus achieving the effect of compressing the feature dimension.

#### Inception V1

![](https://static.oschina.net/uploads/space/2018/0317/141520_31TH_876354.png)

### GoogLeNet

- After understanding the structure of Inception, it is very easy to look at the structure of GoogLeNet. The whole model uses 9 Inception structures (a max pooling layer will be added to the structure above for some), with a total of 22 layers. 
- All convolution layers are activated using the ReLU function.

- The specific parameter table is attached below the structure diagram. 
  - The size of the input image is 224 × 224 × 3. The values of "#3 × 3 reduce" and "#5 × 5 reduce" represent the number of 1 × 1 convolution kernels used for dimensionality reduction before 3 × 3 convolution and 5 × 5 convolution, respectively.

- After convolution, an average pooling layer is used to replace the full connection layer, which greatly reduces the parameters.
  - If full connection layer is used here, 7 × 7 × 1024 × 1024 = ~ 51M parameters will be added. However, the total parameters of the whole model are only ~ 6.8M. This is why GoogLeNet has 22 layers, but the parameter quantity is only one tenth of that of AlexNet. The idea of using global average pooling instead of full connection comes from [Network in Network]( https://arxiv.org/pdf/1312.4400.pdf ), which reduces the number of parameters and improves the interpretability of the model.

- The three softmax functions are set to prevent gradient vanishing. 
  - In the training process, three losses are calculated and given different weights, and then added. In the paper, the loss of the first two auxiliary classifiers is given a weight of 0.3; in the prediction process, the first two auxiliary classifiers are removed.

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/06/28/GoogLeNet/GoogLeNet_20200628.png)

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/06/28/GoogLeNet/GoogLeNet_parameter_20200628.png)


# ResNet

​	ResNet proposed in 2015, the maximum depth reached 152 layers, and won the champion of ILSVRC classification competition of that year, with an error rate of 3.57% of top-5.

​	[Deep Residual Learning for Image Recognition]( https://arxiv.org/abs/1512.03385 )

## Network Degradation

​	Suppose you have an optimized network structure, which is 18 layers. When we design the network structure, we don't know how many layers of network is the optimal network structure. So many 16 layers are redundant. We hope that in the process of training the network, the model can train these 16 layers as identity mapping, that is, the input and output are exactly the same when passing through this layer. However, it is often difficult for the model to learn the parameters of the 16 layer identity mapping correctly, so the performance of the optimized 18 layer network structure will not be better than that of the optimized 18 layer network structure, which means that the model will degenerate with the increase of network depth. It is not produced by over fitting, but by redundant network layer learning parameters that are not identical mapping.

​	The following figure shows the results of 20 layer and 56 layer convolutional neural network training on CIFAR-10. No matter in training set or test set, the performance of deeper 56 layer is worse than that of 20 layer, which is called network degradation. ResNet is mainly proposed to solve this problem.

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/07/03/ResNet/Network_Degradation.png)

## Shortcut Connection

​	As illustrated above, the effect of 56 layers is not as good as that of 20 layers, which shows that the extra 36 layers can not carry out identity transformation (that is, the stack of multiple nonlinear transformations cannot be approximate to identity transformation). Therefore, we need to provide the possibility of identity transformation to prevent network degradation.

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/07/03/ResNet/Shortcut_Connection.png)

​	We find that, assuming that the layer is redundant, before introducing ResNet, we want the parameters learned by this layer to satisfy L(x) = x, that is, the input is X, and after passing through the redundant layer, the output is still X. But we can see that it is difficult to learn the parameters of this layer for identity mapping. ResNet wants to avoid learning the parameters of identity mapping in this layer. It uses the structure shown in the figure above, let L (x) = F(x) + x; here, F(x) is called residual term. We find that if we want the redundant layer to be able to identity mapping, we only need to learn F(x) = 0. Learning F(x) = 0 is simpler than learning L(x) = x, because the initialization of parameters in each layer of the network is generally biased to 0. In this way, compared with updating the parameters of the network layer to learn L(x) = x, the redundant layer can converge faster by learning the update parameters of F(x) = 0.

​	Also, Relu can activate the negative number to 0, filter the linear change of negative number, and make F(x) = 0 faster. In this way, when the network decides which network layers are redundant layers, the network using ResNet solves the problem of learning identity mapping fast.

## Network Structure

### Residual Block

​	Based on the above ideas, the author gave two kinds of residual block design, as shown in the figure below. ResNet18 and ResNet34 use the structure on the left of the figure below. Nowadays, ResNet50 and ResNet101, which are commonly used for feature extraction, use the structure on the right of the figure below. Among them, 1 × 1 convolution is mainly used to reduce and increase dimension, so as to reduce the amount of calculation. Because the number of channels in the middle is less than that on both sides, this structure is called "bottleneck" structure.

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/07/03/ResNet/Residual_Block.png)

### ResNet

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/07/03/ResNet/ResNet.png)

### ResNet50

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/07/03/ResNet/ResNet50.png)

### Residual Block

​	ResNet has two residual blocks, one is identity block, the dimension of input and output is the same, so multiple blocks can be connected in series; the other residual block is conv block, the dimension of input and output is different, so it can not be connected in series. Its function is to change the dimension of feature vector.

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/07/03/ResNet/identity_block.png)

![](https://cdn.jsdelivr.net/gh/mao-jy/mao-jy.github.io/2020/07/03/ResNet/conv_block.png)
