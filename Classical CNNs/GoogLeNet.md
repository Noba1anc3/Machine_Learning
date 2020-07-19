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

