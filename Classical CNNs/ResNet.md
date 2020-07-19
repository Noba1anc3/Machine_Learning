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