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