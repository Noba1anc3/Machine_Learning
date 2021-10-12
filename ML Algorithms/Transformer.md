# Transformer

Transformer改进了RNN最被人诟病的训练慢的缺点，利用self-attention机制实现**快速并行**。并且Transformer可以增加到非常深的深度，充分发掘DNN模型的特性，提升模型准确率。 

 ![img](https://pic1.zhimg.com/80/v2-4b53b731a961ee467928619d14a5fd44_720w.jpg) 

在 Encoder 中，

1. Input 经过 embedding 后，要做 positional encodings
2. 然后是 Multi-head attention
3. 再经过 position-wise Feed Forward
4. 每个子层之间有残差连接

在 Decoder 中，

1. 如上图所示，也有 positional encodings，Multi-head attention 和 FFN，子层之间也要做残差连接
2. 但比 encoder 多了一个 Masked Multi-head attention
3. 最后要经过 Linear 和 softmax 输出概率

Transformer 的 encoder 由 6 个编码器叠加组成，decoder 也由 6 个解码器组成，在结构上都是相同的，但它们不共享权重。 

## Encoder

Encoder由N=6个相同的layer组成，layer指的就是上图左侧的单元，最左边有个“Nx”，这里是x6个。**每个Layer由两个sub-layer**组成，分别是**multi-head self-attention mechanism**和**fully connected feed-forward network**。其中每个sub-layer都加了**residual connection**和**normalization**，因此可以将sub-layer的输出表示为： 
$$
sub\_layer\_output=LayerNorm(x+SubLayer(x))
$$

### Multi-head Self-attention

$$
attention_\_output=Attention(Q,K,V)
$$

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}}V)
$$

### Feed-forward Network

提供非线性变换

Attention输出的维度是：[batch_size * seq_len, num_heads * head_size]

## Decoder

Decoder和Encoder的结构差不多，但是多了一个attention的sub-layer

这里先明确一下decoder的输入输出和解码过程： 

- 输入：encoder的输出 & 对应i-1位置decoder的输出。所以中间的attention不是self-attention，它的K，V来自encoder，Q来自上一位置decoder的输出 
- 输出：对应i位置的输出词的概率分布 

