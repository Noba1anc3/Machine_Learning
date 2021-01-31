# 循环神经网络RNN

## 循环神经网络RNN简介

传统的神经网络是层与层之间是全连接的，但是每层之间的神经元是没有连接的（其实是假设各个数据之间是独立的），这种结构不善于处理序列化的问题。比如要预测句子中的下一个单词是什么，这往往与前面的单词有很大的关联，因为句子里面的单词并不是独立的。下图是一个[全连接网络](https://www.biaodianfu.com/imdb-sentiment-analysis-full-connection.html)，它的隐藏层的值只取决于输入的 x：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/FC.png)

循环神经网络（Recurrent Neural Network, RNN）是根据“**人的认知是基于过往经验和记忆**”这一观点提出的，它不仅考虑当前时刻的输入，还考虑对前面内容记忆。即RNN对之前发生在数据序列中的事是有一定记忆的，对处理有序列的问题效果比较好。**RNN是一类以序列（sequence）数据为输入，在序列的演进方向进行递归（recursion）且所有节点（循环单元）按链式连接的递归神经网络（recursive neural network）。**RNN 的隐藏层的值s不仅仅取决于当前这次的输入x，还取决于上一次隐藏层的值s。这个过程画成简图是这个样子：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/RNN.png)

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/RNN-2.jpg)

其中，t 是时刻， x 是输入层， s 是隐藏层， o 是输出层，矩阵 W 就是隐藏层上一次的值作为这一次的输入的权重。

传统的神经网络中间层每一个神经元和输入的每一个数据进行运算得到一个激励然后产生一个中间层的输出，并没有记忆能力，在输入为序列的情况下的效果有限。而很多东西是受到时域信息影响的，某一时刻产生的激励对下一时刻是有用的，递归神经网络就具备这样的记忆能力，它可以把这一刻的特征与上一刻保存的激励信息相结合(并不只是上一刻s2是由s1,s2产生的，sn不仅有sn−1的信息，还包括sn−2,sn−3,…,s1的信息。

**[CNN](https://www.biaodianfu.com/cnn.html)与RNN的区别**

| 类别   | 特点描述                                                     |
| ------ | ------------------------------------------------------------ |
| 相同点 | 1、 传统神经网络的扩展。2、 前向计算产生结果，反向计算模型更新。3、 每层神经网络横向可以多个神经元共存,纵向可以有多层神经网络连接。 |
| 不同点 | 1、 CNN空间扩展，神经元与特征卷积；RNN时间扩展，神经元与多个时间输出计算2、 RNN可以用于描述时间上连续状态的输出，有记忆功能，CNN用于静态输出（前一输入跟下一输入是否有关联）3、 CNN 需要固定长度的输入、输出，RNN 的输入可以是不定长的4、 CNN 只有 one-to-one 一种结构，而 RNN 有多种结构 |

## 循环神经网络RNN应用场景

标准的全连接神经网络（fully connected neural network）处理序列会有两个问题：

1）全连接神经网络输入层和输出层长度固定，而不同序列的输入、输出可能有不同的长度，选择最大长度并对短序列进行填充（pad）不是一种很好的方式；

2）全连接神经网络同一层的节点之间是无连接的，当需要用到序列之前时刻的信息时，全连接神经网络无法办到，一个序列的不同位置之间无法共享特征。而循环神经网络（Recurrent Neural Network，RNN）可以很好地解决问题。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/sequuence-data.png)

RNN已经被在实践中证明对NLP是非常成功的。如词向量表达、语句合法性检查、词性标注等。在RNN中，目前使用最广泛最成功的模型便是LSTMs(Long Short-Term Memory，长短时记忆模型)模型，该模型通常比vanilla RNNs能够更好地对长短时依赖进行表达，该模型相对于一般的RNN，只是在隐藏层做了手脚。对于LSTM，后面会进行详细地介绍。下面对RNN在NLP中的应用进行简单的介绍。

**语言模型与文本生成(Language Modeling and Generating Text)**

给你一个单词序列，我们需要根据前面的单词预测每一个单词的可能性。语言模型能够一个语句正确的可能性，这是机器翻译的一部分，往往可能性越大，语句越正确。另一种应用便是使用生成模型预测下一个单词的概率，从而生成新的文本根据输出概率的采样。语言模型中，典型的输入是单词序列中每个单词的词向量(如 One-hot vector)，输出时预测的单词序列。当在对网络进行训练时，如果，那么这一步的输出便是下一步的输入。

**机器翻译(Machine Translation)**

机器翻译是将一种源语言语句变成意思相同的另一种源语言语句，如将英语语句变成同样意思的中文语句。与语言模型关键的区别在于，需要将源语言语句序列输入后，才进行输出，即输出第一个单词时，便需要从完整的输入序列中进行获取。

**语音识别(Speech Recognition)**

语音识别是指给一段声波的声音信号，预测该声波对应的某种指定源语言的语句以及该语句的概率值。

**图像描述生成 (Generating Image Descriptions)**

和卷积神经网络(convolutional Neural Network, CNN)一样，RNN已经在对无标图像描述自动生成中得到应用。将CNN与RNN结合进行图像描述自动生成。这是一个非常神奇的研究与应用。该组合模型能够根据图像的特征生成描述。

## 循环神经网络RNN结构

以 (Vanilla) Recurrent Neural Network 为例：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/Vanilla.png)

首先看一个简单的循环神经网络，它由输入层、一个隐藏层和一个输出层组成：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/RNN-2.png)

对于 RNN，一个非常重要的概念就是时刻。RNN 会对每一个时刻的输入结合当前模型的状态给出一个输出。图 中，t时刻 RNN 的主体结构 A 的输入除了来自输入层Xt，还有一个循环的边来提供从t−1时刻传递来的隐藏状态。RNN 可以被看作是同一个神经网络结构按照时间序列复制的结果。下图展示了一个展开的 RNN。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/RNN-3.png)

从 RNN 的展开结构可以很容易得出它最擅长解决的问题是与时间序列相关的。RNN 也是处理这类问题时最自然的神经网络结构。RNN 的主体结构 A 按照时间序列复制了多次，结构 A 也被称之为循环体。如何设计循环体 A 的网络结构是 RNN 解决实际问题的关键。和[卷积神经网络（CNN）](https://www.biaodianfu.com/cnn.html)过滤器中参数共享类似，在 RNN 中，循环体A中的参数在不同时刻也是共享的。下图展示了一个最简单的使用单个全连接层作为循环体A的 RNN，图中黄色的 tanh 小方框表示一个使用 tanh 作为激活函数的全连接层。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/RNN-4.png)

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/RNN-5.png)

t时刻循环体A的输入包括Xt和从t−1时刻传递来的隐藏状态ht−1（依据copy 标志，t−1和t时刻循环体A之间连接的箭头即表示隐藏状态ht−1的传递）。循环体A的两部分输入如何处理呢？如图，将Xt和ht−1直接拼接起来，成为一个更大的矩阵/向量[Xt,ht−1]。假设Xt和ht−1的形状分别为[1, 3]和[1, 4]，则最后循环体A中全连接层输入向量的形状为[1, 7]。拼接完后按照全连接层的方式进行处理即可。

**为了将当前时刻的隐含状态ht转化为最终的输出yt，循环神经网络还需要另一个全连接层来完成这个过程**。这和卷积神经网络中最后的全连接层意义是一样的。RNN 的前向传播计算过程如下图所示：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/RNN-6.png)

RNN 的类型：

- One-to-one: Vanilla Neural Networks。最简单的结构，其实和全连接神经网络并没有什么区别，这一类别算不得是 RNN。
- One-to-many: Image Captioning, image -> sequence of works。输入不是序列，输出是序列。比如：输入一个图片，输出一句描述图片的话
- Many-to-one: Sentiment Classification, sequence of words -> sentiment。输入是序列，输出不是序列。比如：输入一句话，判断是正面还是负面情绪
- Many-to-many: Machine Translation, seq of words -> seq of words。输入和输出都是序列，但两者长度可以不一样。比如机器翻译。
- Many-to-many: Video classification on frame level。输出和输出都是序列，两者长度一样。比如：输入一个视频，判断每帧分类。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/RNN-7.png)

## RNN扩展和改进模型

这些年，研究者们已经提出了多种复杂的RNN去改进vanilla RNN模型的缺点。下面是目前常见的一些RNN模型：

### Simple RNN(SRN) 基本循环神经网络

**SRN是RNN的一种特例，它是一个三层网络，并且在隐藏层增加了上下文单元**，下图中的y便是隐藏层，u便是上下文单元。上下文单元节点与隐藏层中的节点的连接是固定的，并且权值也是固定的，其实是一个上下文节点与隐藏层节点一一对应，并且值是确定的。在每一步中，使用标准的前向反馈进行传播，然后使用学习算法进行学习。上下文每一个节点保存其连接的隐藏层节点的上一步的输出，即保存上文，并作用于当前步对应的隐藏层节点的状态，即**隐藏层的输入由输入层的输出与上一步的自己的状态所决定的**。因此SRN能够解决标准的多层感知机(MLP)无法解决的对序列数据进行预测的任务。SRN网络结构如下图所示：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/SRN.png)

### Bidirectional RNN 双向循环神经网络

Bidirectional RNN(双向网络)的改进之处便是，假设当前的输出(第t步的输出)不仅仅与前面的序列有关，并且还与后面的序列有关。例如：预测一个语句中缺失的词语那么就需要根据上下文来进行预测。Bidirectional RNN是一个相对较简单的RNN，是由两个RNN上下叠加在一起组成的。输出由这两个RNN的隐藏层的状态决定的。如下图所示：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/Bidirectional-RNN.png)

### Deep (Bidirectional) RNN 深度循环神经网络

Deep(Bidirectional) RNN与Bidirectional RNN相似，只是对于每一步的输入有多层网络。这样，该网络便有更强大的表达与学习能力，但是复杂性也提高了，同时需要更多的训练数据。Deep(Bidirectional)RNN的结构如下图所示：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/Deep-RNN.jpg)

### Gated Recurrent Unit Recurrent Neural Network 门控循环单元

GRU也是一般的RNN的改良版本，主要是从以下两个方面进行改进：

- 序列中不同的位置处的单词(以单词举例)对当前的隐藏层的状态的影响不同，越前面的影响越小，即**每个前面状态对当前的影响进行了距离加权，距离越远，权值越小**。
- 在产生误差error时，误差可能是由某一个或者几个单词而引发的，所以应当仅仅对对应的单词weight进行更新。

GRU的结构如下图所示：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/GRU.png)

GRU首先根据当前输入单词向量word vector已经前一个隐藏层的状态hidden state计算出update gate和reset gate。再根据reset gate、当前word vector以及前一个hidden state计算新的记忆单元内容(new memory content)。当reset gate为1的时候，new memory content忽略之前的所有memory content，最终的memory是之前的hidden state与new memory content的结合。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/GRU-2.png)

### LSTM Network 长短期记忆

LSTM与GRU类似，目前非常流行。它与一般的RNN结构本质上并没有什么不同，只是使用了不同的函数去去计算隐藏层的状态。在LSTM中，i结构被称为cells，可以把cells看作是黑盒用以保存当前输入xt之前的保存的状态ht−1，这些cells更加一定的条件决定哪些cell抑制哪些cell兴奋。它们结合前面的状态、当前的记忆与当前的输入。已经证明，该网络结构在对长序列依赖问题中非常有效。LSTM的网络结构如下图所示。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/LSTM.png)

LSTM与GRU的区别如图所示：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/LSTM-GRU.png)

从上图可以看出，它们之间非常相像，不同在于：

- new memory的计算方法都是根据之前的state及input进行计算，但是GRU中有一个reset gate控制之前state的进入量，而在LSTMs里没有这个gate
- 产生新的state的方式不同，LSTM有两个不同的gate，分别是forget gate (f gate)和input gate(i gate)，而GRU只有一个update gate(z gate)
- LSTM对新产生的state有一个output gate(o gate)可以调节大小，而GRUs直接输出无任何调节。

### Bidirectional LSTM

与bidirectional RNN 类似，bidirectional LSTM有两层LSTM。一层处理过去的训练信息，另一层处理将来的训练信息。在bidirectional LSTM中，通过前向LSTM获得前向隐藏状态，后向LSTM获得后向隐藏状态，当前隐藏状态是前向隐藏状态与后向隐藏状态的组合。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/Bidrectional-LSMT.png)

### Stacked LSTM

与Deep RNN类似，stacked LSTM通过将多层LSTM叠加起来得到一个更加复杂的模型。不同于bidirectional LSTM，stacked LSTMs只利用之前步骤的训练信息。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/Stacked-LSTM.png)

### Clockwork RNN(CW-RNN) 时钟频率驱动循环神经网络

CW-RNN是ICML2014上提出的一篇论文，与LSTM模型目的是相同的，就是为了解决经典的SRN对于长距离信息丢失的问题。但是与LSTM（基于三个门进行过滤和调节）的思想完全不同，CW-RNN利用的思想非常简单。CW在原文中作者表示其效果较SRN与LSTM都好。CW-RNN是一种使用时钟频率来驱动的RNN。它将隐藏层分为几个块(组，Group/Module)，每一组按照自己规定的时钟频率对输入进行处理。并且为了降低标准的RNN的复杂性，CW-RNN减少了参数的数目，提高了网络性能，加速了网络的训练。CW-RNN通过不同的隐藏层模块工作在不同的时钟频率下来解决长时间依赖问题。将时钟时间进行离散化，然后在不同的时间点，不同的隐藏层组在工作。因此，所有的隐藏层组在每一步不会都同时工作，这样便会加快网络的训练。并且，时钟周期小的组的神经元的不会连接到时钟周期大的组的神经元，只会周期大的连接到周期小的(认为组与组之间的连接是有向的就好了，代表信息的传递是有向的)，周期大的速度慢，周期小的速度快，那么便是速度慢的连速度快的，反之则不成立。

CW-RNN与SRN网络结构类似，也包括输入层(Input)、隐藏层(Hidden)、输出层(Output)，它们之间也有向前连接，输入层到隐藏层的连接，隐藏层到输出层的连接。但是与SRN不同的是，隐藏层中的神经元会被划分为若干个组，设为g，每一组中的神经元个数相同，设为k，并为每一个组分配一个时钟周期Ti∈{T1,T2,…,Tg}，每一个组中的所有神经元都是全连接，但是组j到组i的循环连接则需要满足大于Tj>Ti。如下图所示，将这些组按照时钟周期递增从左到右进行排序，即T1<T2<…<Tg，那么连接便是从右到左。例如：隐藏层共有256个节点，分为四组，周期分别是[1,2,4,8]，那么每个隐藏层组256/4=64个节点，第一组隐藏层与隐藏层的连接矩阵为64*64的矩阵，第二层的矩阵则为64*128矩阵，第三组为64*(3*64)=64*192矩阵，第四组为64*(4*64)=64*256矩阵。这就解释了上一段的后面部分，速度慢的组连到速度快的组，反之则不成立。

CW-RNN的网络结构如下图所示：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/CW-RNN.png)

参考链接：

- [循环神经网络(RNN, Recurrent Neural Networks)介绍](https://blog.csdn.net/heyongluoyao8/article/details/48636251)