# 长短期记忆网络LSTM

## 普通RNN存在的问题

[循环神经网络（Recurrent Neural Network，RNN）](https://www.biaodianfu.com/rnn.html)是一种用于处理序列数据的神经网络。相比一般的神经网络来说，他能够处理序列变化的数据。比如某个单词的意思会因为上文提到的内容不同而有不同的含义，RNN就能够很好地解决这类问题。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/rnn-1.jpg)

在上图所示的神经网络A中，输入为Xt，输出为ht。A上的环允许将每一步产生的信息传递到下一步中。一个RNN可以看作是同一个网络的多份副本，每一份都将信息传递到下一个副本。将环展开：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/rnn-2-1.png)

在过去的几年里，RNN在一系列的任务中都取得了令人惊叹的成就，比如语音识别，语言建模，翻译，图片标题等等。关于RNN在各个领域所取得的令人惊叹的成就。

有时，我们只需要看最近的信息，就可以完成当前的任务。比如，考虑一个语言模型，通过前面的单词来预测接下来的单词。如果我们想预测句子“the clouds are in the sky”中的最后一个单词，我们不需要更多的上下文信息——很明显下一个单词应该是sky。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/rnn-3.jpg)

然而，有时候我们需要更多的上下文信息。比如，我们想预测句子“I grew up in France… I speak fluent French”中的最后一个单词。最近的信息告诉我们，最后一个单词可能是某种语言的名字，然而如果我们想确定到底是哪种语言的话，我们需要France这个更远的上下文信息。实际上，相关信息和需要该信息的位置之间的距离可能非常的远。不幸的是，随着距离的增大，RNN对于如何将这样的信息连接起来无能为力。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/rnn-4-1.png)

## LSTM网络简介

LSTM，全称为长短期记忆网络(Long Short Term Memory networks)，是一种特殊的RNN。LSTM由Hochreiter & Schmidhuber (1997)提出，许多研究者进行了一系列的工作对其改进并使之发扬光大。LSTM在许多问题上效果非常好，现在被广泛使用。

LSTM在设计上明确地避免了长期依赖的问题。所有的循环神经网络都有着重复的神经网络模块形成链的形式。在普通的RNN中，重复模块结构非常简单，例如只有一个tanh层。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/lstm-1.jpg)

LSTM也有这种链状结构，不过其重复模块的结构不同。LSTM的重复模块中有4个神经网络层，并且他们之间的交互非常特别。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/lstm-2.png)

现在暂且不必关心细节，稍候我们会一步一步地对LSTM的各个部分进行介绍。开始之前，我们先介绍一下将用到的标记。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/lstm-3.jpg)

在上图中，每条线表示向量的传递，从一个结点的输出传递到另外结点的输入。粉红圆表示向量的元素级操作，比如相加或者相乘。黄色方框表示神经网络的层。线合并表示向量的连接，线分叉表示向量复制。

## LSTM网络的实现原理

LSTM的主要思想是采用一个叫做“**细胞状态(state)**”的通道来贯穿整个时间序列。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/lstm-4.png)

细胞状态有点像是传送带，它直接穿过整个链，同时只有一些较小的线性交互。上面承载的信息可以很容易地流过而不改变。

通过精心设计“门”的结构来去除或增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。它们包含一个 sigmoid 神经网络层和一个逐元乘法操作。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/lstm-6.jpg)

Sigmoid层输出0~1之间的值，每个值表示对应的部分信息是否应该通过。0值表示不允许信息通过，1值表示让所有信息通过。一个LSTM有3个这种门，来保护和控制元胞状态。

### 遗忘门

“遗忘门”决定之前状态中的信息有多少应该舍弃。它会读取 ht−1 和 xt的内容,σ符号代表Sigmoid函数，它会输出一个0到1之间的值。其中0代表舍弃之前细胞状态Ct−1中的内容，1代表完全保留之前细胞状态Ct−1中的内容。0、1之间的值代表部分保留之前细胞状态Ct−1中的内容。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/lstm-7.png)

### 输入门

“输入门”决定什么样的信息保留在细胞状态Ct中，它会读取 ht−1 和 xt 的内容,σ符号代表Sigmoid函数，它会输出一个0到1之间的值。和“输入门”配合的还有另外一部分，即下图中计算tanh层的部分，这部分输入也是 ht−1 和 xt，不过采用tanh激活函数，将这部分标记为c~(t)，称作为“候选状态”。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/lstm-8.png)

### 细胞状态更新

由Ct−1计算得到Ct。旧“细胞状态”Ct−1和“遗忘门”的结果进行计算，决定旧的“细胞状态”保留多少，忘记多少。接着“输入门”i(t)和候选状态c~(t)进行计算，将所得到的结果加入到“细胞状态”中，这表示新的输入信息有多少加入到“细胞状态”中。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/lstm-9.png)

### 输出门

和其他门计算一样，它会读取 ht−1 和 xt 的内容,然后计算Sigmoid函数，得到“输出门”的值。接着把“细胞状态”通过tanh进行处理(得到一个在-1到1之间的值)，并将它和输出门的结果相乘，最终得到确定输出的部分。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/10/lstm.jpg)

### 总结

以上，就是LSTM的内部结构。通过门控状态来控制传输状态，记住需要长时间记忆的，忘记不重要的信息；而不像普通的RNN那样只能够“呆萌”地仅有一种记忆叠加方式。对很多需要“长期记忆”的任务来说，尤其好用。但也因为引入了很多内容，导致参数变多，也使得训练难度加大了很多。

## 使用LSTM对IMDB评论进行情感分析

**from** keras.datasets **import** imdb

**from** *keras.layers* **import** LSTM, Dense, Embedding

**from** *keras.models* **import** Sequential

**from** *keras.preprocessing* **import** sequence

max_features = 20000

\# cut texts after this number of words (among top max_features most common words)

maxlen = 80

batch_size = 32

print("Loading data...")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), "train sequences")

print(len(x_test), "test sequences")

print("Pad sequences (samples x time)")

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print("x_train shape:", x_train.shape)

print("x_test shape:", x_test.shape)

print("Build model...")

model = Sequential()

model.add(Embedding(max_features, 128))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation="sigmoid"))

\# try using different optimizers and different optimizer configs

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Train...")

model.fit(

​    x_train, y_train, batch_size=batch_size, epochs=15, validation_data=(x_test, y_test)

)

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print("Test score:", score)

print("Test accuracy:", acc)

参考链接：

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- https://keras.io/examples/imdb_lstm/