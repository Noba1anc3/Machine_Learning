## 什么是XGBoost?

- 全称：eXtreme Gradient Boosting
- 基础：GBDT
- 所属：Boosting迭代型、树类算法
- 适用范围：分类、回归
- 优点：速度快、效果好、能处理大规模数据、支持多种语言、支持自定义损失函数等
- 缺点：算法参数过多，对原理不清楚的很难使用好XGBoost，不适合处理超高维特征数据
- 项目地址：https://github.com/dmlc/xgboost

## XGBoost的原理

XGBoost 所应用的算法就是 gradient boosting decision tree，既可以用于分类也可以用于回归问题中。那什么是 Gradient Boosting？Gradient boosting 是 boosting 的其中一种方法。所谓 Boosting ，就是将弱分离器 f_i(x) 组合起来形成强分类器 F(x) 的一种方法。所以 **Boosting 有三个要素**：

- A **loss function** to be optimized：例如**分类问题中用 cross entropy，回归问题用 mean squared error**。
- A **weak learner** to make predictions：例如[决策树](https://www.biaodianfu.com/decision-tree.html)。
- An **additive model**：将多个弱学习器累加起来组成强学习器，进而使目标损失函数达到极小。

Gradient boosting 就是通过加入新的弱学习器，来努力纠正前面所有弱学习器的残差，最终这样多个学习器相加在一起用来进行最终预测，准确率就会比单独的一个要高。之所以称为 Gradient，是因为在添加新模型时使用了梯度下降算法来最小化的损失。一般来说，gradient boosting 的实现是比较慢的，因为每次都要先构造出一个树并添加到整个模型序列中。而 XGBoost 的特点就是**计算速度快，模型表现好**，这两点也正是这个项目的目标。

**XGBoost的优势**

XGBoost算法可以给预测模型带来能力的提升。当我对它的表现和高准确率背后的原理有更多了解的时候，你会发现它具有很多优势：

- **正则化**
  - XGBoost在代价函数里加入了正则项，用于控制模型的复杂度。
  - 正则项里包含了**树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和**。
  - 从Bias-variance tradeoff角度来讲，正则项降低了模型的variance，使学习出来的模型更加简单，**防止过拟合**，这也是xgboost优于传统GBDT的一个特性。
- **并行**
  - Boosting不是一种串行的结构吗，怎么并行的？注意XGBoost的并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。
  - XGBoost的并行是在**特征粒度**上的。我们知道，**决策树的学习最耗时的一个步骤就是对特征值进行排序（因为要确定最佳分割点）**，XGBoost在训练之前，**预先对数据进行了排序，然后保存为block结构**，后面的迭代中重复地使用这个结构，大大减小计算量。
  - 这个block结构使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的特征做分裂，那么各个特征的增益计算就可以开多线程进行。
- **高度灵活**
  - XGBoost支持用户**自定义目标函数和评估函数**，只要目标函数**二阶可导**。
- **缺失值处理**
  - XGBoost内置处理缺失值的规则。用户需要**提供一个和其它样本不同的值**，然后把它作为一个参数传进去，以此来作为缺失值的取值。
  - XGBoost在不同节点遇到缺失值时采用不同的处理方法，并且会**学习未来遇到缺失值时的处理方法**。
- **剪枝**
  - 当分裂时遇到一个负损失时，GBM会停止分裂。因此GBM实际上是一个贪心算法。
  - **XGBoost会一直分裂到指定的最大深度(max_depth)，然后回过头来剪枝**。如果某个节点之后不再有正值，它会去除这个分裂。
  - 这种做法的优点，当**一个负损失（如-2）后面有个正损失（如+10）**的时候，就显现出来了。GBM会在-2处停下来，因为它遇到了一个负值。但是XGBoost会继续分裂，然后发现这两个分裂综合起来会得到+8，因此会保留这两个分裂。比起GBM，这样**不容易陷入局部最优解**。
- **内置交叉验证**
  - XGBoost允许在每一轮boosting迭代中使用交叉验证。
  - 因此，可以方便地获得最优boosting迭代次数。而GBM使用网格搜索，只能检测有限个值。
- **在已有的模型基础上继续**
  - XGBoost可以在上一轮的结果上继续训练。

## 基础知识——GBDT

XGBoost是在GBDT的基础上对boosting算法进行的改进，内部决策树使用的是回归树，简单回顾GBDT如下：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/09/gbdt.png)

回归树的分裂结点：

对于平方损失函数，拟合的是残差；

对于一般损失函数（梯度下降），拟合的是残差的近似值，分裂结点划分时枚举所有特征的值，选取划分点。

最后预测的结果是每棵树的预测结果相加。

## XGBoost算法原理知识

**定义树的复杂度**

1、把树拆分成树结构部分q和叶子权重部分w

![img](https://www.biaodianfu.com/wp-content/uploads/2020/09/xgboost-1.png)

2、树的复杂度函数和样例

![img](https://www.biaodianfu.com/wp-content/uploads/2020/09/xgboost-2.png)

定义树的结构和复杂度的原因很简单，这样就可以衡量模型的复杂度了啊，从而可以有效控制过拟合。

**XGBoost中的boosting tree模型**

![img](https://www.biaodianfu.com/wp-content/uploads/2020/09/xgboost-3.png)

和传统的boosting tree模型一样，XGBoost的提升模型也是采用的残差（或梯度负方向），不同的是分裂结点选取的时候不一定是最小平方损失。

![1611881437434](C:\Users\yi\AppData\Roaming\Typora\typora-user-images\1611881437434.png)

**对目标函数的改写——二阶泰勒展开（关键）**

![img](https://www.biaodianfu.com/wp-content/uploads/2020/09/xgboost-5.png)

最终的目标函数只依赖于每个数据点的在误差函数上的一阶导数和二阶导数。这么写的原因很明显，由于之前的目标函数求最优解的过程中只对平方损失函数时候方便求，对于其他的损失函数变得很复杂，通过二阶泰勒展开式的变换，这样求解其他损失函数变得可行了。当定义了分裂候选集合的时候Ij={i|q(xi)=j}。可以进一步改目标函数。分裂结点的候选集是很关键的一步，这是xgboost速度快的保证，怎么选出来这个集合，后面会介绍。

![1611881685547](C:\Users\yi\AppData\Roaming\Typora\typora-user-images\1611881685547.png)

求解：

![img](https://www.biaodianfu.com/wp-content/uploads/2020/09/xgboost-7.png)

**树结构的打分函数**

Obj代表了当指定一个树的结构的时候，在目标上面最多减少多少？

![img](https://www.biaodianfu.com/wp-content/uploads/2020/09/xgboost-8.png)

对于每一次尝试去对已有的叶子加入一个分割

![img](https://www.biaodianfu.com/wp-content/uploads/2020/09/xgboost-9.png)

这样就可以在建树的过程中动态的选择是否要添加一个结点。

![img](https://www.biaodianfu.com/wp-content/uploads/2020/09/xgboost-10.png)

假设要枚举所有x < a 这样的条件，对于某个特定的分割a，要计算a左边和右边的导数和。对于所有的a，我们只要做一遍从左到右的扫描就可以枚举出所有分割的梯度和GL、GR。然后用上面的公式计算每个分割方案的分数就可以了。

**寻找分裂结点的候选集**

1、暴力枚举

2、近似方法 ，近似方法通过特征的分布，按照百分比确定一组候选分裂点，通过遍历所有的候选分裂点来找到最佳分裂点。两种策略：全局策略和局部策略。

- 在全局策略中，对每一个特征确定一个全局的候选分裂点集合，就不再改变；
- 在局部策略中，每一次分裂都要重选一次分裂点。

前者需要较大的分裂集合，后者可以小一点。对比补充候选集策略与分裂点数目对模型的影响。全局策略需要更细的分裂点才能和局部策略差不多

3、Weighted Quantile Sketch

![img](https://www.biaodianfu.com/wp-content/uploads/2020/09/xgboost-11.png)

陈天奇提出并从概率角度证明了一种带权重的分布式的Quantile Sketch。

参考链接：

- [通俗理解kaggle比赛大杀器xgboost](https://blog.csdn.net/v_july_v/article/details/81410574)
- https://fuhailin.github.io/XGBoost/
