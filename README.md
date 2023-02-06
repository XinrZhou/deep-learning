## machine-learning

**说明**：涉及数学公式，可安装**MathJax Plugin for Github**插件获得更好阅读体验

- [machine-learning](#machine-learning)
  - [机器学习算法](#机器学习算法)
    - [监督学习：Supervised learning](#监督学习supervised-learning)
    - [无监督学习：Unsupervised learning](#无监督学习unsupervised-learning)
    - [Terminology](#terminology)
  - [成本函数：J(w,b)](#成本函数jwb)
  - [梯度下降：找到局部最小值](#梯度下降找到局部最小值)
    - [梯度下降算法](#梯度下降算法)
    - [应用](#应用)
  - [多元线性回归](#多元线性回归)
  - [矢量化](#矢量化)
  - [梯度下降及多元线性回归](#梯度下降及多元线性回归)
  - [特征缩放](#特征缩放)
  - [两个技巧](#两个技巧)
    - [如何判断梯度下降是否收敛](#如何判断梯度下降是否收敛)
    - [怎样设置学习率](#怎样设置学习率)
  - [特征工程及多项式回归](#特征工程及多项式回归)
  - [逻辑回归](#逻辑回归)
    - [引例：使用线性回归解决分类问题](#引例使用线性回归解决分类问题)
    - [逻辑回归算法](#逻辑回归算法)
    - [决策边界](#决策边界)
    - [逻辑回归中的代价函数](#逻辑回归中的代价函数)
    - [简化逻辑回归代价函数](#简化逻辑回归代价函数)
    - [过拟合问题](#过拟合问题)
    - [正则化](#正则化)
  - [神经网络](#神经网络)
    - [简介](#简介)
    - [前向传播](#前向传播)
    - [反向传播](#反向传播)
  - [激活函数](#激活函数)
    - [常见激活函数](#常见激活函数)
    - [如何选择激活函数](#如何选择激活函数)
  - [多分类问题](#多分类问题)
    - [Softmax algorithm](#softmax-algorithm)
    - [Softmax Regression](#softmax-regression)
    - [高级优化方法](#高级优化方法)
  - [全连接层和卷积层](#全连接层和卷积层)
    - [全连接层](#全连接层)
    - [卷积层](#卷积层)
  - [模型评估](#模型评估)
    - [通过方差、偏差评估](#通过方差偏差评估)
    - [正则化](#正则化-1)
    - [学习曲线](#学习曲线)
    - [误差分析](#误差分析)
  - [数据增强与迁移学习](#数据增强与迁移学习)
    - [数据增强](#数据增强)
    - [迁移学习](#迁移学习)
  - [机器学习项目流程](#机器学习项目流程)
    - [流程](#流程)
    - [部署模型常见的方法](#部署模型常见的方法)
  - [精确率与召回率](#精确率与召回率)
    - [倾斜数据的误差指标](#倾斜数据的误差指标)
    - [精确率和召回率的权衡](#精确率和召回率的权衡)

### 机器学习算法

#### 监督学习：Supervised learning  

x(input) ——> y(output label), learnings from being given 'right answers'   

applications: machine translation, online advertising, self-driving car 

supervised algorithms 

- 回归算法：Regression ——> Predict a number
- 分类算法：Classification ——> Predict categories 

#### 无监督学习：Unsupervised learning

only x 

Unsupervised algorithms 

- Clustering：聚类算法，将未标记的数据自动分组到集群中 (google news, DNA microarray, grouping customers)
- Anomaly detection：异常检测
- Dimensionality reduction：降维

#### Terminology

Training set: data used to train the model 

x = "input" variable / feature 

y = "output" variable 

m = number of training examples 

(x,y) = single training examples 

(x<sup>(i)</sup>, y<sup>(i)</sup>) = i<sup>th</sup> training examples 

### 成本函数：J(w,b) 

用于衡量拟合程度

how to represent f？$f_{w,b} (x) = wx +b$ 单变量线性回归，实现它之前先定义一个成本函数 
$$
\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2  或  \frac{1}{2m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})^2
$$

### 梯度下降：找到局部最小值

![](https://image.jiqizhixin.com/uploads/editor/b7b9d1ad-4e48-455e-876c-70d37a191ca2/1531629686692.png)

#### 梯度下降算法

$$
w = w - \alpha\frac{\partial}{\partial w}J(w,b)\\
b = b - \alpha\frac{\partial}{\partial b}J(w,b)
$$

重复执行这两个步骤，直到算法收敛（达到局部最小值）。$\alpha$学习率，控制更新w,b的步长 

如何实现 w, b同步更新？
$$
tempW = w - \alpha\frac{\partial}{\partial w}J(w,b)\\
tempB = b - \alpha\frac{\partial}{\partial b}J(w,b)\\
w = tempW\\
b = tempB
$$
$\alpha$小：梯度下降慢，$\alpha$大：可能永远无法收敛

#### 应用

使用线性回归模型：$f_{w,b} (x) = wx +b$ 的均方误差代价函数：$\frac{1}{2m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})^2$，并利用梯度下降算法更新  

- 这种方式称为“批量梯度下降” 
- Batch gradient descent: Each step of gradient descent uses all the training examples 

### 多元线性回归

符号：$x_j$ = $j^{th}$ feature, n = number of the features,  $\vec{x}^{(i)}$ = feature of $i^{th}$ training examples 

模型：

- previously: $f_{w,b}(x) = wx+b$
- now: $f_{w,b}(x) = w_1x_1+w_2x_2+...+w_nx_n+b$ 

参数：$\vec{w} = [w_1, w_2, ..., w_n]$, b is a number 

简写：$f_{w,b}(\vec{x})=\vec{w}\cdot\vec{x}+b$

### 矢量化

$f_{\vec{w},b}(\vec{x})=\vec{w}\cdot\vec{x}+b$ 矢量化```f = np.dot(w,x) +b``` ，np.dot()实现了$\vec{w}$, $\vec{x}$的点积 

优点：代码简单，仅一行；运行效率高，调用了GPU

### 梯度下降及多元线性回归

参数：$\vec{w} = [w_1, w_2, ..., w_n]$, b is a number  

模型：$f_{\vec{w},b}(\vec{x})=\vec{w}\cdot\vec{x}+b$ 

成本函数：J($\vec{w}$,b) 

梯度下降算法

- $w_j = w_j - \alpha\frac{\partial}{\partial w_j}J(\vec{w_j},b)$
- $b = b - \alpha\frac{\partial}{\partial b}J(\vec{w_j},b)$

### 特征缩放

Feature Scaling，让梯度下降进行更快 

当特征的可能值很大时，其对应参数的合理取值很小；当特征的可能值很大时，其对应参数的合理取值比较大 

例：$300\leq{x_1}\leq2000$, $0\leq{x_2}\leq5$

- 特征缩放：$x_1.scaled = \frac{x_1}{2000}$, $x_2.scaled = \frac{x_2}{5}$
- 均值归一化：$x_1= \frac{x_1-\mu_1}{2000-300}$, $x_2= \frac{x_2-\mu_2}{5-0}$, $\mu_1$,$\mu_2$为均值
- Z-score标准化：$x_1=\frac{x_1-\mu_1}{\sigma_1}$, $x_2=\frac{x_2-\mu_2}{\sigma_2}$, $\sigma_1$,$\sigma_2$为标准差

当特征的可能值很大或很小时都需要进行特征缩放

### 两个技巧

#### 如何判断梯度下降是否收敛

画学习曲线：每次迭代J($\vec{w}$,b)都会减少，若迭代后增大，意味着学习率$\alpha$太大或代码存在bug 

自动收敛测试：若J($\vec{w}$,b)两次迭代，减少量小于某个特定的值，则判定为收敛

#### 怎样设置学习率

在学习率足够小的情况下，每一次迭代，代价函数都应该减小。所以可以将$\alpha$设为一个很小的数字，看看每次迭代的代价是否会降低，若不降低，代码中有bug  

足够小的$\alpha$仅作调试，实际应用中，若学习率太小，梯度下降需要经过很多次迭代才能收敛  

可以尝试一系列的$\alpha$值，如...0.001  0.01  0.1   1...

### 特征工程及多项式回归

特征工程：选择或输入合适的特征是让算法正常工作的关键步骤，在特征工程中，通常通过变换或合并问题的原始特征，使其帮助算法更简单地做出准确的预测  

可以选择使用不同的特征，通过特征工程和多项式函数为数据搭建一个更好的模型

### 逻辑回归

#### 引例：使用线性回归解决分类问题

$f_{w,b}(x) = wx +b$

设置一个阈值，若$f_{w,b}(x)$ < 0.5，则$\hat{y}$ = 0（负样本），反之$\hat{y}$ = 1（正样本）

这种方式不正确，增加决策样本后，**决策边界**移动，改变之前的正确结论

#### 逻辑回归算法

sigmoid函数：$g(z) = \frac{1}{1+e^{(-z)}}$ （0 < g(z) <1 )

$z = \vec{w}\cdot\vec{x} +b$ ， $g(z) = \frac{1}{1+e^{(-z)}}$，逻辑回归模型：$f_{\vec{w},b}(\vec{x}) = g(\vec{w}\cdot\vec{x} +b)=g(z) = \frac{1}{1+e^{(-(\vec{w}\cdot\vec{x} +b))}}$

#### 决策边界

线性决策边界：$z=\vec{w}\cdot\vec{x} +b=0$

非线性决策边界

例：$f_{\vec{w},b}(\vec{x}) = g(z) = g(w_1x_1^2 + w_2x_2^2 + b)$，若$w_1,w_2,b$为1，1，-1，则$z = x_1^2 + x_2^2 -1 =0$，边界：$x_1^2 + x_2^2 =1$

#### 逻辑回归中的代价函数

平方误差代价函数：$ J(\vec{w},b)=\frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2$

线性回归：代价函数是凸函数。逻辑回归：代价函数是非凸函数，若使用梯度下降算法，存在许多局部极小值

解决
$$
J(\vec{w},b)=\frac{1}{m}\sum_{i=1}^{m}L(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)}))
$$

$$
L(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})=\begin{cases}
-log(f_{\vec{w},b}(\vec{x}^{(i)})),&{y^{(i)}=1}\\
-log(1-f_{\vec{w},b}(\vec{x}^{(i)})),&{y^{(i)}=0}
\end{cases}
$$

#### 简化逻辑回归代价函数

$$
L(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)}))=-y^{(i)}log(f_{\vec{w},b}(\vec{x}^{(i)}))-(1-y^{(i)})log(1-f_{\vec{w},b}(\vec{x}^{(i)}))
$$

#### 过拟合问题

模型不具有泛化到新样本的能力，有时被称为高方差（high variance）

如何解决

- 收集更多数据
- 选择并使用最小特征子集，有时被称为特征选择
- 利用正则化减少参数大小

#### 正则化

尽可能让算法缩小参数的值，参数值越小，模型可能越简单

通常惩罚所有的特征，$\lambda$：正则化参数，$\lambda$>0
$$
J(\vec{x},b)=\frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2+\frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2
$$

只正则化参数$w_j$，而不正则化参数b

### 神经网络

#### 简介

![](http://imgcdn.atyun.com/2019/01/1_ozBVCzy6acVfLuSESiyeBw.jpg)

input layer：$\vec{x}$：特征向量

hidden layer：可以包含多个神经元，$\vec{a}$：activation values 激活值

output layer

**只要输入不同数据，神经网络就会自动学习检测不同的特征**。多层神经网络有时也被称为**多层感知器**

#### 前向传播

将上一层的输出作为下一层的输入，并计算下一层的输出，一直到运算到输出层为止

$a_j^{[l]}=g(\vec{w_j}^{[l]}\cdot\vec{a}^{[l-1]}+b_j^{[l]})$

$g$：sigmoid or activation function

#### 反向传播

”误差反向传播“的简称

该方法对网络中所有权重计算损失函数的梯度，这个梯度会反馈给最优化方法，用来更新权值以最小化损失函数

反向传播仅指用于计算梯度的方法，而非神经网络的整个学习算法

### 激活函数

#### 常见激活函数

Sigmoid

ReLU：线性修正函数 $g(z)=max(0,z)$

线性激活函数：$g(z)=z$

#### 如何选择激活函数

输出层

- 二元分类问题：y=0/1  Sigmoid函数
- 回归问题：y=+/- 线性激活函数 ； y=0 or +：ReLU

隐藏层

- 通常使用ReLU函数(faster)
- 若处理二分类问题，使用Sigmoid函数(slower)

**若对所有节点都使用线性激活函数，神经网络和线性回归无区别。所以不要在神经网络的隐藏层使用线性激活函数**

### 多分类问题

#### Softmax algorithm

$$
z_j=\vec{w_j}\cdot\vec{x}+b_j\\
a_j=\frac{e^{z_j}}{\sum_{k=1}^N{e^{z_k}}}
$$

#### Softmax Regression

if  y=j，$loss=-log\,a_j$ (交叉熵损失函数)

Softmax算法改进：计算方式不同，得到的数值也可能不同，可改变计算方式，减小数值舍入误差

#### 高级优化方法

"Adam" algorithm：自动调整学习率，模型的每个参数使用不同的学习率，比梯度下降更快

若 $w_j$ 或 b 向同一个方向移动，增加$\alpha$；若参数来回振荡，减小$\alpha$

### 全连接层和卷积层

#### 全连接层

input layer

hidden layer

output layer

#### 卷积层

每个神经元关注不同的区域

加快计算速度

需要训练的数据更少，不容易过拟合

![](https://img-blog.csdnimg.cn/20190529231941555.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTQ1MTMyMw==,size_16,color_FFFFFF,t_70)

### 模型评估

训练集：70%，测试集：30%，计算$J_{train},J_{test}$

训练集：60%，交叉验证集：20%，测试集：20%

#### 通过方差、偏差评估

High bias (uderfit)：高偏差

- $J_{train}$ will be high
- $J_{train}≈J_{cv}$

High variance (overfit)：高方差

- $J_{train}$ is low
- J_{cv}>>J_{train}$

高偏差、高方差同时存在：可能一部分过拟合，一部分欠拟合

![](https://img-blog.csdn.net/20180329172617763)

#### 正则化

$$
J(\vec{x},b)=\frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2+\frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2
$$

λ越大，正则化项权重越大，越不重视算法在训练集上的表现

尝试不同的λ值，并在不同的点上评估交叉验证误差

**制定用于性能评估的基准**(1,2用于判断是否存在高偏差，2,3用于判断是否存在高方差)

- Base performance
- Training error
- Cross validation error

#### 学习曲线

高偏差：若存在高偏差，添加更多训练数据无效，需要使用更大的神经网络，如增加额外特征、添加多项式、减小λ

![](http://imgtec.eetrend.com/files/2019-08/%E5%8D%9A%E5%AE%A2/100044650-77501-111.png)

高方差：通过扩大训练集来降低交叉验证误差，也可以减少特征数量、增大λ

![](http://imgtec.eetrend.com/files/2019-08/%E5%8D%9A%E5%AE%A2/100044650-77502-112.png)

#### 误差分析

手动检查一组算法错误，分类或标记样本，然后进行分析

### 数据增强与迁移学习

#### 数据增强

对输入x进行变形或变换，得到另一个有相同标签的示例

技巧：对数据所做的改变应该是测试集中噪声或变形类型的代表

#### 迁移学习

过程

- 下载带有参数的神经网络 
- 根据自己的数据进一步训练或微调网络

类型

- 只训练输出层的参数
- 训练所有的参数

监督预训练：现在数据集上训练，然后在较小的数据集上进行参数调优（微调）

### 机器学习项目流程

#### 流程

scope project —> collect data —> train model —> deploy in production

#### 部署模型常见的方法

通过API调用：将你的机器学习模型部署在服务器上，通过API调用

![](https://pic2.zhimg.com/v2-d377581d876fc350db7f7499adf637b1_r.jpg)



### 精确率与召回率

#### 倾斜数据的误差指标

构造混淆矩阵，计算精确率与召回率

![混淆矩阵](https://ts1.cn.mm.bing.net/th/id/R-C.3fdb7e8b53d26ad83c426dd83ee85676?rik=jAe2AsGa0TgjMg&riu=http%3a%2f%2fblog.hackerearth.com%2fwp-content%2fuploads%2f2017%2f01%2fmyprobbb.jpg&ehk=dxKx7Kd4JX5qdw6tmmXyTunOyWRSVEOmY%2f4t%2bJYLVsM%3d&risl=&pid=ImgRaw&r=0)

- 精确率：分类正确的正样本个数占分类器判定为正样本的样本个数的比例，代表对正样本结果中的预测准确程度

$$
\frac{True\ Positive}{Predicted\ positive}=\frac{True\ positive}{True\ pos+False\ pos}
$$

- 召回率：正确预测为正的占全部实际为正的比例，高的召回率意味着可能会有更多的误检

$$
\frac{True\ Positive}{Actual\ positive}=\frac{True\ positive}{True\ pos+False\ neg}
$$

#### 精确率和召回率的权衡

理想情况：高精度、高召回

实际情况：精确率高，召回率低；召回率高，精确率低

权衡：F-Score
$$
Precision(P),\ Recall(R)\\
F-Score=2\frac{PR}{P+R}
$$

