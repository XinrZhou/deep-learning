# machine-learning

**说明**：涉及数学公式，可安装**MathJax Plugin for Github**插件获得更好阅读体验

### machine learning algorithms

监督学习：Supervised learning  

无监督学习：Unsupervised learning

### Supervised learning

x(input) ——> y(output label), learnings from being given 'right answers'   

applications: machine translation, online advertising, self-driving car 

supervised algorithms 

- 回归算法：Regression ——> Predict a number
- 分类算法：Classification ——> Predict categories 

### Unsupervised learning

only x 

Unsupervised algorithms 

- Clustering：聚类算法，将未标记的数据自动分组到集群中 (google news, DNA microarray, grouping customers)
- Anomaly detection：异常检测
- Dimensionality reduction：降维

### Terminology

Training set: data used to train the model 

x = "input" variable / feature 

y = "output" variable 

m = number of training examples 

(x,y) = single training examples 

(x<sup>(i)</sup>, y<sup>(i)</sup>) = i<sup>th</sup> training examples 

### 成本函数：J(w,b) 衡量拟合程度

training set —> learning algorithms —> (x —> f —> $\hat{y}$) 

how to represent f？$f_{w,b} (x) = wx +b$ 单变量线性回归，实现它之前先定义一个成本函数 
$$
\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^(i)-y^(i))^2  或  \frac{1}{2m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}-y^{(i)})^2
$$

### 梯度下降：找到局部最小值

梯度下降算法：

- $w = w - \alpha\frac{\partial}{\partial w}T(w,b)$
- $b = b - \alpha\frac{\partial}{\partial b}T(w,b)$

重复执行这两个步骤，直到算法收敛（达到局部最小值）。$\alpha$学习率，控制更新w,b的步长 

正确做法：w, b同步更新。如何实现？ 

1. $tempW = w - \alpha\frac{\partial}{\partial w}T(w,b)$
2. $tempB = b - \alpha\frac{\partial}{\partial b}T(w,b)$
3. $w = tempW$
4. $b = tempB$

$\alpha$小：梯度下降慢，$\alpha$大：可能永远无法收敛

应用：使用线性回归模型：$f_{w,b} (x) = wx +b$ 的均方误差代价函数：$\frac{1}{zm}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}-y^{(i)})^2$，并利用梯度下降算法更新  

- 这种方式称为“批量梯度下降” 
- Batch gradient descent: Each step of gradient descent uses all the training examples 

### 多维特征（多元线性回归）

符号：$x_j$ = $j^{th}$ feature, n = number of the features,  $\vec{x}^{(i)}$ = feature of $i^{th}$ training examples 

模型：

- previously: $f_{w,b}(x) = wx+b$
- now: $f_{w,b}(x) = w_1x_1+w_2x_2+...+w_nx_n+b$ 

参数：$\vec{w} = [w_1, w_2, ..., w_n]$, b is a number 

简写：$f_{w,b}(\vec{x})=\vec{w}\cdot\vec{x}+b$

### 矢量化

$f_{w,b}(\vec{x})=\vec{w}\cdot\vec{x}+b$ 矢量化```f = np.dot(w,x) +b``` ，np.dot()实现了$\vec{w}$, $\vec{x}$的点积 

优点：代码简单，仅一行；运行效率高，调用了GPU

### 梯度下降及多元线性回归

参数：$\vec{w} = [w_1, w_2, ..., w_n]$, b is a number  

模型：$f_{w,b}(\vec{x})=\vec{w}\cdot\vec{x}+b$ 

成本函数：J($\vec{w}$,b) 

梯度下降算法

- $w_j = w_j - \alpha\frac{\partial}{\partial w_j}T(\vec{w_j},b)$
- $b = b - \alpha\frac{\partial}{\partial b}T(w,b)$

### 特征缩放

Feature Scaling，让梯度下降进行更快 

当特征的可能值很大时，其对应参数的合理取值很小；当特征的可能值很大时，其对应参数的合理取值比较大 

例：$300\leq{x_1}\leq2000$, $0\leq{x_2}\leq5$

- 特征缩放：$x_1.scaled = \frac{x_1}{2000}$, $x_2.scaled = \frac{x_2}{5}$
- 均值归一化：$x_1= \frac{x_1-\mu_1}{2000-300}$, $x_2= \frac{x_2-\mu_2}{5-0}$, $\mu_1$,$\mu_2$为均值
- Z-score标准化：$x_1=\frac{x_1-\mu_1}{\sigma_1}$, $x_2=\frac{x_2-\mu_2}{\sigma_2}$, $\sigma_1$,$\sigma_2$为标准差

当特征的可能值很大或很小时都需要进行特征缩放

### 两个技巧

如何判断梯度下降是否收敛

- 画学习曲线：每次迭代J($\vec{w}$,b)都会减少，若迭代后增大，意味着学习率$\alpha$太大或代码存在bug 
- 自动收敛测试：若J($\vec{w}$,b)两次迭代，减少量小于某个特定的值，则判定为收敛

怎样设置学习率

- 在学习率足够小的情况下，每一次迭代，代价函数都应该减小。所以可以将$\alpha$设为一个很小的数字，看看每次迭代的代价是否会降低，若不降低，代码中有bug  

- 足够小的$\alpha$仅作调试，实际应用中，若学习率太小，梯度下降需要经过很多次迭代才能收敛  

- 可以尝试一系列的$\alpha$值，如...0.001  0.01  0.1   1...

### 特征工程以及多项式回归

特征工程：选择或输入合适的特征是让算法正常工作的关键步骤，在特征工程中，通常通过变换或合并问题的原始特征，使其帮助算法更简单地做出准确的预测  

可以选择使用不同的特征，通过特征工程和多项式函数为数据搭建一个更好的模型

### 逻辑回归

引例：使用线性回归解决分类问题

- $f_{w,b}(x) = wx +b$
- 设置一个阈值，若$f_{w,b}(x)$ < 0.5，则$\hat{y}$ = 0（负样本），反之$\hat{y}$ = 1（正样本）
- 这种方式不正确，增加决策样本后，**决策边界**移动，改变之前的正确结论

逻辑回归算法：classification / logistic regression

- sigmoid函数：$g(z) = \frac{1}{1+e^{(-z)}}$ （0 < g(z) <1 )
- $z = \vec{w}\cdot\vec{x} +b$ ， $g(z) = \frac{1}{1+e^{(-z)}}$，逻辑回归模型：$f_{\vec{w},b}(\vec{x}) = g(\vec{w}\cdot\vec{x} +b)=g(z) = \frac{1}{1+e^{(-(\vec{w}\cdot\vec{x} +b))}}$

决策边界：decision boundary

- 线性决策边界：$z=\vec{w}\cdot\vec{x} +b=0$
- 非线性决策边界
  - 例：$f_{\vec{w},b}(\vec{x}) = g(z) = g(w_1x_1^2 + w_2x_2^2 + b)$，若$w_1,w_2,b$为1，1，-1，则$z = x_1^2 + x_2^2 -1 =0$，边界：$x_1^2 + x_2^2 =1$
