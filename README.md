# machine-learning

## machine learning algorithms

Supervised learning  

Unsupervised learning

## Supervised learning

x(input) ——> y(output label), learnings from being given 'right answers'   

applications: machine translation, online advertising, self-driving car 

supervised algorithms 

- 回归算法：Regression ——> Predict a number
- 分类算法：Classification ——> Predict categories 

## Unsupervised learning

only x 

Unsupervised algorithms 

- Clustering：聚类算法，将未标记的数据自动分组到集群中 (google news, DNA microarray, grouping customers)
- Anomaly detection：异常检测
- Dimensionality reduction：降维

## Terminology

Training set: data used to train the model 

x = "input" variable / feature 

y = "output" variable 

m = number of training examples 

(x,y) = single training examples 

(x<sup>(i)</sup>, y<sup>(i)</sup>) = i<sup>th</sup> training examples 

## 成本函数：J(w,b) 衡量拟合程度

training set —> learning algorithms —> (x —> f —> $\hat{y}$) 

how to represent f？$f_{w,b} (x) = wx +b$ 单变量线性回归，实现它之前先定义一个成本函数 
$$
\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^(i)-y^(i))^2  或  \frac{1}{zm}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}-y^{(i)})^2
$$

## 梯度下降：找到局部最小值

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

## 综合

使用线性回归模型：$f_{w,b} (x) = wx +b$ 的均方误差代价函数：$\frac{1}{zm}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}-y^{(i)})^2$，并利用梯度下降算法更新  

这种方式称为“批量梯度下降” 

Batch gradient descent: Each step of gradient descent uses all the training examples 

## 多维特征（多元线性回归）

符号：$x_j$ = $j^{th}$ feature, n = number of the features,  $\vec{x}^{(i)}$ = feature of $i^{th}$ training examples 

模型：

- previously: $f_{w,b}(x) = wx+b$
- now: $f_{w,b}(x) = w_1x_1+w_2x_2+...+w_nx_n+b$ 

参数：$\vec{w} = [w_1, w_2, ..., w_n]$, b is a number 

简写：$f_{w,b}(\vec{x})=\vec{w}\cdot\vec{x}+b$

## 矢量化

$f_{w,b}(\vec{x})=\vec{w}\cdot\vec{x}+b$ 矢量化```f = np.dot(w,x) +b``` ，np.dot()实现了$\vec{w}$, $\vec{x}$的点积 

优点：代码简单，仅一行；运行效率高，调用了GPU

## 梯度下降及多元线性回归

参数：$\vec{w} = [w_1, w_2, ..., w_n]$, b is a number  

模型：$f_{w,b}(\vec{x})=\vec{w}\cdot\vec{x}+b$ 

成本函数：J($\vec{w}$,b) 

梯度下降算法

- $w_j = w_j - \alpha\frac{\partial}{\partial w_j}T(\vec{w_j},b)$
- $b = b - \alpha\frac{\partial}{\partial b}T(w,b)$

## 特征缩放

Feature Scaling，让梯度下降进行更快 

当特征的可能值很大时，其对应参数的合理取值很小；当特征的可能值很大时，其对应参数的合理取值比较大 

例：$300\leq{x_1}\leq2000$, $0\leq{x_2}\leq5$

- 特征缩放：$x_1.scaled = \frac{x_1}{2000}$, $x_2.scaled = \frac{x_2}{5}$
- 均值归一化：$x_1= \frac{x_1-\mu_1}{2000-300}$, $x_2= \frac{x_2-\mu_2}{5-0}$, $\mu_1$,$\mu_2$为均值
- Z-score标准化：$x_1=\frac{x_1-\mu_1}{\sigma_1}$, $x_2=\frac{x_2-\mu_2}{\sigma_2}$, $\sigma_1$,$\sigma_2$为标准差

当特征的可能值很大或很小时都需要进行特征缩放

## 如何判断梯度下降是否收敛

画学习曲线：每次迭代J($\vec{w}$,b)都会减少，若迭代后增大，意味着学习率$\alpha$太大或代码存在bug 

自动收敛测试：若J($\vec{w}$,b)两次迭代，减少量小于某个特定的值，则判定为收敛

