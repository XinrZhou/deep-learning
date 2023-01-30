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

