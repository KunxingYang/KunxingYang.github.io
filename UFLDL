# UFLDL 笔记

标签： UFLDL 稀疏自编码器 神经网络

---

## 2.稀疏自编码器
### 2.1 神经网络模型
神经网络能够提供一种复杂且非线性的假设模型$h_{\omega,b}(x)$, 它具有参数$\omega, b$, 可以以此参数来拟合我们的数据。
神经元是组成神经网络的基本单位，其输入为$x=[x_1,x_2,\cdots,x_n]$以及截距$b$, 输出为$$h_{\omega,b}(x)=f(\sum^n_{i=1}\omega_i x_i+b) \tag 1$$
其中， $f:R \mapsto R$为激活函数，教程中使用Sigmoid函数作为激活函数$f(\cdot)$:
$$
f(z)=\frac{1}{1+e^{-z}} \tag 2
$$
Sigmoid函数的一个重要的性质是，可以将自变量z转换到$[0,1]$范围内，同时其导数值为$f^{\prime}(z)=f(z)(1-f(z))$

所谓**神经网络**就是将许多个单一神经元联结在一起，这样一个神经元的输出成了另一个神经元的输入。
![NN](http://deeplearning.stanford.edu/wiki/images/thumb/3/3d/SingleNeuron.png/300px-SingleNeuron.png)
其中，$+1$圆圈表示**偏置节点**，也是截距项。最左边是输入层，最右边叫输出层，中间叫做隐层。
用$n_l$表示网络的层数，第$l$层记为$L_l$。上图中神经网络有参数$(\omega,b)=(\omega^{(1)},b^{(1)},\omega^{(2)},b^{(2)})$,其中$\omega_{ij}^{(l)}$是第$l$层第$j$个单元与第$l+1$层第$i$个单元之间的联结参数，$b_i^{(l)}$是第$l+1$层第$i$个单元的偏置项，用$s_l$表示第$l$层的节点数。
用$a_i^{(l)}$表示第$l$层第$i$个节点的激活值，对于给定的参数集合$\omega,b$，我们的神经网络就可以按照函数$h_{\omega,b}(x)$来计算输出结果.
$$
\begin{align}
a_1^{(2)}=f(\omega_{11}^{(1)}x_1+\omega_{12}^{(1)}x_2+\omega_{13}^{(1)}x_3+b_1^{(1)}) \tag 3 \\
a_2^{(2)}=f(\omega_{21}^{(1)}x_1+\omega_{22}^{(1)}x_2+\omega_{23}^{(1)}x_3+b_2^{(1)}) \tag 4 \\
a_3^{(2)}=f(\omega_{31}^{(1)}x_1+\omega_{32}^{(1)}x_2+\omega_{33}^{(1)}x_3+b_3^{(1)}) \tag 5 \\
h_{\omega,b}(x)=a_1^{(3)}=f(\omega_{11}^{(2)}a_1+\omega_{12}^{(2)}a_2+\omega_{13}^{(2)}a_3+b_1^{(2)}) \tag 6 \\
\end{align}
$$
用$z_i^{(l)}$表示第$l$层第$i$单元输入加权和，则$a_i^{(l)}=f(z_i^{(l)})$,上述计算公式可以写为：
$$
\begin{align}
z^{(2)}=\omega^{(1)}x+b^{(1)} \tag 7 \\
a^{(2)}=f(z^{(2)}) \tag 8 \\
z^{(3)}=\omega^{(2)}x+b^{(2)} \tag 9 \\
h_{\omega,b}(x)=a^{(3)}=f(z^{(3)}) \tag {10} \\
\end{align}
$$
这个计算步骤叫做前向传播，通用计算公式为
$$
\begin{align}
z^{(l+1)}=\omega^{(l)}x+b^{(l)} \tag {11} \\
a^{(l+1)}=f(z^{(l+1)}) \tag {12} \\
\end{align}
$$
根据以上模型，可以构建有多隐层的神经网络，对更复杂的数据进行拟合。
### 2.2 反向传播算法
假设有一个固定样本集$\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\cdots,(x^{(m)},y^{(m)}) \}$,包含$m$个样例，可以使用批量梯度下降来求解神经网络。
对于单个样例，代价函数(cost function)为：
$$
J(\omega,b;x,y)=\frac{1}{2}||h_{\omega,b}(x)-y||^2 \tag {13}
$$
给定一个包含有m个样本的数据集，定义其整体代价函数为:
$$
\begin{align}
J(\omega,b) &= [\frac{1}{2}\sum^m_{i=1}J(\omega,b;x^{(i)},y^{(i)})]+\frac{\lambda}{2}\sum^{n_l-1}_{l=1}\sum^{s_l}_{i=1}\sum^{s_l+1}_{j=1}(\omega_{ji}^{(l)})^2 \tag {14} \\
&= [\frac{1}{2}\sum^m_{i=1}(\frac{1}{2}||h_{\omega,b}(x^{(i)})-y^{(i)}||^2)]+\frac{\lambda}{2}\sum^{n_l-1}_{l=1}\sum^{s_l}_{i=1}\sum^{s_l+1}_{j=1}(\omega_{ji}^{(l)})^2 \tag {15}
\end{align}
$$
第一项$J(\omega,b)$是一个均方差项，第二项是一个规则化项，目的在于减少权重的幅度，防止过拟合。
$J(\omega,b;x,y)$是针对样本$(x,y)$的代价函数，$J(\omega,b)$是整体的代价函数。
我们优化的目标是调整参数$\omega,b$使得代价函数$J(\omega,b)$取得最小值。初始化时将$\omega,b$随机初始化为一个很小的，接近0的值。因为$J(\omega,b)$是一个非凸函数，所以使用梯度下降是可能会收敛到局部最优解，但是大部分情况下使用梯度下降算法都可以得到令人满意的结果。
梯度下降发中每一次迭代都按照如下公式对参数$\omega,b$沿着负梯度方向进行更新：
$$
\begin{align}
\omega_{ij}^{(l)} := \omega_{ij}^{(l)}-\alpha\frac{\partial}{\partial\omega_{ij}^{(l)}}J(\omega,b) \tag {16} \\
b_i^{(l)} := b_i^{(l)}-\alpha\frac{\partial}{\partial b_i^{(l)}}J(\omega,b) \tag {17}
\end{align}
$$
其中$\alpha$是学习速率。使用梯度下降进行权值更新时，关键步骤是计算偏导数，对于神经网络中的偏导数计算，**反向传播算法**是计算偏导数的一种有效方法。
求式$(14)$对参数$\omega,b$的偏导数有：
$$
\begin{align}
\frac{\partial}{\partial\omega_{ij}^{(l)}}J(\omega,b)=[\frac{1}{m}\sum^m_{i=1}\frac{\partial}{\partial\omega_{ij}^{(l)}}J(\omega,b;x^{(i)},y^{(i)})]+\lambda\omega_{ij}^{(l)} \tag {18} \\
\frac{\partial}{\partial b_i^{(l)}}J(\omega,b)=\frac{1}{m}\sum^m_{i=1}\frac{\partial}{\partial b_i^{(l)}}J(\omega,b;x^{(i)},y^{(i)}) \tag {19}
\end{align}
$$
反向传播算法细节：
1.进行前向传播计算，得到从第二层开始的以后各层的激活值。
2.对于第$n_l$层(输出层)的每一个输出单元$i$，根据下面公式计算残差：
$$
\delta_i^{(n_l)}=\frac{\partial}{\partial z_i^{(n_l)}}\frac{1}{2}||h_{\omega,b}(x)-y||^2=-(y_i-a_i^{(n_l)}) \cdot f^{\prime}(z_i^{(n_l)}) \tag {20}
$$
注：向量写法：
$$
\delta^{(n_l)}=-(y-a^{(n_l)}) \cdot f^{\prime}(z^{(n_l)}) \tag {21}
$$
3.对$l=n_l-1,n_l-2,n_l-3,\cdots,2$各层，计算第$l$层的第$i$个节点的残差：
$$
\delta_i^{(l)}=(\sum^{s_{l+1}}_{j=1}\omega_{ij}^{(l)}\delta_i^{(l+1)})f^{\prime}(z_i^{(l)}) \tag {22}
$$
注：向量写法：
$$
\delta_i^{(l)}=((\omega^{(l)})^T\delta^{(l+1)})f^{\prime}(z^{(l)}) \tag {23}
$$
4.计算偏导数：
$$
\begin{align}
\frac{\partial}{\partial\omega_{ij}^{(l)}}J(\omega,b;x,y)=a_j^{l}\delta_i^{(l+1)} \tag {24} \\
\frac{\partial}{\partial b_i^{(l)}}J(\omega,b;x,y)=\delta_i^{(l+1)} \tag {25}
\end{align}
$$
向量写法：
$$
\begin{align}
\nabla_{\omega^{(l)}}J(\omega,b;x,y)=\delta^{(l+1)}(a^{l})^T \tag {26} \\
\nabla_{b^{(l)}}J(\omega,b;x,y)=\delta^{(l+1)} \tag {27}
\end{align}
$$

使用上面计算出来的偏导数进行一次梯度下降的迭代：
1.对于所有$l$，令$\Delta_{\omega^{(l)}}:=0,\Delta_{b^{(l)}}:=0$(设置为全零矩阵或全零向量)。
2.对于$i=1$到m,
(a) 使用反向传播算法计算$\nabla_{\omega^{(l)}}J(\omega,b;x,y),\nabla_{b^{(l)}}J(\omega,b;x,y)$.
(b) 计算$\Delta_{\omega^{(l)}}=\Delta_{\omega^{(l)}}+\nabla_{\omega^{(l)}}J(\omega,b;x,y)$；
(c) 计算计算$\Delta_{b^{(l)}}=\Delta_{b^{(l)}}+\nabla_{b^{(l)}}J(\omega,b;x,y)$
3.更新权重：
$$
\begin{align}
\omega^{(l)}=\omega^{(l)}-\alpha[(\frac{1}{m}\Delta_{\omega^{(l)}})+\lambda\omega^{(l)}] \tag {28} \\
b^{(l)}=b^{(l)}-\alpha[(\frac{1}{m}\Delta_{b^{(l)}}) \tag {29}
\end{align}
$$
这样就可以重复梯度下降法的迭代步骤来最优化代价函数，进而求解神经网络权重值。

### 2.4 梯度检验和高级优化
略
### 2.5 自编码算法与稀疏性
假设有一个没有带类别标签的训练样本集合$\{x^{(1)},x^{(2)},x^{(3)},\cdots\}$,其中$x^{(i)} \in R^n$,自编码神经网络是一种无监督学习算法，使用了反向传播算法，让目标值等于输入值，比如$y^{(i)}=x^{(i)}$。
自编码神经网络尝试学习一个$h_{\omega,b}(x) \approx x$函数，它尝试逼近一个恒等函数，从而使得输出$\hat{x}$接近与输入$x$，当为自编码神经网络加入某些限制，就可以从输入数据中发现一些特征。如果数据没有特征，是随机的数据，比如独立同分布的高斯随机变量，模型将会非常难学习。如果隐层数量比输入层神经元数量少的时候，可以实现数据的压缩；即使隐层神经元数量较大，可以通过给自编码神经网络加入稀疏性限制来发现数据集的结构。
稀疏性的解释，如果神经元的输出接近于1，那么神经元被激活，否则被抑制，那么神经元大部分的时间都是被抑制的限制被称为稀疏性限制。使用$a_j^{(2)}(x)$表示在给定输入为$x$情况下，自编码神经网络隐层神经元$j$的激活度。所以定义，隐层神经元$j$的平均活跃度(在训练集上取平均)$$\hat{\rho}=\frac{1}{m}\sum^m_{i=1}[a_j^{(2)}(x^{(i)})] \tag {30}$$
稀疏性限制可以表示为$\hat{\rho}_j=\rho$,$\rho$是稀疏性参数，通常是一个接近于0的较小的值，即要让隐层神经元$j$的平均活跃度比较小，接近于0。
为了实现这一限制，将在我们的优化目标函数中加入一个额外的**惩罚因子**，而这一惩罚因子将惩罚那些$\hat{\rho}_j$和$\rho$有显著不同的情况从而使得隐层神经元的平均活跃度保持在较小的范围内。惩罚因子选择KL相对熵来实现
$$
\sum^{s_2}_{j=1}\rho log\frac{\rho}{\hat{\rho}_j}+(1-\rho) log\frac{1-\rho}{1-\hat{\rho}_j}=\sum^{s_2}_{j=1}KL(\rho||\hat{\rho}_j) \tag {31}
$$
$s_2$是隐层中隐藏神经元的数量，而索引$j$依次代表隐层的每一个神经元,上式是一个以$\rho$为均值和一个以$\hat{\rho}_j$为均值的两个伯努利随机变量之间的相对熵，相对熵是一种标准的用来衡量两个分布之间的差异的方法，二者相等时等于0，相差很大时熵会变得很大。所以最小化这个惩罚因子，使得$\hat{\rho}_j$趋近于$\rho$。这样总体代价函数可以表示为：
$$
J_{sparse}(\omega,b)=J(\omega,b)+\beta\sum^{s_2}_{j=1}KL(\rho||\hat{\rho}_j) \tag {32}
$$
对上面这个代价函数进行求偏导时，使用反向传播算法计算的残差可以变为
$$
\delta_i^{(l)}=((
\sum_{i=1}^{s_l}\omega_{ji}^{(l)}\delta_j^{(l+1)})+\beta(-\frac{\rho}{\hat{\rho}_i}+\frac{1-\rho}{1-\hat{\rho}_i}))f^{\prime}(z_i^{(l)}) \tag {33}
$$

## 3 矢量化编程
略

## 4 预处理：主成分分析与白化
### 4.1 主成分分析
PCA是一种可以极大提升无监督特征学习速度的数据降维算法，PCA可以将输入向量转换为一个维数低很多的近似向量，而且误差很小。
假设有数据集$x=\{x^{(1)},x^{(2)},\cdots,x^{(n)}\}$,对数据进行预处理，使得数据集的每个维度(即特征)具有相同的均值(均值皆等于0)和方差。PCA将寻找一个低维空间来投影我们的数据。
首先计算数据集的协方差矩阵
$$
\Sigma=\frac{1}{m}\sum_{i=1}^m(x^{(i)})(x^{(i)})^T \tag {34}
$$
求解协方差矩阵的特征值和特征向量，对应特征值最大的特征向量即是主特征向量，即数据的主成分方向。将协方差矩阵的特征向量按照其对应的特征值大小按列以此排列组成矩阵$U$:
$$
U=
\begin{bmatrix}
| & | &   & | \\
u_1 & u_2 & \cdots & u_n\\
| & | &   & | \\
\end{bmatrix}
\tag {35}
$$
向量$u_1,u_2,\cdots,u_n$构成了数据的新基，可以将数据投影在其上来表示数据，令$x \in R^2$为训练样本，那么$u_i^Tx$就是样本点在基向量$u_i^T$上的投影长度(幅值)。

所以，将原始数据集$x$用基向量$u_1,u_2,\cdots,u_n$来表示
$$
x_{rot}=U^Tx=
\begin{bmatrix}
u_1^Tx \\
u_2^Tx \\
\vdots \\
u_n^Tx
\end{bmatrix} \tag {36}
$$
表示，将原始数据集旋转到以特征向量为基的坐标系里。矩阵$U$有正交性，即满足$U^TU=UU^T=I$,若想将旋转后的向量$x_{rot}$还原成原始数据$x$，将其左乘矩阵$U$即可
$$
x=Ux_{rot}
$$



## 5 Softmax回归
### 5.1 Softmax回归
Softmax回归模型是logistic回归模型在多分类问题上推广。
回顾logistic回归，有训练集$\{(x^{(1)},y^{(1)}),\cdots,(x^{(m)},y^{(m)})\}$,其中输入特征$x^{(i)} \in R^{n+1}$,类别$y^{(i)} \in \{0,1\}$,其假设函数(hypothesis function)
$$
h_{\theta}(x)=\frac{1}{1+exp(-\theta^Tx)}  \tag {}
$$
代价函数(cost function):
$$
J(\theta)=-\frac{1}{m}[\sum^m_{i=1}y^{(i)}logh_{\theta}(x^{(i)})+(1-y^{(i)})log(1-h_{\theta}(x^{(i)}))]    \tag {}
$$
对于Softmax回归，要解决的是多分类问题，对于数据集$\{(x^{(1)},y^{(1)}),\cdots,(x^{(m)},y^{(m)})\}$，有$y^{(i)} \in \{1,2，\cdots,k\}$
对于给定的测试输入$x$,使用假设函数针对每一个类别$j$估算出概率值$p(y=j|x)$,就是估计$x$的每一种分类结果出现的概率，因此假设函数输出一个$k$维的向量来表示这$k$个估计的概率值。
首先，Softmax函数为：
$$
\delta(z)_j=\frac{e^{z_j}}{\sum^K_{k=1}e^{z_k}} \text{  ,for 1,...,K} \tag {}
$$
对于分类结果$j$，计算其出现的概率为：
$$
P(y=j|x)=\frac{e^{\theta_j^Tx}}{\sum^K_{k=1}e^{\theta_k^Tx}}  \tag {}
$$
所以Softmax回归的假设函数$h_{\theta}(x)$为：
$$
h_{\theta}(x^{(i)})=
\begin{bmatrix}
p(y^{(i)}=1|x^{(i)};\theta) \\
p(y^{(i)}=2|x^{(i)};\theta) \\
\vdots \\
p(y^{(i)}=k|x^{(i)};\theta) 
\end{bmatrix}=
\frac{1}{\sum^K_{j=1}e^{\theta_{j}^Tx^{(i)}}}
\begin{bmatrix}
e^{\theta_{1}^Tx^{(i)}} \\
e^{\theta_{2}^Tx^{(i)}} \\
\vdots \\
e^{\theta_{k}^Tx^{(i)}} 
\end{bmatrix}
\tag {}
$$
其中$\theta_1,\theta_2,\cdots,\theta_k \in R^{n+1}$是模型的参数，上式中，$\frac{1}{\sum^K_{j=1}e^{\theta_{j}^Tx^{(i)}}}$是对概率分布进行归一化，使得所有概率之和为1.
