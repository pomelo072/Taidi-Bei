# Taidi-Bei

# 算法

## 1，决策树

**注意**：决策树最后只有两种结果；也许可以用lasso回归做预剪枝。

```python
#基本调用的库

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
import graphviz

#基本参数说明
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)[source]
```

- criterion：特征选择标准，可选参数，**默认是gini（基尼指数）**，可以设置为entropy。


- splitter：特征划分点选择标准，可选参数，默认是best(数据量少)，**可以设置为random(数据量大)**。每个结点的选择策略。best参数是根据算法选择最佳的切分特征，例如gini、entropy。random随机的在部分划分点中找局部最优的划分点。


- max_features：(**这里讲了一大堆，默认就行**)划分时考虑的最大特征数，可选参数，默认是None。寻找最佳切分时考虑的最大特征数(n_features为总共的特征数)，有如下6种情况：
    - 整型的数，则考虑max_features个特征；
    - 浮点型的数，则考虑int(max_features * n_features)个特征
    - 设为auto，那么max_features = sqrt(n_features)；
    - 设为sqrt，那么max_featrues = sqrt(n_features)，跟auto一样；
    - 设为log2，那么max_features = log2(n_features)；
    - **None，那么max_features = n_features，也就是所有特征都用。**
        一般来说，如果样本特征数不多，比如小于50，我们用默认的”None”就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。

- max_depth：(**可以先看看树的深度，再决定**)决策树最大深，可选参数，默认是None。推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。
- min_samples_split：内部节点再划分所需最小样本数，可选参数，默认是2。这个值限制了子树继续划分的条件。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
- min_weight_fraction_leaf**(不管)**：叶子节点最小的样本权重和，可选参数，默认是0。这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。
- max_leaf_nodes：**(不管，剪枝再说)**最大叶子节点数，可选参数，默认是None。通过限制最大叶子节点数，可以防止过拟合。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。
- class_weight**(暂时不管)**：类别权重，可选参数，默认是None，
- random_state：可选参数，默认是None。随机数种子。（李子夏知道，我看不懂）
- min_impurity_split：节点划分最小不纯度,可选参数，默认是1e-7。这是个阈值，这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。
- presort：**(不重要)数据是否预排序，可选参数，默认为False。

1. 采用CART树，以基尼系数为评判准测。

2. 剪枝，就是防止过拟合，有预剪枝和后剪枝两种，我们决定使用后剪枝：

    后剪枝是一种惩罚函数，可以将复杂度降低，过拟合现象减小。

    ![Decision_tree_Post-Pruning](https://img-blog.csdn.net/20180724163032640?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FtMjkwMzMzNTY2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    

## 2，带有lasso回归的逻辑回归函数

**注意**：逻辑回归只有两种结果

1. lasso回归参考代码：

    ```python
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Lasso
    
    inputFile = 'data.csv'            # 输入的数据文件
    outputFile = 'new_reg_data.csv'   # 输出的数据文件
    data = pd.read_csv(inputFile)     # 读取数据
    pearson = np.round(data.corr(method='pearson'), 2)   # 保留两位小数
    print('相关系数矩阵为：', pearson)
    
    lasso = Lasso(1000)               # 调用Lasso()函数，设置λ的值为1000
    lasso.fit(data.iloc[:, 0:13], data['y'])
    print('相关系数为：', np.round(lasso.coef_, 5))           # 输出结果，保留五位小数
    print('相关系数非零个数为：', np.sum(lasso.coef_ != 0))   # 计算相关系数非零的个数
    
    mask = lasso.coef_ != 0                 # 返回一个相关系数是否为零的布尔数组
    new_reg_data = data.iloc[:, mask]       # 返回相关系数非零的数据
    new_reg_data.to_csv(outputFile, index=None)         # 存储数据
    print('输出数据的维度为：', new_reg_data.shape)      # 查看输出数据的维度
    
    ```

    **注意**：我们可以先进行lasso回归确定因子数据

2. logistic回归：Sigmoid函数（核心）

3. 代码参考：

    ```python
    import pandas as pd
    from sklearn.linear_model import LogisticRegression #logistic回归
    from sklearn.model_selection import train_test_split #训练测试集
    from sklearn.metrics import classification_report  #数据分析报告
    
    data = pd.read_csv('LogisticRegression.csv')
    data_tr, data_te, label_tr, label_te = train_test_split(data.iloc[:, 1:], data['admit'], test_size=0.2)
    clf = LogisticRegression()
    clf.fit(data_tr, label_tr)
    pre = clf.predict(data_te)
    res = classification_report(label_te, pre)
    print(res)
    
    ```

    

4. 不知道为什么我看到的数据都是自己定义函数，没有直接用，我感觉直接用省事多了。官方有参数说明：*http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html*

  ​    

## 3，K—近邻算法

- 代码参考

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()  # 鸢尾花数据
data_tr, data_te, label_tr, label_te = train_test_split(iris.data, iris.target, test_size=0.2)   # 拆分专家样本集

model = KNeighborsClassifier(n_neighbors=5)   # 构建模型
model.fit(data_tr, label_tr)   # 模型训练
pre = model.predict(data_te)   # 模型预测
acc = model.score(data_te, label_te)   # 模型在测试集上的精度
acc

```

- ```python
    class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, 
    											algorithm=’auto’, leaf_size=30, 
    											p=2, metric=’minkowski’, 
    											metric_params=None, 
    											n_jobs=None, **kwargs)
    
    ```

    参数：

    - n_neighbors ： int，optional(default = 5)默认情况下kneighbors查询使用的邻居数。就是k-NN的k的值，选取最近的k个点。
    - weights ： str或callable，可选(默认=‘uniform’)
        默认是uniform，参数可以是uniform、distance，也可以是用户自己定义的函数。uniform是均等的权重，就说所有的邻近点的权重都是相等的。distance是不均等的权重，距离近的点比距离远的点的影响大。用户自定义的函数，接收距离的数组，返回一组维数相同的权重。
    - algorithm ： {‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}，可选
        快速k近邻搜索算法，默认参数为auto，可以理解为算法自己决定合适的搜索算法。除此之外，用户也可以自己指定搜索算法ball_tree、kd_tree、brute方法进行搜索，brute是蛮力搜索，也就是线性扫描，当训练集很大时，计算非常耗时。kd_tree，构造kd树存储数据以便对其进行快速检索的树形数据结构，kd树也就是数据结构中的二叉树。以中值切分构造的树，每个结点是一个超矩形，在维数小于20时效率高。ball tree是为了克服kd树高纬失效而发明的，其构造过程是以质心C和半径r分割样本空间，每个节点是一个超球体。
    - leaf_size ： int，optional(默认值= 30)
        默认是30，这个是构造的kd树和ball树的大小。这个值的设置会影响树构建的速度和搜索速度，同样也影响着存储树所需的内存大小。需要根据问题的性质选择最优的大小。
    - p ： **(可以调整试试看，用3)**  整数，可选(默认= 2)
        距离度量公式。在上小结，我们使用欧氏距离公式进行距离度量。除此之外，还有其他的度量方法。
    - metric ： 字符串或可调用，默认为’minkowski’
        用于距离度量，默认度量是minkowski，也就是p=2的欧氏距离(欧几里德度量)。
    - metric_params ： dict，optional(默认=None)
        距离公式的其他关键参数，这个可以不管，使用默认的None即可。
    - n_jobs ： int或None，可选(默认=None)

# 处理方法

1. 硬投票，三个模型，少数服从多数

2. ```python
    #简化版参考代码
    
    from sklearn.linear_model import LogisticRegression  	 
    #logistics回归
    from sklearn.svm import SVC   #支持向量机
    from sklearn.tree import DecisionTreeClassifier  #决策树
    from sklearn.ensemble import VotingClassifier  #投票
    
    # 实例化
    voting_clf = VotingClassifier(estimators=[
        ('log_clf', LogisticRegression()),
        ('svm_clf', SVC()),
        ('dt_clf', DecisionTreeClassifier(random_state=666))
    ], voting='hard')
    
    voting_clf.fit(X_train, y_train)
    voting_clf.score(X_test, y_test)
    ```

    

# 因子池

| 因子类型 | 名称·                                     | 说明                                            | 变量说明      |
| -------- | ----------------------------------------- | ----------------------------------------------- | ------------- |
| 因变量   | 是否高送转                                | 送转比例[0,1],>0.5为高送转                      | Is_GSZ        |
| 自变量   | 每股资本公积(元/股)+每股未分配利润(元/股) | 年数据                                          | p_ZBGJ_WFLR   |
|          | 每股现金流量净额(元/股)                   | 年数据                                          | p_XJLL        |
|          | 每股收益(期末摊薄，元/股)                 | 年数据                                          | p_TTM         |
|          | 每股净资产(元/股)                         | 年数据                                          | p_XZC         |
|          | 实收资本(或股本)                          | 年数据                                          | total_GB      |
|          | 营业总收入同必增长(%)                     | 年数据                                          | rate_SRZZ     |
|          | 交易日平均价                              | 以1月1日为标准前20日的收盘价平均值为标准,日数据 | average_price |
|          | 上市年限                                  | 上市时间（year）                                | year_listed   |
|          | 次新股                                    | 是为1，否为0                                    | snew_share    |
|          | 预增或预减                                | 预增为1，预减为-1，其他为0                      | YZ_YJ         |
|          | 超涨或超跌                                | 超涨为1，超跌为-1，其他为0                      | CZ_CD         |
|          | 近两年送转比率                            | 前两年送转比率平均值                            | transfrom     |
|          |                                           |                                                 |               |
|          |                                           |                                                 |               |
|          |                                           |                                                 |               |
