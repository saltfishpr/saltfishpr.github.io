---
title: 数据挖掘入门与实践
date: 2020-03-09T00:00:00+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
description: 《Python 数据挖掘入门与实践》学习笔记
tags: ["python", "data science"]
categories: ["Programming"]
---

<!--more-->

书上的源码在[官网](http://www.packtpub.com/support)上可以注册账号下载，这里只为记录自己的学习过程。

如果有侵权情况，请给我发邮件通知我删除 526191197@qq.com

此笔记的代码均在 `pycharm - python3.8` 中运行通过

学习数据挖掘，让数据服务于人类

## 第一章

### 亲和性分析

亲和性分析根据样本个体（物体）之间的相似度，确定他们的关系亲疏。应用场景有以下几个方面：

- 向用户投放定向广告
- 为用户提供推荐（如歌曲推荐，电影推荐等）

名词：

- 规则：一条规则由前提条件和结论两部分组成
- 支持度：<u>数据集</u>中规则应验的次数
- 置信度：规则（结果）出现的次数 / 条件出现的次数（条件相同的规则数量），衡量规则的准确率

```python
# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from operator import itemgetter

if __name__ == '__main__':
    dataset_filename = "affinity_dataset.txt"
    X = np.loadtxt(dataset_filename)
    n_samples, n_features = X.shape  # 样本数，特征数
    features = ["bread", "milk", "cheese", "apples", "bananas"]  # 商品名列表

    # 如果 xxx，那么 xxx 就是一条规则。规则由前提条件和结论两部分组成
    # 这里注意'如果买 A 则他们会买 B'和'如果买 B 则他们会买 A'不是一个规则，在下面的循环中体现出来
    valid_rules = defaultdict(int)  # 规则应验
    invalid_rules = defaultdict(int)  # 规则无效
    num_occurences = defaultdict(int)  # 商品购买数量字典

    for sample in X:  # 对数据集里的每个消费者
        for premise in range(n_features):
            if sample[premise] == 0:  # 如果这个商品没有买，继续看下一个商品
                continue
            num_occurences[premise] += 1  # 记录这个商品购买数量
            for conclusion in range(n_features):
                if premise == conclusion:  # 跳过此商品
                    continue
                if sample[conclusion] == 1:
                    valid_rules[(premise, conclusion)] += 1  # 规则应验
                else:
                    invalid_rules[(premise, conclusion)] += 1  # 规则无效
    support = valid_rules  # 支持度字典，即规则应验次数
    confidence = defaultdict(float)  # 置信度字典
    for premise, conclusion in valid_rules.keys():  # 条件/结论
        rule = (premise, conclusion)
        # 置信度 = 规则发生的次数/条件发生的次数
        confidence[rule] = valid_rules[rule] / num_occurences[premise]

    def print_rule(premise, conclusion, support, confidence, features):
        premise_name = features[premise]
        conclusion_name = features[conclusion]
        print(
            "Rule: If a person buys {0} they will also buy {1}".format(
                premise_name,
                conclusion_name))
        print(
            " - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
        print(" - Support: {0}".format(support[(premise, conclusion)]))
        print("")

    # 得到支持度最高的规则，items() 返回字典所有元素的列表，itemgetter(1) 表示用支持度的值作为键，进行降序排列
    sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)
    for i in range(5):
        print("Rule #{0}".format(i + 1))
        premise, conclusion = sorted_support[i][0]
        print_rule(premise, conclusion, support, confidence, features)

    sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)
    for i in range(5):
        print("Rule #{0}".format(i + 1))
        premise, conclusion = sorted_confidence[i][0]
        print_rule(premise, conclusion, support, confidence, features)
```

Output：

    Rule #1
    Rule: If a person buys cheese they will also buy bananas
    - Confidence: 0.659
    - Support: 27
    Rule #2
    Rule: If a person buys bananas they will also buy cheese
    - Confidence: 0.458
    - Support: 27
    Rule #3
    Rule: If a person buys cheese they will also buy apples
    - Confidence: 0.610
    - Support: 25

    Rule #1
    Rule: If a person buys apples they will also buy cheese
    - Confidence: 0.694
    - Support: 25
    Rule #2
    Rule: If a person buys cheese they will also buy bananas
    - Confidence: 0.659
    - Support: 27
    Rule #3
    Rule: If a person buys bread they will also buy bananas
    - Confidence: 0.630
    - Support: 17

### One Rule 算法

`OneR`(One Rule) 算法根据已有的数据中，具有相同特征值的个体最可能属于哪个类别进行分类。One Rule 就是从四个特征中选择分类效果最好的哪个作为分类依据。

> 假如数据集的某一个特征可以取 0 或 1 两个值。数据集共有三个类别。特征值为 0 的情况下，A 类有 20 个这样的个体，B 类有 60 个，C 类也有 20 个。那么特征值为 0 的个体最可能属于 B 类，当然还有 40 个个体确实是特征值为 0，但是它们不属于 B 类。将特征值为 0 的个体分到 B 类的错误率就是 40%，因为有 40 个这样的个体分别属于 A 类和 C 类。特征值为 1 时，计算方法类似，不再赘述；其他各特征值最可能属于的类别及错误率的计算方法也一样。

```python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris  # Iris 植物分类数据集
from collections import defaultdict  # 初始化数据字典
from operator import itemgetter  # 得到一个列表的制定元素
from sklearn.model_selection import train_test_split  # 将一个数据集且分为训练集和测试集
from sklearn.metrics import classification_report  # 分析预测结果

# 这里保留函数的文档方便查阅
def train(X, y_true, feature):
    """
    Computes the predictors and error for a given feature using the OneR algorithm
    Parameters
    ----------
    X: array [n_samples, n_features]
        The two dimensional array that holds the dataset. Each row is a sample, each column
        is a feature.

    y_true: array [n_samples,]
        The one dimensional array that holds the class values. Corresponds to X, such that
        y_true[i] is the class value for sample X[i].

    feature: int
        An integer corresponding to the index of the variable we wish to test.
        0 <= variable < n_features

    Returns
    -------
    predictors: dictionary of tuples: (value, prediction)
        For each item in the array, if the variable has a given value, make the given prediction.

    error: float
        The ratio of training data that this rule incorrectly predicts.
    """
    # 检查是否为有效数字
    n_samples, n_features = X.shape
    assert 0 <= feature < n_features
    # X[:, feature] 为 numpy 矩阵的索引用法，第一维：所有数组，第二维：feature，set 去重得到 value 有几个取值
    # 这个 feature 特征值在每个数据中有多少个取值
    values = set(X[:, feature])
    # Stores the predictors array that is returned
    predictors = dict()
    errors = []
    # 对每个特征值的每个取值调用 train_feature_value 函数获得该取值出现最多的类和错误率
    for current_value in values:
        most_frequent_class, error = train_feature_value(
            X, y_true, feature, current_value)
        predictors[current_value] = most_frequent_class  # 该取值出现最多的类
        errors.append(error)  # 存储错误率
    total_error = sum(errors)
    # 返回预测方案（即 feature 的取值分别对应哪个类别）和总错误率
    return predictors, total_error


def train_feature_value(X, y_true, feature, value):
    class_counts = defaultdict(int)
    # Iterate through each sample and count the frequency of each class/value pair
    # 第 feature 个特征的值为 value 的时候，在每个种类中出现的次数，这里的植物有三个种类
    # 因此最终 class_counts 有三个键值对
    for sample, y in zip(X, y_true):
        if sample[feature] == value:
            class_counts[y] += 1
    # 对 class_count 以 value 由大到小排列
    sorted_class_counts = sorted(
        class_counts.items(),
        key=itemgetter(1),
        reverse=True)
    most_frequent_class = sorted_class_counts[0][0]  # 出现最多次的类
    n_samples = X.shape[1]
    error = sum([class_count for class_value, class_count in class_counts.items(
    ) if class_value != most_frequent_class])  # error 就是除去上面那个类的其它 value 的和
    return most_frequent_class, error  # 返回出现次数最多的类和错误率


def predict(X_test, model):
    variable = model['variable']  # 使用哪个 feature 作为 OneRule 进行预测
    predictor = model['predictor']  # 一个字典，保存着 feature 取值对应哪一类
    y_predicted = np.array([predictor[int(sample[variable])]
                            for sample in X_test])
    return y_predicted  # 返回预测结果


if __name__ == '__main__':
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    n_samples, n_features = X.shape

    # 计算每个属性的均值
    attribute_means = X.mean(axis=0)
    assert attribute_means.shape == (n_features,)
    # 对数据集离散化
    X_d = np.array(X >= attribute_means, dtype='int')

    random_state = 14
    X_train, X_test, y_train, y_test = train_test_split(
        X_d, y, random_state=random_state)  # 分割训练集和测试集
    print("There are {} training samples".format(y_train.shape))  # 训练集数量
    print("There are {} testing samples".format(y_test.shape))  # 测试集数量

    # 对每个特征返回预测器和错误率 [0：{0: x, 1: x}, sum_error， ...]
    all_predictors = {
        variable: train(
            X_train,
            y_train,
            variable) for variable in range(
            X_train.shape[1])}

    errors = {variable: error for variable,
              (mapping, error) in all_predictors.items()}  # 把每个预测器的值提取出来
    # 找出最好（错误最少）的那个 feature 构成的预测器
    best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]
    print(
        "The best model is based on variable {0} and has error {1:.2f}%".format(
            best_variable,
            best_error))

    # Choose the bset model
    model = {'variable': best_variable,
             'predictor': all_predictors[best_variable][0]}
    y_predicted = predict(X_test, model)
    print(classification_report(y_test, y_predicted))  # 生成测试结果
    print(np.mean(y_predicted == y_test) * 100)  # 预测正确率

```

Output：

    |              | precision | recall | f1-score | support |
    |      0       |   0.94    |  1.00  |   0.97   |   17    |
    |      1       |   0.00    |  0.00  |   0.00   |   13    |
    |      2       |   0.40    |  1.00  |   0.57   |    8    |
    |              |           |        |          |         |
    |   accuracy   |           |        |   0.66   |   38    |
    |  macro avg   |   0.45    |  0.67  |   0.51   |   38    |
    | weighted avg |   0.51    |  0.66  |   0.55   |   38    |

    正确率： 65.78947368421053%

## 第二章

主要学习数据挖掘通用框架的搭建方法

- 估计器 (Estimator)：用于分类、聚类和回归分析
- 转换器 (Transformer)：用于数据预处理和数据转换
- 流水线 (Pipline)：组合数据挖掘流程，便于再次使用

### scikit-learn 估计器

估计器用于分类，主要包含下面两个函数：

- `fit()`: 训练算法，设置内部参数。该函数接受训练集和类别两个参数
- `predict()`: 参数为测试集。预测测试集类别，返回一个包含测试集各条数据类别的数组

**近邻算法**

- 用途广泛
- 计算量很大

**距离度量**

- 欧氏距离：即真实距离
- 曼哈顿距离：两个特征在标准坐标系中绝对轴距之和 (x1,y1),(x2,y2) 即 abs(x1-x2)+abs(y1-y2)
- 余弦距离：指的是特征向量夹角的余弦值，更适合解决异常值和数据稀疏问题。

电离层 (Ionosphere) 数据集分析

Input:

```python
# -*- coding: utf-8 -*-
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # 导入 K 近邻分类器
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score  # 导入交叉检验的
# 把每个特征值的值域规范化到 0，1 之间，最小值用 0 代替，最大值用 1 代替
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline  # 流水线


if __name__ == '__main__':
    # 数据集大小已知有 351 行，每行 35 个值前 34 个为天线采集的数据，最后一个 g/b 表示数据的好坏
    X = np.zeros((351, 34), dtype='float')
    y = np.zeros((351,), dtype='bool')

    # 打开根目录的数据集文件
    with open("ionosphere.data", 'r', encoding='utf-8') as input_file:
        # 创建 csv 阅读器对象
        reader = csv.reader(input_file)
        # 使用枚举函数为每行数据创建索引
        for i, row in enumerate(reader):
            # 获取行数据的前 34 个值，并将其转化为浮点型，保存在 X 中
            data = [float(datum) for datum in row[:-1]]
            # Set the appropriate row in our dataset
            X[i] = data  # 数据集
            # 1 if the class is 'g', 0 otherwise
            y[i] = row[-1] == 'g'  # 类别

    # 创建训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
    print(
        "There are {} samples in the training dataset".format(
            X_train.shape[0]))
    print(
        "There are {} samples in the testing dataset".format(
            X_test.shape[0]))
    print("Each sample has {} features".format(X_train.shape[1]))
```

Output:

    There are 263 samples in the training dataset
    There are 88 samples in the testing dataset
    Each sample has 34 features

---

Input:

```python
    # 初始化一个 K 近邻分类器实例，该算法默认选择 5 个近邻作为分类依据
    estimator = KNeighborsClassifier()
    # 用训练数据进行训练
    estimator.fit(X_train, y_train)
    # 使用测试集测试算法，评价其表现
    y_predicted = estimator.predict(X_test)
    # 准确性
    accuracy = np.mean(y_test == y_predicted) * 100
    print("The accuracy is {0:.1f}%".format(accuracy))

    # 使用交叉检验的方式获得平均准确性
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    average_accuracy = np.mean(scores) * 100
    print("The average accuracy is {0:.1f}%".format(average_accuracy))
```

Output:

    The accuracy is 86.4%
    The average accuracy is 82.6%

---

Input:

```python
    # 设置参数
    # 参数的选取跟数据集的特征息息相关
    avg_scores = []
    all_scores = []
    parameter_values = list(range(1, 21))
    for n_neighbors in parameter_values:
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(estimator, X, y, scoring='accuracy')
        avg_scores.append(np.mean(scores))
        all_scores.append(scores)

    # 作出 n_neighbors 不同取值和分类正确率之间的关系的折线图
    plt.figure(figsize=(32, 20))
    plt.plot(parameter_values, avg_scores, '-o', linewidth=5, markersize=24)
    plt.show()
```

Output:

![result2.1](/img/in-post/data-mining/ch2/result2.1.png)

经过上面的例子，可以总结数据挖掘最简单基本的流程如下：

- 载入数据集，数据分类提取到内存中
- 创建训练集和测试集
- 选择合适的算法进行训练
- 使用测试集测试算法，评估其表现

为了保证算法的准确性，可以将大数据集分为几个部分，通过交叉检验的方法测试算法。使用 cross_val_score 函数是一个不错的选择。

在参数的设置上，可以针对不同的参数进行交叉测试，使用图表直观地表示出参数的影响。

### 流水线在预处理中的作用

sckit-learn 的预处理工具叫做转换器`Transformer`

Input:

```python
    # 模拟脏数据
    X_broken = np.array(X)
    X_broken[:, ::2] /= 10
    # 对比两种情况下预测准确率
    estimator = KNeighborsClassifier()
    original_scores = cross_val_score(estimator, X, y, scoring='accuracy')
    print(
        "The original average accuracy for is {0:.1f}%".format(
            np.mean(original_scores) * 100))
    broken_scores = cross_val_score(estimator, X_broken, y, scoring='accuracy')
    print(
        "The broken average accuracy for is {0:.1f}%".format(
            np.mean(broken_scores) * 100))
```

Output:

    The original average accuracy for is 82.6%
    The broken average accuracy for is 73.8%

---

Input:

```python
    # 组合成为一个工作流
    X_transformed = MinMaxScaler.fit_transform(X_broken)    # 完成训练和转换
    estimator = KNeighborsClassifier()
    transformed_scores = cross_val_score(
        estimator, X_transformed, y, scoring='accuracy')
    print("The average accuracy for is {0:.1f}%".format(
        np.mean(transformed_scores) * 100))
```

Output:

    The average accuracy for is 82.9%

将数据经过规范化后，正确率再次提高

其它的规范化函数举例：

- 为使每条数据各特征值的和为 1：`sklearn.preprocessing.Normalizer`
- 为使各特征值的均值为 0，方差为 1：`sklearn.preprocessing.StandardScaler`
- 为将数值型特征二值化：`sklearn.preprocessing.Binarizer`

### 流水线

`sklearn.pipeline.Pipeline`用于创建流水线。流水线的输入为一连串的数据挖掘步骤，最后一步必须是估计器，前几步是转换器。

Input:

```python
    # 创建流水线
    # 流水线的每一步都用 ('名称',步骤) 的元组表示
    scaling_pipeline = Pipeline([('scale', MinMaxScaler()),  # 规范特征取值
                                 ('predict', KNeighborsClassifier())])  # 预测

    # 调用流水线
    scores = cross_val_score(scaling_pipeline, X_broken, y, scoring='accuracy')
    print(
        "The pipelin scored an average accuracy for is {0:.1f}%".format(
            np.mean(scores) * 100))
```

Output:

    The pipelin scored an average accuracy for is 82.9%

## 第三章

决策树也是一种分类算法，它的优点如下：

- 机器和人都能看懂
- 能够处理多种不同的特征

### 加载数据集

pandas(Python Data Analysis 的简写)

逗号分隔值（Comma-Separated Values，CSV，有时也称为字符分隔值，因为分隔字符也可以不是逗号），其文件以纯文本形式存储表格数据（数字和文本），来源[百度百科](https://baike.baidu.com/item/CSV/10739?fr=aladdin)。

这里使用 `pandas` 导入.csv 文件，生成一个 `dataframe` （数据框）的类。导入使用 `read_csv()` 函数，常用参数如下：

- `sep=','` 以，为数据分隔符
- `parse_dates='col_name'` 将某个特征值读取为日期格式
- `error_bad_lines=False` 当某行数据有问题时，跳过而不报错
- `skiprows=[<param>]` 跳过列表中所包括的行，参数可以是 0,1,...的数字序列，也可以用切片表达式`[0:]`
- `usecols=[<param>]` 选择使用哪几个特征值，参数同上

在使用 `dataframe.ix[]`获取 `dataframe` 中的某几行数据时，提示错误信息，原因是 `pandas` 在 0.20.0 版本后就废弃掉了这个函数。在这里我改为使用 `iloc` 函数。

Input:

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier  # 创建决策树的类
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder  # 能将字符串类型的特征转化成整型
from sklearn.preprocessing import OneHotEncoder  # 将特征转化为二进制数字
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.model_selection import GridSearchCV  # 网格搜索，找到最佳参数

if __name__ == '__main__':
    # 清洗数据集
    results = pd.read_csv(
        "NBA_data.csv", parse_dates=["Date"], skiprows=[
            0, ], usecols=[
            0, 2, 3, 4, 5, 6, 7, 9])  # 加载数据集
    # 修复数据特征名
    results.columns = [
        "Date",
        "Visitor Team",
        "VisitorPts",
        "Home Team",
        "HomePts",
        "Score Type",
        "OT?",
        "Notes"]
    # results.ix[] 已被弃用
    print(results.loc[:5])  # 查看数据集前五行
```

Output:

            Date          Visitor Team  VisitorPts  ... Score Type  OT? Notes
    0 2013-10-29         Orlando Magic          87  ...  Box Score  NaN   NaN
    1 2013-10-29         Chicago Bulls          95  ...  Box Score  NaN   NaN
    2 2013-10-29  Los Angeles Clippers         103  ...  Box Score  NaN   NaN
    3 2013-10-30         Brooklyn Nets          94  ...  Box Score  NaN   NaN
    4 2013-10-30        Boston Celtics          87  ...  Box Score  NaN   NaN
    5 2013-10-30            Miami Heat         110  ...  Box Score  NaN   NaN
    [6 rows x 8 columns]

### 决策树

创建新的特征列，可以从数据集中导入：

`dataset["New Feature"] = feature_creator()`

也可以一开始为新特征值设置默认的值：

`dataset["My New Feature"] = 0`

这里的 `X_previouswins = results[["HomeLastWin", "VisitorLastWin"]].values` 生成一个数据集，这个数据集有两个特征

`DecisionTreeClassifier()` 用来创建决策树，常用参数如下：

- `min_samples_split`: 指定了创建一个新节点至少需要多少个个体
- `min_samples_leaf`: 指定为了保留节点，每个节点至少应该包含的个体数量
- 创建决策的标准：基尼不纯度/信息增益

Input:

```python
    # 提取新特征，值为这场中主场队伍是否胜利
    results["HomeWin"] = results["VisitorPts"] < results["HomePts"]
    y_true = results["HomeWin"].values  # 胜负情况
    # 创建两个新 feature，初始值都设为 0，保存这场比赛的两个队伍上场比赛的情况
    results["HomeLastWin"] = False
    results["VisitorLastWin"] = False
    won_last = defaultdict(int)

    for index, row in results.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        # 这场比赛之前两个球队上次是否获胜保存在 result 中
        row["HomeLastWin"] = won_last[home_team]
        row["VisitorLastWin"] = won_last[visitor_team]
        results.iloc[index] = row
        # 这场比赛的结果更新 won_last 中的情况
        won_last[home_team] = row["HomeWin"]
        won_last[visitor_team] = not row["HomeWin"]

    X_previouswins = results[["HomeLastWin", "VisitorLastWin"]].values
    # 创建决策树生成器实例
    clf = DecisionTreeClassifier(random_state=14)
    # 交叉训练
    scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
    print("Using just the last result from the home and visitor teams")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
```

Output:

    Using just the last result from the home and visitor teams
    Accuracy: 56.4%

---

这里为了创建一个新的特征导入了上一年的 NBA 排名。

Input:

```python
    ladder = pd.read_csv("NBA_standings.csv", skiprows=[0, ])
    # 创建一个新特征，两个队伍在上个赛季的排名哪个比较高
    results["HomeTeamRanksHigher"] = 0
    for index, row in results.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        # 这个球队改名了
        if home_team == "New Orleans Pelicans":
            home_team = "New Orleans Hornets"
        elif visitor_team == "New Orleans Pelicans":
            visitor_team = "New Orleans Hornets"
        # 这里源代码无法运行，少加了一个括号 ladder[(ladder["Team"] == home_team)] 表示根据条件获取这一行的数据
        home_row = ladder[(ladder["Team"] == home_team)]
        visitor_row = ladder[(ladder["Team"] == visitor_team)]
        home_rank = home_row["Rk"].values[0]
        visitor_rank = visitor_row["Rk"].values[0]
        row["HomeTeamRanksHigher"] = int(home_rank > visitor_rank)
        results.iloc[index] = row

    X_homehigher = results[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values
    clf = DecisionTreeClassifier(random_state=14)
    scores = cross_val_score(clf, X_homehigher, y_true, scoring='accuracy')
    print("Using whether the home team is ranked higher")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
```

Output:

    Using whether the home team is ranked higher
    Accuracy: 60.0%

---

Input:

```python
    # 创建新特征，两个队伍上一次进行比赛时的获胜者
    last_match_winner = defaultdict(int)
    results["HomeTeamWonLast"] = 0
    for index, row in results.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        # 按照英文字母表排序，不去考虑哪个是主场球队
        teams = tuple(sorted([home_team, visitor_team]))
        # 找到两支球队上次比赛的赢家，更新框中的数据，初始为 0
        # 这里的 HomeTeamWonLast 跟主场客场没有什么关系，也可以叫 WhichTeamWonLast，这里为了和源码尽量保持一致使用了源码
        row["HomeTeamWonLast"] = 1 if last_match_winner[teams] == row["Home Team"] else 0
        results.iloc[index] = row
        winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
        # 将两个球队上次遇见比赛的情况存到字典中去
        last_match_winner[teams] = winner

    X_home_higher = results[["HomeTeamRanksHigher", "HomeTeamWonLast"]].values
    clf = DecisionTreeClassifier(random_state=14)
    scores = cross_val_score(clf, X_home_higher, y_true, scoring='accuracy')
    print("Using whether the home team is ranked higher")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
```

Output:

    Using whether the home team is ranked higher
    Accuracy: 59.9%

### 随机森林

`LabelEncoder()` 用来将一个字符串型的特征转化为整型

`OneHotEncoder()` 将整数转化成消除差异的二进制数字，即将 1,2,3 转换成 001,010,100

stacking（向量组合），这里 `np.vstack()` 将两个队伍名向量纵向组合成一个矩阵`.T`表示将矩阵转置

决策树存在的问题：

1. 创建的多颗决策树在很大程度上是相同的，训练集相同，则生成的决策树也相同。一个解决办法是*装袋*(bagging)
2. 用于前几个决策节点的特征非常突出，即使采用不同的训练集，创建的决策树相似性依旧很大。解决办法是随机选取部分特征作为决策数据

`RandomForestClassifier()` 用来调用随机森林算法，因为它调用了 DecisionTreeClassifier 的大量实例，所以他们的参数有很多是一致的。其引入的一部分新参数如下：

- `n_estimators` 用来指定创建决策树的数量，值越高，耗时越长，准确率 (可能) 越高
- `oob_score` 如果设置为真，测试时将不适用训练模型时用过的数据
- `n_jobs` 采用并行算法训练时所用到的内核数量，设置为 -1 则启用全部内核

Input:

```python
    # 创建一个转化器实例
    encoding = LabelEncoder()
    # 将球队名转化为整型
    encoding.fit(results["Home Team"].values)
    # 抽取所有比赛中主客场球队的球队名，组合起来形成一个矩阵
    home_teams = encoding.transform(results["Home Team"].values)
    visitor_teams = encoding.transform(results["Visitor Team"].values)
    # 建立训练集，[["Home Team Feature"，"Visitor Team Feature"],["Home Team Feature"，"Visitor Team Feature"]...]
    X_teams = np.vstack([home_teams, visitor_teams]).T
    # 创建转化器实例
    onehot = OneHotEncoder()
    # 生成转化后的特征
    X_teams = onehot.fit_transform(X_teams).todense()

    clf = DecisionTreeClassifier(random_state=14)
    scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

    clf = RandomForestClassifier(random_state=14, n_jobs=-1)
    scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
    print("Using full team labels is ranked higher")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
```

Output:

    Accuracy: 60.5%
    Using full team labels is ranked higher
    Accuracy: 61.4%

---

将上面生成的特征整合起来，创建新的决策方案

这里使用 `np.hstack()`横向拼接两个决策方案矩阵

Input:

```python
    X_all = np.hstack([X_home_higher, X_teams])  # 将上面计算的特征进行组合
    print(X_all.shape)
    scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
    print("Using whether the home team is ranked higher")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
```

Output:

    (1319, 62)
    Using whether the home team is ranked higher
    Accuracy: 61.6%

---

使用 `GridSearchCV` （网格搜索）搜索最佳参数

Input:

```python
    # 设置参数搜索范围
    parameter_space = {
        "max_features": [2, 10, 'auto'],
        "n_estimators": [100, ],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [2, 4, 6],
    }
    grid = GridSearchCV(clf, parameter_space)
    grid.fit(X_all, y_true)
    print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))
    # 输出最佳方案
    print(grid.best_estimator_)
```

Output:

    Accuracy: 65.6%
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                        criterion='gini', max_depth=None, max_features='auto',
                        max_leaf_nodes=None, max_samples=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=2, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=100,
                        n_jobs=-1, oob_score=False, random_state=14, verbose=0,
                        warm_start=False)

### 课后练习

拿到了数据，如何创建新的特征，如何在数据中发现其关键点，如何找出数据内部的联系，也是一个需要斟酌的方面

创建下述特征并看一下效果：

- 球队上次打比赛距今有多长时间？
- 两支球队过去五场比赛结果如何？
- 球队是不是跟某支特定球队打比赛时发挥更好？

在这里使用了上面书中的方法，完成了前两个点，第三个点实现起来有点麻烦，现在只有一个思路：建立一个字典，数据形式为 (两支球队建立一个元组:(前一个队伍获胜的次数，后一个队伍获胜的次数))

在处理 dataset 中的数据项时，对于 `pandas` 中的 `Timestamp` 类型没有了解，耗费了太长时间，查阅文档后发现可以用 `date()` 将其转化为 `datetime.date` 日期。

使用前两个特征作为决策标准时，效果还算可以，加上书上的所有特征后，准确率反而较上面的结果降低了。（不知道为什么）

这个“课后练习”使我对于标准库了解匮乏的短板显现出来，要抽出时间学习一下 `python`, `numpy` 和 `pandas` 标准库中常用函数及其参数。

Input:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ch3.nba_test import X_all
from sklearn.model_selection import GridSearchCV  # 网格搜索，找到最佳参数


if __name__ == '__main__':
    """
    - 球队上次打比赛距今有多长时间？
    - 两支球队过去五场比赛结果如何？
    - 球队是不是跟某支特定球队打比赛时发挥更好？
    """
    dataset = pd.read_csv(
        "NBA_data.csv", parse_dates=["Date"], skiprows=[
            0, ], usecols=[
            0, 2, 3, 4, 5, 6, 7, 9])  # 加载数据集
    dataset.columns = [
        "Date",
        "Visitor Team",
        "VisitorPts",
        "Home Team",
        "HomePts",
        "Score Type",
        "OT?",
        "Notes"]
    dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
    y_true = dataset["HomeWin"].values  # 胜负情况

    # 保存上次打比赛的时间
    last_played_date = defaultdict(datetime.date)
    # 手动为每个球队初始化
    for team in set(dataset["Home Team"]):
        last_played_date[team] = datetime.date(year=2013, month=10, day=25)
    # 两支球队过去的比赛结果，每个球队的数据是 [True,False,,,] 的序列
    last_five_games = defaultdict(list)

    # 存放 Home 和 Visitor 前五次比赛的获胜次数
    dataset["HWinTimes"] = 0
    dataset["VWinTimes"] = 0
    # 存放距离上次比赛的时间间隔，用天计数
    dataset["HLastPlayedSpan"] = 0
    dataset["VLastPlayedSpan"] = 0
    for index, row in dataset.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]

        row["HWinTimes"] = sum(last_five_games[home_team][-5:])
        row["VWinTimes"] = sum(last_five_games[visitor_team][-5:])
        row["HLastPlayedSpan"] = (
            row["Date"].date() -
            last_played_date[home_team]).days
        row["VLastPlayedSpan"] = (
            row["Date"].date() -
            last_played_date[visitor_team]).days

        dataset.iloc[index] = row

        last_played_date[home_team] = row["Date"].date()
        last_played_date[visitor_team] = row["Date"].date()
        last_five_games[home_team].append(row["HomeWin"])
        last_five_games[visitor_team].append(not row["HomeWin"])

    X_1 = dataset[["HLastPlayedSpan",
                             "VLastPlayedSpan", "HWinTimes", "VWinTimes"]].values
    clf = DecisionTreeClassifier(random_state=14)
    scores = cross_val_score(clf, X_1, y_true, scoring='accuracy')
    print("DecisionTree: Using time span and win times")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

    clf = RandomForestClassifier(random_state=14, n_jobs=-1)
    scores = cross_val_score(clf, X_1, y_true, scoring='accuracy')
    print("RandomForest: Using time span and win times")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
    print("---------------------------------")

    X_all = np.hstack([X_1, X_all])

    clf = DecisionTreeClassifier(random_state=14)
    scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
    print("DecisionTree: Using time span and win times")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

    clf = RandomForestClassifier(random_state=14, n_jobs=-1)
    scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
    print("RandomForest: Using time span and win times")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
    print("---------------------------------")
    parameter_space = {
        "max_features": [2, 10, 'auto'],
        "n_estimators": [100, ],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [2, 4, 6],
    }
    grid = GridSearchCV(clf, parameter_space)
    grid.fit(X_all, y_true)
    print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))
    print(grid.best_estimator_)
```

Output:

    DecisionTree: Using time span and win times
    Accuracy: 56.4%
    RandomForest: Using time span and win times
    Accuracy: 58.3%
    ---------------------------------
    DecisionTree: Using time span and win times
    Accuracy: 57.2%
    RandomForest: Using time span and win times
    Accuracy: 61.0%
    ---------------------------------
    Accuracy: 64.6%
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                        criterion='entropy', max_depth=None, max_features=2,
                        max_leaf_nodes=None, max_samples=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=4, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=100,
                        n_jobs=-1, oob_score=False, random_state=14, verbose=0,
                        warm_start=False)

![数据集情况](/img/in-post/data-mining/ch3/homework.jpg)

## 第四章

本章重点：

- 亲和性分析
- 用 Apriori 算法挖掘关联特征
- 数据稀疏问题

### 亲和性分析

Apriori 算法是经典的亲和性分析算法，它只从数据集中频繁出现的商品中选取出共同出现的商品组成频繁项集，避免了复杂度呈指数级增长的问题。一旦找到频繁项集，生成关联规则就变得容易了。

原理：确保了规则在数据集中有足够的支持度。Apriori 算法一个重要参数就是最小支持度，如果想要生成 (A,B,C) 的频繁项集，则其子集必须都要满足最小支持度标准。

其它亲和性算法还有 Eclat 和频繁项集挖掘算法 (FP-growth)。这些算法比起基础的 Apriori 算法有很多改进，性能也有进一步提升。

第一阶段，为 Apriori 算法指定一个项集要成为频繁项集所需的最小支持度。第二阶段，根据置信度取关联规则，设定最小置信度，返回大于此值的规则。

### 电影推荐问题

[下载](http://files.grouplens.org/datasets/movielens/ml-100k.zip)并加载数据集

Input:

```python
import sys
import pandas as pd
from collections import defaultdict
from operator import itemgetter


if __name__ == '__main__':
    # header=None 不把第一行当做表头
    all_ratings = pd.read_csv(
        "ml-100k/u.data",
        delimiter="\t",
        header=None,
        names=[
            "UserID",
            "MovieID",
            "Rating",
            "Datetime"])
    # 转化时间戳为 datetime
    all_ratings["Datetime"] = pd.to_datetime(all_ratings["Datetime"], unit='s')
    # 输出用户 - 电影 - 评分稀疏矩阵
    print(all_ratings[:5])
    print()
    # 创建 Favorite 特征，将评分属性二值化为是否喜欢
    all_ratings["Favorable"] = all_ratings["Rating"] > 3
    # 取用户 ID 为前 200 的用户的打分数据
    ratings = all_ratings[all_ratings["UserID"].isin(range(200))]
    favorable_ratings = ratings[ratings["Favorable"]]
    # 创建用户喜欢哪些电影的字典
    favorable_reviews_by_users = dict(
        (k,
         frozenset(
             v.values)) for k,
        v in favorable_ratings.groupby("UserID")["MovieID"])
    # 创建一个数据框，了解每部电影的影迷数量
    num_favorable_by_movie = ratings[[
        "MovieID", "Favorable"]].groupby("MovieID").sum()
    # 查看最受欢迎的五部电影
    print(num_favorable_by_movie.sort_values("Favorable", ascending=False)[:5])
```

Output:

    UserID  MovieID  Rating            Datetime
    0     196      242       3 1997-12-04 15:55:49
    1     186      302       3 1998-04-04 19:22:22
    2      22      377       1 1997-11-07 07:18:36
    3     244       51       2 1997-11-27 05:02:03
    4     166      346       1 1998-02-02 05:33:16

            Favorable
    MovieID
    50           100.0
    100           89.0
    258           83.0
    181           79.0
    174           74.0

### Apriori 算法的实现

1. 把各项目放到只包含自己的项集中，生成最初的频繁项集。只使用达到最小支持度的项目。
2. 查找现有频繁项集的超集，发现新的频繁项集，并用其生成新的备选项集。
3. 测试新生成的备选项集的频繁程度（与最小支持度比较），如果不够频繁则舍弃。如果没有新的频繁项集，就跳到最后一步。
4. 存储新发现的频繁项集，跳到步骤 2
5. 返回所有的频繁项集

Input:

```python
    # 字典保存最新发现的频繁项集
    frequent_itemsets = {}
    min_support = 50

    # 第一步，每一步电影生成只包含它自己的项集
    # frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素
    # 普通集合可变，集合中不能有可变的元素，因此普通集合不能被放在集合中；冻结集合不可变，因此可以被放入集合
    frequent_itemsets[1] = dict((frozenset((movie_id,)),
         row["Favorable"]) for movie_id,
        row in num_favorable_by_movie.iterrows() if row["Favorable"] > min_support)

    # 会有重复，导致喜欢电影 1,50 的人分别为 50,100 但是 {1,50} 的集合有 100 个
    # 两个原因，第一在 current_superset 时项集有时候会突然调换位置
    def find_frequent_itemsets(
            favorable_reviews_by_users,
            k_1_itemsets,
            min_support):
        counts = defaultdict(int)
        # 遍历每一个用户，获取其喜欢的电影
        for user, reviews in favorable_reviews_by_users.items():
            # 遍历每个项集
            for itemset in k_1_itemsets:
                if itemset.issubset(reviews):  # 判断 itemset 是否是用户喜欢的电影的子集
                    # 对用户喜欢的电影中除了这个子集的电影进行遍历
                    for other_reviewed_movie in reviews - itemset:
                        # 将该电影并入项集中
                        current_superset = itemset | frozenset(
                            {other_reviewed_movie})
                        counts[current_superset] += 1  # 这个项集的支持度 +1
        # 返回元素数目 +1 的项集和数量
        res = dict([(itemset, frequency) for itemset,
                                             frequency in counts.items() if frequency >= min_support])
        return res

    for k in range(2, 20):
        cur_frequent_itemsets = find_frequent_itemsets(
            favorable_reviews_by_users, frequent_itemsets[k - 1], min_support)
        frequent_itemsets[k] = cur_frequent_itemsets
        if len(cur_frequent_itemsets) == 0:
            print("Did not find any frequent itemsets of length {}".format(k))
            sys.stdout.flush()  # 将缓冲区内容输出到终端，不宜多用，输出操作带来的计算开销会拖慢程序运行速度
            break
        else:
            print(
                "I found {} frequent itemsets of length {}".format(
                    len(cur_frequent_itemsets), k))
            sys.stdout.flush()
    # 除去只包含一个元素的初始集合
    del frequent_itemsets[1]
```

Output:

    I found 93 frequent itemsets of length 2
    I found 295 frequent itemsets of length 3
    I found 593 frequent itemsets of length 4
    I found 785 frequent itemsets of length 5
    I found 677 frequent itemsets of length 6
    I found 373 frequent itemsets of length 7
    I found 126 frequent itemsets of length 8
    I found 24 frequent itemsets of length 9
    I found 2 frequent itemsets of length 10
    Did not find any frequent itemsets of length 11

### 抽取关联规则

对每个频繁项集，选出其中的一个元素当结论，剩下的元素都作为条件，生成规则。

Input:

```python
    # 规则形式：如果用户喜欢前提中的所有电影，那么他们也会喜欢结论中的电影
    candidate_rules = []
    for itemset_length, itemset_counts in frequent_itemsets.items():
        for itemset in itemset_counts.keys():
            for conclusion in itemset:
                premise = itemset - {conclusion}
                candidate_rules.append((premise, conclusion))
    print(candidate_rules[:5])
```

Output:

    [(frozenset({7}), 1), (frozenset({1}), 7), (frozenset({50}), 1), (frozenset({1}), 50), (frozenset({1}), 56)]

---

置信度计算，方法与第一章类似。

```python
    # 计算置信度
    correct_counts = defaultdict(int)
    incorrect_counts = defaultdict(int)

    # 遍历每一个用户，获取其喜欢的电影
    for user, reviews in favorable_reviews_by_users.items():
        # 遍历每个规则
        for candidate_rule in candidate_rules:
            # 获取规则的条件和结论
            premise, conclusion = candidate_rule
            # 如果条件是喜欢电影的子集（条件成立）
            if premise.issubset(reviews):
                # 如果用户也喜欢结论的电影
                if conclusion in reviews:
                    correct_counts[candidate_rule] += 1
                else:
                    incorrect_counts[candidate_rule] += 1
    # 计算置信度，结论发生的次数除以条件发生的次数
    rule_confidence = {
        candidate_rule: correct_counts[candidate_rule] /
        float(
            correct_counts[candidate_rule] +
            incorrect_counts[candidate_rule]) for candidate_rule in candidate_rules}
    # 给置信度排序
    sorted_confidence = sorted(
        rule_confidence.items(),
        key=itemgetter(1),
        reverse=True)
    for index in range(5):
        print("Rule #{}".format(index + 1))
        (premise, conclusion) = sorted_confidence[index][0]
        print(
            "Rule: If a person recommends {} they will also recommand {}".format(
                premise,
                conclusion))
        print(
            "- Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
        print("--------------------")
```

Output:

    Rule #1
    Rule: If a person recommends frozenset({98, 181}) they will also recommand 50
    - Confidence: 1.000
    --------------------
    Rule #2
    Rule: If a person recommends frozenset({172, 79}) they will also recommand 174
    - Confidence: 1.000
    --------------------
    Rule #3
    Rule: If a person recommends frozenset({258, 172}) they will also recommand 174
    - Confidence: 1.000
    --------------------
    Rule #4
    Rule: If a person recommends frozenset({1, 181, 7}) they will also recommand 50
    - Confidence: 1.000
    --------------------
    Rule #5
    Rule: If a person recommends frozenset({1, 172, 7}) they will also recommand 174
    - Confidence: 1.000
    --------------------

---

调整输出，加上电影名

Input:

```python
    movie_name_data = pd.read_csv(
        "ml-100k/u.item",
        delimiter='|',
        header=None,
        encoding="mac-roman")
    movie_name_data.columns = [
        'MovieID',
        'Title',
        'Release Date',
        'Video Release',
        'IMDB',
        '<UNK>',
        'Action',
        'Adventure',
        'Animation',
        "Children's",
        'Comedy',
        'Crime',
        'Documentary',
        'Drama',
        'Fantasy',
        'Film-Noir',
        'Horror',
        'Musical',
        'Mystery',
        'Romance',
        'Sci-Fi',
        'Thriller',
        'War',
        'Western']

    for index in range(5):
        print('Rule #{0}'.format(index + 1))
        (premise, conclusion) = sorted_confidence[index][0]
        premise_names = ', '.join(get_movie_name(idx) for idx in premise)
        conclusion_name = get_movie_name(conclusion)
        print(
            'Rule: if a person recommends {0} they will also recommend {1}'.format(
                premise_names,
                conclusion_name))
        print(
            ' - Confidence: {0:.3f}'.format(rule_confidence[(premise, conclusion)]))
        print("--------------------")
```

Output:

    Rule #1
    Rule: if a person recommends Silence of the Lambs, The (1991), Return of the Jedi (1983) they will also recommend Star Wars (1977)
    - Confidence: 1.000
    --------------------
    Rule #2
    Rule: if a person recommends Empire Strikes Back, The (1980), Fugitive, The (1993) they will also recommend Raiders of the Lost Ark (1981)
    - Confidence: 1.000
    --------------------
    Rule #3
    Rule: if a person recommends Contact (1997), Empire Strikes Back, The (1980) they will also recommend Raiders of the Lost Ark (1981)
    - Confidence: 1.000
    --------------------
    Rule #4
    Rule: if a person recommends Toy Story (1995), Return of the Jedi (1983), Twelve Monkeys (1995) they will also recommend Star Wars (1977)
    - Confidence: 1.000
    --------------------
    Rule #5
    Rule: if a person recommends Toy Story (1995), Empire Strikes Back, The (1980), Twelve Monkeys (1995) they will also recommend Raiders of the Lost Ark (1981)
    - Confidence: 1.000
    --------------------

### 评估测试

使用剩下的数据集计算规则的置信度，也是查看每条规则表现的一个方法。

Input:

```python
    # 评估测试
    test_dataset = all_ratings[~all_ratings['UserID'].isin(range(200))]
    test_favorable = test_dataset[test_dataset["Favorable"]]
    test_favorable_by_users = dict((k, frozenset(v.values))
                                   for k, v in test_favorable.groupby("UserID")["MovieID"])

    correct_counts = defaultdict(int)
    incorrect_counts = defaultdict(int)
    for user, reviews in test_favorable_by_users.items():
        for candidate_rule in candidate_rules:
            premise, conclusion = candidate_rule
            if premise.issubset(reviews):
                if conclusion in reviews:
                    correct_counts[candidate_rule] += 1
                else:
                    incorrect_counts[candidate_rule] += 1

    test_confidence = {
        candidate_rule: correct_counts[candidate_rule] /
        float(
            correct_counts[candidate_rule] +
            incorrect_counts[candidate_rule]) for candidate_rule in rule_confidence}
    for index in range(5):
        print("Rule #{0}".format(index + 1))
        (premise, conclusion) = sorted_confidence[index][0]
        premise_names = ", ".join(get_movie_name(idx) for idx in premise)
        conclusion_name = get_movie_name(conclusion)
        print(
            'Rule: if a person recommends {0} they will also recommend {1}'.format(
                premise_names,
                conclusion_name))
        print(
            ' - Confidence: {0:.3f}'.format(rule_confidence[(premise, conclusion)]))
        print("--------------------")
```

Output:

    Rule #1
    Rule: if a person recommends Silence of the Lambs, The (1991), Return of the Jedi (1983) they will also recommend Star Wars (1977)
    - Confidence: 1.000
    --------------------
    Rule #2
    Rule: if a person recommends Empire Strikes Back, The (1980), Fugitive, The (1993) they will also recommend Raiders of the Lost Ark (1981)
    - Confidence: 1.000
    --------------------
    Rule #3
    Rule: if a person recommends Contact (1997), Empire Strikes Back, The (1980) they will also recommend Raiders of the Lost Ark (1981)
    - Confidence: 1.000
    --------------------
    Rule #4
    Rule: if a person recommends Toy Story (1995), Return of the Jedi (1983), Twelve Monkeys (1995) they will also recommend Star Wars (1977)
    - Confidence: 1.000
    --------------------
    Rule #5
    Rule: if a person recommends Toy Story (1995), Empire Strikes Back, The (1980), Twelve Monkeys (1995) they will also recommend Raiders of the Lost Ark (1981)
    - Confidence: 1.000
    --------------------

这一章用电影进行亲和度分析，由于元素的数量变多了，时间复杂度呈指数级增长，遍历的笨方法已经不适用。需要寻找更加巧妙地解决方案。

在用集合计算电影的项集时，`{1, 2}` 与 `{2, 1}` 是同一个事件，但在遍历的时候会被多次计算，可能这是一个错误的点。

## 第五章

本章讨论如何从数据集中抽取数值和类别型特征，并选出最佳特征。还会介绍特征抽取的常用模式和技巧。

### 特征抽取

把实体用特征表示出来，通过特征建模，再通过机器挖掘算法能够理解的近似方式来表示现实。

特征可以是数值型或类别型。数值特征可以离散化生成类别特征。

Input:

```python
import numpy as np
import pandas as pd

if __name__ == '__main__':
    adult = pd.read_csv("adult.data", header=None, names=["Age", "Work-Class", "fnlwgt", "Education",
                                                          "Education-Num", "Marital-Status", "Occupation",
                                                          "Relationship", "Race", "Sex", "Capital-gain",
                                                          "Capital-loss", "Hours-per-week", "Native-Country",
                                                          "Earnings-Raw"])
    # 去除空值
    adult.dropna(how='all', inplace=True)
    # 输出详细描述
    print(adult["Hours-per-week"].describe())
    # 输出中位数
    print(adult["Education-Num"].median())
    # 输出工作的种类
    print(adult["Work-Class"].unique())
    # 将工作时长二值化为是否超过 40h
    adult["LongHours"] = adult["Hours-per-week"] > 40
```

Output:

    count    32561.000000
    mean        40.437456
    std         12.347429
    min          1.000000
    25%         40.000000
    50%         40.000000
    75%         45.000000
    max         99.000000
    Name: Hours-per-week, dtype: float64
    10.0
    [' State-gov' ' Self-emp-not-inc' ' Private' ' Federal-gov' ' Local-gov'
    ' ?' ' Self-emp-inc' ' Without-pay' ' Never-worked']

### 特征选择

实物的特征有很多，我们只选择其中一小部分。

- 降低复杂度，提高算法运行速度
- 减低噪音，增加无关的特征会干扰算法的工作
- 增加模型可读性，特征较少，人们易于理解

拿到数据后，先进行简单直接的分析，了解数据的特点。

`sklearn.feature_selection.VarianceThreshold` 转换器可以用来删除特征值的方差达不到最低标准的特征。

Input:

```python
    # 构造测试数据集
    X = np.arange(30).reshape((10, 3))
    X[:, 1] = 1
    print(X)
    print("----------------")
    vt = VarianceThreshold()
    Xt = vt.fit_transform(X)
    # 第二列消失了，因为第二列都是 1，方差为 0，不包括具有区别意义的信息
    print(Xt)
    print("----------------")
    print(vt.variances_)
```

Output:

    [[ 0  1  2]
    [ 3  1  5]
    [ 6  1  8]
    [ 9  1 11]
    [12  1 14]
    [15  1 17]
    [18  1 20]
    [21  1 23]
    [24  1 26]
    [27  1 29]]
    ----------------
    [[ 0  2]
    [ 3  5]
    [ 6  8]
    [ 9 11]
    [12 14]
    [15 17]
    [18 20]
    [21 23]
    [24 26]
    [27 29]]
    ----------------
    [27.  0. 27.]

---

选择最佳特征

随着特征数量的增加，寻找最佳特征组合的任务复杂度呈指数级增长。分类任务通常的做法是寻找表现好的单个特征，依据是他们能达到的精确度。

scikit-learn 提供了几个用于选择单变量特征的转换器。

- SelectKBest 返回 k 个最佳特征
- SelectPercentile 返回表现最佳的 r%个特征

这两个转换器都提供计算特征表现的一系列方法。

单个特征和某一类别之间的相关性计算方法有卡方检验 (x²)、互信息和信息熵等。

Input:

```python
    # 构造数据集
    X = adult[["Age",
               "Education-Num",
               "Capital-gain",
               "Capital-loss",
               "Hours-per-week"]]
    y = (adult["Earnings-Raw"] == ' >50K').values
    # 使用 SelectKBest 转换器，用卡方打分
    transformer = SelectKBest(score_func=chi2, k=3)
    # 调用 fit_transform 方法对相同的数据集进行预处理和转换
    Xt_chi2 = transformer.fit_transform(X, y)
    # 输出每个特征的得分
    print(transformer.scores_)
    print("----------------")

    # 用皮尔逊相关系数计算相关性，创建包装函数
    def mutivariate_pearsonr(X, y):
        scores, pvalues = [], []
        for column in range(X.shape[1]):
            cur_score, cur_p = pearsonr(X[:, column], y)
            scores.append(abs(cur_score))
            pvalues.append(cur_p)
        return np.array(scores), np.array(pvalues)

    transformer = SelectKBest(score_func=mutivariate_pearsonr, k=3)
    Xt_pearson = transformer.fit_transform(X, y)
    print(transformer.scores_)
    print("----------------")

    clf = DecisionTreeClassifier(random_state=14)
    scores_chi2 = cross_val_score(clf, Xt_chi2, y, scoring='accuracy')
    scores_pearson = cross_val_score(clf, Xt_pearson, y, scoring='accuracy')
    print('卡方：{}'.format(np.mean(scores_chi2)))
    print("----------------")
    print("pearson:  {}".format(np.mean(scores_pearson)))
```

Output:

    [8.60061182e+03 2.40142178e+03 8.21924671e+07 1.37214589e+06 6.47640900e+03]
    ----------------
    [0.2340371  0.33515395 0.22332882 0.15052631 0.22968907]
    ----------------
    卡方: 0.8291514400795839
    ----------------
    pearson:  0.7721507467016449

### 创建特征

特征之间相关性很强，或者特征冗余，会增加算法处理难度。

这里在加载 ad 数据集之前先创建了一个转换器，用于在加载时转换数据集中的值。

源码运行会产生报错，第一个原因是，用函数初始化转换器并没有把函数名传入，因此将 defaultdict 中每一个索引都进行了初始化。第二个原因是，PCA 转换器无法对 NaN 数据进行处理，于是我在处生成数据集之前将所有含有 NaN 的行删掉。

Input:

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


# 创建转换函数
def convert_number(x):
    try:
        res = float(x)
        return res
    except ValueError:
        return np.nan


if __name__ == '__main__':
    # 创建数据加载的转换器
    converters = defaultdict(convert_number, {i: convert_number for i in range(1588)})
    converters[1558] = lambda x: 1 if x.strip() == "ad." else 0
    # 使用转换器读取数据集
    temp = pd.read_csv("ad.data", header=None, converters=converters)
    # 删除所有含有 nan 的行，axis=0 是数据索引 (index)，axis=1 是列标签 (column)
    ads = temp.dropna(axis=0, how='any')
    print(ads[10:15])
```

Output:

           0      1       2     3     4     5  ...  1553  1554  1555  1556  1557  1558
    11  90.0   52.0  0.5777   1.0   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0     1
    12  90.0   60.0  0.6666   1.0   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0     1
    13  90.0   60.0  0.6666   1.0   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0     1
    14  33.0  230.0  6.9696   1.0   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0     1
    15  60.0  468.0  7.8000   1.0   0.0   0.0  ...   0.0   1.0   1.0   0.0   0.0     1
    [5 rows x 1559 columns]

---

主成分分析 (PCA)

目的是找到能用较少信息描述数据集的特征组合。主成分的方差跟整体方差没有多大差距。经过分析主成分，第一个特征的方差对数据集方差的贡献率为 85.4%，第二个为 14.5%，后面越来越少。

Input:

```python
    X = ads.drop(1558, axis=1).values
    y = ads[1558]
    # 参数为主成分数量
    pca = PCA(n_components=5)
    Xd = pca.fit_transform(X)
    # 设置输出选项
    # 第一个参数为输出精度位数，第二个参数是使用定点表示法打印浮点数
    np.set_printoptions(precision=3, suppress=True)
    print(pca.explained_variance_ratio_)
```

Output:

    [0.854 0.145 0.001 0.    0.   ]

---

使用随机森林验证模型正确率，并将 pca 转换结果绘制出来。

Input:

```python
    clf = DecisionTreeClassifier(random_state=14)
    scores_reduced = cross_val_score(clf, Xd, y, scoring='accuracy')
    print(np.mean(scores_reduced))

    # 获取数据集类别的所有取值
    classes = set(y)
    # 指定在图形中用什么颜色表示这两个类别
    colors = ['red', 'green']
    # 同时遍历这两个容器
    for cur_class, color in zip(classes, colors):
        # 为属于当前类别的所有个体创建遮罩层
        mask = (y == cur_class).values
        plt.scatter(Xd[mask, 0], Xd[mask, 1], marker='o',
                    color=color, label=int(cur_class))
    plt.legend()
    plt.show()
```

Output:

    0.936405592140775

![pca](/img/in-post/data-mining/ch5/pca.png "输出结果")

### 创建自己的转换器

转换器有两个关键函数

- `fit()` 接收训练数据，设置内部参数
- `transform()` 转换过程。接收训练数据集或相同格式的新数据集

接口要与 scikit-learn 接口一致，便于在流水线中使用。

Input:

```python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array
from numpy.testing import assert_array_equal


class MeanDiscrete(TransformerMixin):
    def fit(self, X):
        # 尝试对 X 进行转换，数据转换成 float 类型
        X = as_float_array(X)
        # 计算数据集的均值
        self.mean = X.mean(axis=0)
        # 返回它本身，进行链式调用 transformer.fit(X).transform(X)
        return self

    def transform(self, X):
        X = as_float_array(X)
        # 检查输入是否合法
        assert X.shape[1] == self.mean.shape[0]
        # 返回 X 中大于均值的数据
        return X > self.mean


def test_meandiscrete():
    X_test = np.array([[0, 2], [3, 5], [6, 8], [9, 11], [12, 14], [15, 17], [18, 20], [21, 23], [24, 26], [27, 29]])
    mean_discrete = MeanDiscrete()
    mean_discrete.fit(X_test)
    # 与正确的计算结果进行比较，检查内部参数是否正确设置
    assert_array_equal(mean_discrete.mean, np.array([13.5, 15.5]))
    # 转换后的 X
    X_transfromed = mean_discrete.transform(X_test)
    # 验证数据
    X_expected = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
    assert_array_equal(X_transfromed, X_expected)


if __name__ == '__main__':
    test_meandiscrete()
```

Output:

    # 没有输出，说明测试通过

## 第六章

本章介绍如何从文本数据中提取特征。通过强大却简单的朴素贝叶斯算法消除社会媒体用语的歧义。

朴素贝叶斯算法在计算用于分类的概率时，为了简化计算，假定各特征间是相互独立的，因此名字中含有*朴素*二字。

### 消歧

由于无法申请到 Twitter app 暂时搁置。。。%>\_<%

### 文本转换器

词袋：一种最简单却非常有效的模型就是只统计数据集中每个单词的出现次数。模型主要分为以下三种

- 使用词语实际出现的次数作为词频。缺点是当文章长度明显差异时，词频差距会非常大。
- 使用归一化后的词频，每篇文章中所有词语的词频之和为 1
- 直接使用二值特征来表示，单词在文档中出现值为 1，不出现值为 0

还有一种更通用的规范化方法叫做*词频 - 逆文档频率法*，该加权方法用词频来代替词的出现次数，然后再用词频除以包含该词的文档的数量。

Input:

```python
# -*- coding: utf-8 -*-
from collections import Counter

if __name__ == '__main__':
    s = """Three Rings for the Elven-kings under the sky,
Seven for the Dwarf-lords in halls of stone,
Nine for Mortal Men, doomed to die,
One for the Dark Lord on his dark throne
In the Land of Mordor where the Shadows lie.
One Ring to rule them all, One Ring to find them,
One Ring to bring them all and in the darkness bind them.
In the Land of Mordor where the Shadows lie""".lower()
    words = s.split()
    c = Counter(words)
    # 输出出现次数最多的前 5 个词
    print(c.most_common(5))
```

Output:

    [('the', 9), ('for', 4), ('in', 4), ('to', 4), ('one', 4)]

---

N 元语法是指由几个连续的词组成的子序列。

### 朴素贝叶斯

我们用 C 表示某种类别，用 D 表示数据集中一篇文档，来计算贝叶斯公式所要用到的各种统计量，对于不好计算，出朴素假设，简化计算。朴素贝叶斯分类算法使用贝叶斯定理计算个体从属于某一类别的概率。

`P(C)` 为某一类别的概率，可以从训练集中计算得到（方法跟上文检测垃圾邮件例子所用到的一致）。统计训练集所有文档从属于给定类别的百分比。

`P(D)` 为某一文档的概率，它牵扯到各种特征，计算起来很困难，但是在计算文档属于哪个类别时，对于所有类别来说，P(D) 相同，因此根本就不用计算它。稍后我们来看下怎么处理。

`P(D|C)` 为文档 D 属于 C 类的概率。由于 D 包含多个特征，计算起来可能很困难，这时朴素贝叶斯算法就派上用场了。我们朴素地假定各个特征之间是相互独立的，分别计算每个特征（D1、D2、D3 等）在给定类别出现的概率，再求它们的积。

`P(D|C) = P(D1|C) x P(D2|C) ... x P(Dn|C)`

举例说明下计算过程，假如数据集中有以下一条用二值特征表示的数据：[1, 0, 0, 1]

训练集中有 75% 的数据属于类别 0，25% 属于类别 1，且每个特征属于每个类别的似然度如下。

- 类别 0：[0.3, 0.4, 0.4, 0.7]
- 类别 1：[0.7, 0.3, 0.4, 0.9]

拿类别 0 中特征 1 的似然度举例子，上面这两行数据可以这样理解：类别 0 中有 30% 的数据，特征 1 的值为 1。

我们来计算一下这条数据属于类别 0 的概率。类别为 0 时，P(C=0) = 0.75。

朴素贝叶斯算法用不到 P(D)，因此我们不用计算它。

P(D|C=0) = P(D1|C=0) x P(D2|C=0) x P(D3|C=0) x P(D4|C=0)
= 0.3 x 0.6 x 0.6 x 0.7
= 0.0756

我们就可以计算该条数据从属于每个类别的概率。我们没有计算 P(D)，因此，计算结果不是实际的概率。由于两次都不计算 P(D)，结果具有可比较性，能够区分出大小就足够了。来看下计算结果。

P(C=0|D) = P(C=0) P(D|C=0)
= 0.75 \* 0.0756
= 0.0567

P(D|C=1) = P(D1|C=1) x P(D2|C=1) x P(D3|C=1) x P(D4|C=1)
= 0.7 x 0.7 x 0.6 x 0.9
= 0.2646

P(C=1|D) = P(C=1)P(D|C=1)
= 0.25 \* 0.2646
= 0.06615

因此这条数据属于类别 1 的概率大于属于类别 2 的概率

### 应用

创建流水线，接收一条消息，仅根据消息内容，确定它与编程语言 Python 是否相关。

- 用 NLTK 的 word_tokenize 函数，将原始文档转换为由单词及其是否出现组成的字典。
- 用 scikit-learn 中的 DictVectorizer 转换器将字典转换为向量矩阵，这样朴素贝叶斯分类器就能使用第一步中抽取的特征。
- 正如前几章做过的那样，训练朴素贝叶斯分类器。
- 还需要新建一个笔记本文件 ch6_classify_twitter（本章最后一个），用于分类。

F1 值来评估算法

F1 值是以每个类别为基础进行定义的，包括两大概念：准确率（precision）和召回率（recall）。准确率是指预测结果属于某一类的个体，实际属于该类的比例。召回率是指被正确预测为某个类别的个体数量与数据集中该类别个体总量的比例

Input:

```python
# -*- coding: utf-8 -*-
import json
import numpy as np
from sklearn.base import TransformerMixin
from nltk import word_tokenize
from sklearn.feature_extraction import DictVectorizer  # 接受元素为字典的列表，将其转换为矩阵
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB  # 用于二值特征分类的 BernoulliNB 分类器，
from sklearn.pipeline import Pipeline


# 创建转换器类
class NLTKBOW(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [{word: True for word in word_tokenize(document)} for document in X]


if __name__ == '__main__':
    tweets = []
    input_filename = ""
    classes_filename = ""
    with open(input_filename) as inf:
        for line in inf:
            if len(line.strip()) == 0:
                continue
            tweets.append(json.loads(line)['text'])

    with open(classes_filename, 'r') as inf:
        labels = json.load(inf)

    # 组装流水线
    pipline = Pipeline([('bag-of-words', NLTKBOW()), ('vectorizer', DictVectorizer()), ('naive-bayes', BernoulliNB())])
    # 用 F1 值来评估
    scores = cross_val_score(pipline, tweets, labels, scoring='f1')
    print("Score: {:.3f}".format(np.mean(scores)))

    model = pipline.fit(tweets, labels)
    nb = model.named_steps['naive-bayes']
    feature_probabilities = nb.feature_log_prob_
    top_features = np.argsort(-feature_probabilities[1])[:50]
    dv = model.named_steps['vectorizer']
    for i, feature_index in enumerate(top_features):
        print(i, dv.feature_names_[feature_index], np.exp(feature_probabilities[1][feature_index]))
```

Output:

    暂时没有数据集

## 第七章

本章介绍的算法引入聚类分析概念--根据相似度，把大数据集划分为几个子集。

### 加载数据集

由于申请不到 Twitter 开发者账号，我想办法爬了一些 b 站用户关注数据，做成了本次试验相仿的形式

Input:

```python
# -*- coding: utf-8 -*-
import json
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import silhouette_score

if __name__ == '__main__':
    with open('bili.txt', mode='r') as fin:
        temp = json.load(fin)
    users = pd.DataFrame(temp)
    users.columns = ['Id', 'Friends']
    print(users[:5])
```

Output:

            Id                                            Friends
    0  214582845  [4370617, 259345180, 186334806, 546195, 477132...
    1    4370617                    [74507, 883968, 122879, 585267]
    2  259345180                                                 []
    3  186334806                                                 []
    4     546195                                                 []

---

将每个记录的用户左右 main_users，把他们关注的人作为边，生成有向图

由于对 matplotlib 库和 networkx 库了解太少，在作图时遇到了许多困难（根基不牢，地动山摇。(>\_<)）

Input:

```python
    G = nx.DiGraph()
    main_users = list(users['Id'].values)
    for u in main_users:
        G.add_node(u, label=u)
    for user in users.values:
        friends = user[1]
        for friend in friends:
            if friend in main_users:
                G.add_edge(user[0], int(friend))
    print('graph finished')
    plt.figure(3, figsize=(100, 100))
    nx.draw(G, alpha=0.1, edge_color='b', with_labels=True, font_size=16, node_size=30, node_color='r')
    plt.savefig('fix1.png')
```

Output:

![fix1](/img/in-post/data-mining/ch7/fix1.png)

---

创建用户相似度图

由于每个用户关注的人数可能相差很大，因此使用杰卡德相似系数（两个用户关注的集合的交集除以并集），该系数在 0 到 1 之间，代表两者重合的比例。

规范化是数据挖掘的一个重要方法，要坚持使用（除非有充足的理由不这样做）

访问<http://networkx.lanl.gov/reference/drawing/html>了解 networkx 的布局方法

Input:

```python
    friends = {user: set(friends) for user, friends in users.values}

    def compute_similarity(friends1, friends2):
        return len(friends1 & friends2) / len(friends1 | friends2)

    def create_graph(followers, threshold=0.0):
        G = nx.Graph()
        for user1 in friends.keys():
            if len(friends[user1]) == 0:
                continue
            for user2 in friends.keys():
                if len(friends[user2]) == 0:
                    continue
                if user1 == user2:
                    continue
                weight = compute_similarity(friends[user1], friends[user2])
                if weight >= threshold:
                    G.add_node(user1, lable=user1)
                    G.add_node(user2, lable=user2)
                    G.add_edge(user1, user2, weight=weight)
        return G

    G = create_graph(friends)
    plt.figure(3, figsize=(100, 100))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=30)
    edgewidth = [d['weight'] for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=edgewidth)
    plt.savefig('fix2.png')
```

Output:

![fix2](/img/in-post/data-mining/ch7/fix2.png)

### 寻找子图

networkx 的 `connected_component_subgraphs()` 函数在 2.1 版本中被移除了（代码过时的比较多，并且使用 Twitter 作为演示数据集让我这两章做的很头疼），我查看官方文档后发现可以使用 `connected_components()` 替代，但是此函数返回的是一个生成器，一次生成一组连通顶点，可以配合 G.subgraph(nodes) 使用获得连通分支

Input:

```python
    # 生成新图，指定最低阈值为 0.1
    G = create_graph(friends, 0.1)
    sub_graphs = nx.connected_components(G)
    for i, sub_graphs in enumerate(sub_graphs):
        n_nodes = len(sub_graphs)
        print("Subgraph{} has {} nodes".format(i, n_nodes))
    print('---------------------')
    G = create_graph(friends, 0.15)
    sub_graphs = nx.connected_components(G)
    for i, sub_graphs in enumerate(sub_graphs):
        n_nodes = len(sub_graphs)
        print("Subgraph{} has {} nodes".format(i, n_nodes))

    sub_graphs = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    n_subgraphs = nx.number_connected_components(G)
    fig = plt.figure(figsize=(20, (n_subgraphs*3)))
    for i, sub_graph in enumerate(sub_graphs):
        # sub_graph 是一个连通分支顶点的集合
        ax = fig.add_subplot(int(n_subgraphs / 3) + 1, 3, i + 1)
        # 将坐标轴标签关掉
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        pos = nx.spring_layout(G)
        nx.draw(G=G.subgraph(sub_graph), alpha=0.1, edge_color='b', with_labels=True, font_size=16, node_size=30, node_color='r', ax=ax)
    plt.show()
```

Output:

![fix3](/img/in-post/data-mining/ch7/fix3.png)

---

轮廓系数定义： `s = (b - a) / max(a, b)`

其中 a 为簇内距离，表示与簇内其它个体之间的平均距离。b 为簇间距离，也就是最近簇内各个个体之间的平均距离

Input:

```python
    def compute_silhouette(threshold, friends):
        G = create_graph(friends, threshold=threshold)\
        # 图是否至少有两个顶点
        if len(G.nodes()) < 2:
            # 返回 -99 表示问题无效
            return -99
        # 抽取连通分支
        sub_graphs = nx.connected_components(G)
        # 至少有两个连通分支
        if not (2 <= nx.number_connected_components(G) < len(G.nodes()) - 1):
            return -99
        label_dict = {}
        for i, sub_graph in enumerate(sub_graphs):
            for node in sub_graph:
                # 给不同连通分支的顶点分配不同的标签
                label_dict[node] = i
        labels = np.array([label_dict[node] for node in G.nodes()])
        X = nx.to_scipy_sparse_matrix(G).todense()
        # 这里要将相似度转换为距离，所以用最大相似度减去现有相似度，把相似度转化为距离
        X = 1 - X
        # 这里将距离矩阵的对角线处理为 0，因为自己到自己的距离为 0
        np.fill_diagonal(X, 0)
        return silhouette_score(X, labels, metric='precomputed')


    def inverted_silhouette(threshold, friends):
        # 对轮廓系数取反，将打分函数转化成损失函数
        res = compute_silhouette(threshold, friends=friends)
        return - res
    # minimize 函数是一个损失函数，值越小越好
    # 参数：inverted_silhouette 要寻找的函数；0.1 开始时猜测的阈值；options={'maxiter': 10} 只进行 10 轮迭代，增加迭代次数，效果可能更好，但运行时间会增加，method='nelder-mead'使用"下山单纯形法"优化方法
    result = minimize(inverted_silhouette, 0.1, args=(friends,), options={'maxiter': 10})
    print(result.x)
```

Output:

    [0.10005086]

---

本章探讨了社交网络和图以及如何对其进行聚类分析。目标是推荐用户，使用聚类分析方法能够找到不同的用户簇，主要步骤有根据相似度创建加权图，从图中寻找连通分支。创建图时用到了 NetworkX 库。

还比较了几对意义相反的概念。对于两者之间的相似度这个概念，值越大，表明两者之间更相像。相反，对于距离而言，值越小，两者更相像。另外一对是损失函数和打分函数。对于损失函数，值越小，效果越好（也就是损失越少）。而对于打分函数，值越大，效果越好。

## 第八章

本章使用神经网络分析自己生成的验证码图像

### 人工神经网络

*神经网络*算法最初是根据人类大脑的工作机制设计的。神经网络由一系列相互连接的神经元组成。每个神经元都是一个简单的函数，接收一定输入，给出相应输出。

神经元可以使用任何标准函数来处理数据，比如线性函数，这些函数统称为激活函数（activation function）。一般来说，神经网络学习算法要能正常工作，激活函数应当是可导（derivable）和光滑的。常用的激活函数有*逻辑斯谛*函数，函数表达式如下（x 为神经元的输入，k、L 通常为 1，这时函数达到最大值）。

$$
f(x) = \frac{L}{1+e^{-k(x-x_{0})}}
$$

每个神经元接收几个输入，根据这几个输入，计算输出。这样的一个个神经元连接在一起组成了神经网络，对数据挖掘应用来说，它非常强大。这些神经元紧密连接，密切配合，能够通过学习得到一个模型，使得神经网络成为机器学习领域最强大的概念之一。

数据挖掘应用的神经网络，神经元按照层级进行排列，至少有三层

1. 第一层：输入层。用来接收数据集的输入。第一层中的每个神经元对输入进行计算，把得到的结果传给第二层的神经元。这种叫作*前向神经网络*
2. 隐含层：数据表现方式令人难以理解，一层或多层
3. 最后一层：输出层。输出结果表示的是神经网络分类器给出的分类结果

神经元激活函数通常使用逻辑斯谛函数，每层神经元之间为全连接，创建和训练神经网络还需要用到其他几个参数。

创建过程，指定神经网络的规模需要用到两个参数：神经网络共有多少层，隐含层每层有多少个神经元（输入层和输出层神经元数量通常由数据集来定）。

### 创建数据集

使用长度为 4 个字母的英文单词作为验证码

Input:

```python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops  # 用于图像分割
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from pybrain.datasets.supervised import SupervisedDataSet  # 神经网络数据集
from pybrain.tools.shortcuts import buildNetwork  # 构建神经网络
from pybrain.supervised.trainers.backprop import BackpropTrainer  # 反向传播算法
from sklearn.metrics import f1_score
from nltk.corpus import words  # 导入语料库 用于生成单词
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from nltk.metrics import edit_distance  # 编辑距离
from operator import itemgetter


# 用于生成验证码，接收一个单词和错切值，返回用 numpy 数组格式表示的图像
def create_captcha(text, shear=0.0, size=(100, 26)):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    # 验证码文字所用字体，该开源字体可在 github 下载
    font = ImageFont.truetype("FiraCode-Medium.otf", 22)
    draw.text((0, 0), text, fill=1, font=font)
    # 将 PIL 图像转换为 numpy 数组，以便用 scikit-image 库为图像添加错切变化效果
    image = np.array(im)
    # 应用错切变化效果
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    # 对图像进行归一化处理，确保特征值落在 0 到 1 之间
    return image / image.max()


if __name__ == '__main__':
    image = create_captcha('GENE', shear=0.5)
    plt.imshow(image, cmap='Greys')
    plt.show()
```

Output:

![8.1](/img/in-post/data-mining/ch8/myplot1.png)

---

将图像切分为单个的字母

Input:

```python
    def segment_image(image):
        """
        接收图像，返回小图像列表
        :param image:
        :return:
        """
        # 找出像素值相同又连接在一起的像素块，类似上一章的连通分支
        labeled_image = label(image > 0)
        subimages = []
        for region in regionprops(labeled_image):
            # 获取当前位置的起始和结束坐标
            start_x, start_y, end_x, end_y = region.bbox
            subimages.append(image[start_x:end_x, start_y:end_y])
        # 如果没有找到小图像，则将原图像作为子图返回
        if len(subimages) == 0:
            return [image, ]
        return subimages


    subimages = segment_image(image)
    f, axes = plt.subplots(1, len(subimages), figsize=(10, 3))
    for i in range(len(subimages)):
        axes[i].imshow(subimages[i], cmap='gray')
    plt.show()
```

Output:

![8.2](/img/in-post/data-mining/ch8/myplot2.png)

---

创建训练集

Input:

```python
    # 指定随机状态值
    random_state = check_random_state(14)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    shear_values = np.arange(0, 0.5, 0.05)

    # 用来生成一条训练数据
    def generate_sample(random_state=None):
        random_state = check_random_state(random_state)
        letter = random_state.choice(letters)
        shear = random_state.choice(shear_values)
        return create_captcha(letter, shear=shear, size=(25, 25)), letters.index(letter)

    image, target = generate_sample(random_state)
    plt.imshow(image, cmap='Greys')
    print("The target for this image is {}".format(target))
    plt.show()

    # 调用 3000 次此函数，生成训练数据传到 numpy 的数组里
    dataset, targets = zip(*(generate_sample(random_state) for i in range(3000)))
    dataset = np.array(dataset, dtype=float)
    targets = np.array(targets)

    # 对 26 个字母类别进行编码
    onehot = OneHotEncoder()
    y = onehot.fit_transform(targets.reshape(targets.shape[0], 1))
    # 将稀疏矩阵转换为密集矩阵
    y = y.todense()

    # 调整图像大小
    dataset = np.array([resize(segment_image(sample)[0], (20, 20)) for sample in dataset])
    # 将最后三维的 dataset 的后二维扁平化
    X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
```

Output:

    The target for this image is 11

![8.3](/img/in-post/data-mining/ch8/myplot3.png)

### 训练和分类

反向传播算法（back propagation，backprop）的工作机制为对预测错误的神经元施以惩罚。从输出层开始，向上层层查找预测错误的神经元，微调这些神经元输入值的权重，以达到修复输出错误的目的。

神经元之所以给出错误的预测，原因在于它前面为其提供输入的神经元，更确切来说是由这两个神经元之间边的权重及输入值决定的。我们可以尝试对权重进行微调。每次调整的幅度取决于以下两个方面

- 神经元各边权重的误差函数的偏导数
- 一个叫作学习速率的参数（通常使用很小的值）

计算出函数误差的梯度，再乘以学习速率，用总权重减去得到的值。梯度的符号由误差决定，每次对权重的修正都是朝着给出正确的预测值努力。有时候，修正结果为局部最优（local optima），比起其他权重组合要好，但所得到的各权重还不是最优组合。

反向传播算法从输出层开始，层层向上回溯到输入层。到达输入层后，所有边的权重更新完毕。

这里在导入 `SupervisedDataSet` 时发生了错误，使用 `pip install pybrain` 安装的包会有找不到方法的现象，因此我从 github-pybrain 下载了源码包，在解压后的文件夹中输入 `python setup.py install` 进行安装，解决了这个问题。还有一个问题是原文使用 `from pybrain.datasets import SupervisedDataSet` 来导入 `SupervisedDataSet` 但是我在导入时发现并没有这个类，于是看了项目结构后使用 `from pybrain.datasets.supervised import SupervisedDataSet` 进行导入。还有几处相同的问题均是这样解决的。

这里在使用 f1_score 进行评估时也出现了错误，原因见代码注释。

Input:

```python
    # 为 pybrain 库创建格式适配的数据集
    training = SupervisedDataSet(X.shape[1], y.shape[1])
    for i in range(X_train.shape[0]):
        training.addSample(X_train[i], y_train[i])
    testing = SupervisedDataSet(X.shape[1], y.shape[1])
    for i in range(X_test.shape[0]):
        testing.addSample(X_test[i], y_test[i])
    # 指定维度，创建神经网络，第一个参数为输入层神经元数量，第二个参数隐含层神经元数量，第三个参数为输出层神经元数量
    # bias 在每一层使用一个一直处于激活状态的偏置神经元
    net = buildNetwork(X.shape[1], 100, y.shape[1], bias=True)

    # 使用反向传播算法调整权重
    trainer = BackpropTrainer(net, training, learningrate=0.01, weightdecay=0.01)
    # 设定代码的运行步数
    trainer.trainEpochs(epochs=20)
    # 预测值
    predictions = trainer.testOnClassData(dataset=testing)
    # f1_score 的 average 默认值为'binary'，如果不指定 average 则会发生 ValueError
    print("F-score:{0:.2f}".format(f1_score(y_test.argmax(axis=1), predictions, average='weighted')))
    print("F-score:{0:.2f}".format(f1_score(y_test.argmax(axis=1), predictions, average='micro')))
    print("F-score:{0:.2f}".format(f1_score(y_test.argmax(axis=1), predictions, average='macro')))
```

Output:

    F-score:1.00
    F-score:1.00
    F-score:1.00

---

预测单词

Input:

```python
    # 接收验证码，用神经网络进行训练，返回单词预测结果
    def predict_captcha(captcha_image, neural_network):
        subimages = segment_image(captcha_image)
        predicted_word = ""
        # 遍历四张小图像
        for subimage in subimages:
            # 调整每张小图像的大小为 20*20 像素
            subimage = resize(subimage, (20,20))
            # 把小图像数据传入神经网络的输入层，激活神经网络。这些数据将在神经网络中进行传播，返回输出结果
            outputs = net.activate(subimage.flatten())
            # 神经网络输出 26 个值，每个值都有索引号，分别对应 letters 列表中有着相同索引的字母，每个值的大小表示与对应字母的相似度。为了获得实际的预测值，我们取到最大值的索引，再通过 letters 列表找到对应的字母
            prediction = np.argmax(outputs)
            # 把上面得到的字母添加到正在预测的单词中
            predicted_word += letters[prediction]
        return predicted_word

    word = "GENE"
    captcha = create_captcha(word, shear=0.2)
    print(predict_captcha(captcha, net))
```

Output:

    GENE

---

nltk 下载语料库时可能会很慢，需要的可以在这里[下载](https://pan.baidu.com/s/1mYm_1CdkNrVScHyiCyIdnQ "e3pw")。如何离线安装 nltk 语料库自行百度。

Input:

```python
    def test_prediction(word, net, shear=0.2):
        captcha = create_captcha(word, shear=shear)
        prediction = predict_captcha(captcha, net)
        prediction = prediction[:4]
        # 返回预测结果是否正确，验证码中的单词和预测结果的前四个字符
        return word == prediction, word, prediction

    # 语料库中字长为 4 的单词列表
    valid_words = [word.upper() for word in words.words() if len(word) == 4]
    num_correct = 0
    num_incorrect = 0
    for word in valid_words:
        correct, word, prediction = test_prediction(word, net, shear=0.2)
        if correct:
            num_correct += 1
        else:
            num_incorrect += 1
    print("Number correct is {}".format(num_correct))
    print("Number incorrect is {}".format(num_incorrect))

    # 二维混淆矩阵，每行每列均为一个类别
    cm = confusion_matrix(np.argmax(y_test,axis=1), predictions)
    # 混淆矩阵作图
    plt.figure(figsize=(20, 20))
    plt.imshow(cm)
    tick_marks = np.arange(len(letters))
    plt.xticks(tick_marks, letters)
    plt.yticks(tick_marks, letters)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
```

Output:

    Number correct is 3738
    Number incorrect is 1775

![8.4](/img/in-post/data-mining/ch8/myplot4.png)

### 用词典提升准确率

假设验证码全部都是英语单词

_列文斯坦编辑距离_（Levenshtein edit distance）是一种通过比较两个短字符串，确定它们相似度的方法。它不太适合扩展，字符串很长时通常不用这种方法。编辑距离需要计算从一个单词变为另一个单词所需要的步骤数。以下操作都算一步

- 在单词的任意位置插入一个新字母
- 从单词中删除任意一个字母
- 把一个字母替换为另外一个字母

Input:

```python
    # 获得两个单词的编辑距离
    steps = edit_distance("STEP", "STOP")
    print("The num of steps needed is: {}".format(steps))

    # 用词长 4 减去同等位置上相同的字母数量，得到的值越小表示两个词相似度越高
    def compute_distance(prediction, word):
        return len(prediction) - sum(prediction[i] == word[i] for i in range(len(prediction)))

    # 改进预测函数
    def improved_prediction(word, net, dictionary, shear=0.2):
        captcha = create_captcha(word, shear=shear)
        prediction = predict_captcha(captcha, net)
        prediction = prediction[:4]
        # 如果单词不在词典中则比较取词典中距离最小的单词
        if prediction not in dictionary:
            distance = sorted([(w, compute_distance(prediction, w)) for w in dictionary], key=itemgetter(1))
            best_word = distance[0]
            prediction = best_word[0]
        return word == prediction, word, prediction


    num_correct = 0
    num_incorrect = 0
    for word in valid_words:
        correct, word, prediction = improved_prediction(word, net, valid_words,shear=0.2)
        if correct:
            num_correct += 1
        else:
            num_incorrect += 1
    print("Number correct is {}".format(num_correct))
    print("Number incorrect is {}".format(num_incorrect))
```

Output:

    The num of steps needed is: 1
    Number correct is 3785
    Number incorrect is 1728

正确率稍有提升

## 第九章

昨天跑去搞 wordpress 搭建网站了 (๑•́ ₃•̀๑) （摸鱼真舒服

本章主要介绍如下内容

- 特征工程和如何根据应用选择特征
- 带着新问题，重新回顾词袋模型
- 特征类型和字符 N 元语法模型
- 支持向量机
- 数据集清洗

### 为作品找到作者

作者归属可以看作是一种分类问题，已知一部分作者，数据集为多个作者的作品（训练集），目标是确定一组作者不详的作品（测试集）是谁写的。如果作者恰好是已知的作者里面的，这种问题叫作封闭问题

如果作者可能不在里面，这种问题就叫作开放问题

获取数据，书中的链接有很多已经失效，我参考网上的取得了下载方式。

Input:

```python
# -*- coding: utf-8 -*-
# get_data.py
import requests
import os
import time
from collections import defaultdict

titles = {'burton': [4657, 2400, 5760, 6036, 7111, 8821, 18506, 4658, 5761, 6886, 7113],
          'dickens': [24022, 1392, 1414, 1467, 2324, 580, 786, 888, 963, 27924, 1394, 1415, 15618, 25985, 588, 807,
                      914, 967, 30127, 1400, 1421, 16023, 28198, 644, 809, 917, 968, 1023, 1406, 1422, 17879, 30368,
                      675, 810, 924, 98, 1289, 1413, 1423, 17880, 32241, 699, 821, 927],
          'doyle': [2349, 11656, 1644, 22357, 2347, 290, 34627, 5148, 8394, 26153, 12555, 1661, 23059, 2348, 294,
                    355, 5260, 8727, 10446, 126, 17398, 2343, 2350, 3070, 356, 5317, 903, 10581, 13152, 2038, 2344,
                    244, 32536, 423, 537, 108, 139, 2097, 2345, 24951, 32777, 4295, 7964, 11413, 1638, 21768, 2346,
                    2845, 3289, 439, 834],
          'gaboriau': [1748, 1651, 2736, 3336, 4604, 4002, 2451, 305, 3802, 547],
          'nesbit': [34219, 23661, 28804, 4378, 778, 20404, 28725, 33028, 4513, 794],
          'tarkington': [1098, 15855, 1983, 297, 402, 5798, 8740, 980, 1158, 1611, 2326, 30092, 483, 5949, 8867,
                         13275, 18259, 2595, 3428, 5756, 6401, 9659],
          'twain': [1044, 1213, 245, 30092, 3176, 3179, 3183, 3189, 74, 86, 1086, 142, 2572, 3173, 3177, 3180, 3186,
                    3192, 76, 91, 119, 1837, 2895, 3174, 3178, 3181, 3187, 3432, 8525]}

assert len(titles) == 7

assert len(titles['tarkington']) == 22
assert len(titles['dickens']) == 44
assert len(titles['nesbit']) == 10
assert len(titles['doyle']) == 51
assert len(titles['twain']) == 29
assert len(titles['burton']) == 11
assert len(titles['gaboriau']) == 10

url_base = 'http://www.gutenberg.org/files/'
url_format = '{url_base}{id}/{id}-0.txt'

# 修复 URL
url_fix_format = 'http://www.gutenberg.org/cache/epub/{id}/pg{id}.txt'

fiexes = defaultdict(list)
# fixes = {}
# fixes[4657] = 'http://www.gutenberg.org/cache/epub/4657/pg4657.txt'

# make parent folder if not exists
# data_folder = os.path.join(os.path.expanduser('~'),'Data','books') #
# 这是在用户 user 目录中存储
data_folder = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    print(data_folder)

    for author in titles:
        print('Downloading titles from', author)
        # make author's folder if not exists
        author_folder = os.path.join(data_folder, author)
        if not os.path.exists(author_folder):
            os.makedirs(author_folder)
        # download each title to this folder
        for bookid in titles[author]:
            # if bookid in fixes:
            #     print(' - Applying fix to book with id', bookid)
            #     url = fixes[bookid]
            # else:
            #     print(' - Getting book with id', bookid)
            #     url = url_format.format(url_base=url_base, id=bookid)

            url = url_format.format(url_base=url_base, id=bookid)
            print(' - ', url)
            filename = os.path.join(author_folder, '%s.txt' % bookid)
            if os.path.exists(filename):
                print(' - File already exists, skipping')
            else:
                r = requests.get(url)
                if r.status_code == 404:
                    print('url 404:', author, bookid, 'add to fixes list')
                    fiexes[author].append(bookid)
                else:
                    txt = r.text
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(txt)
                time.sleep(1)
    print('Download complete')

    print('开始下载修复列表')
    for author in fiexes:
        print('开始下载<%s>的作品' % author)
        author_folder = os.path.join(data_folder, author)
        if not os.path.exists(author_folder):
            os.makedirs(author_folder)

        for bookid in fiexes[author]:
            filename = os.path.join(author_folder, '%s.txt' % bookid)
            if os.path.exists(filename):
                print('文件已经下载，跳过')
            else:
                url_fix = url_fix_format.format(id=bookid)
                print(' - ', url_fix)
                r = requests.get(url_fix)
                if r.status_code == 404:
                    print('又出错了！', author, bookid)
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(r.text)
                time.sleep(1)
    print('修复列表下载完毕')
```

最后下载完成有 177 本书

---

支持向量机是一种二类分类器，扩展后可用来对多个类别进行分类 9（对于多种类别的分类问题，我们创建多个 SVM 分类器——每个还是二类分类器）

C 参数对于训练 SVM 来说很重要，C 参数与分类器正确分类比例相关，但可能带来过拟合的风险。C 值越高，间隔越小，表示要尽可能把所有数据正确分类。C 值越小，间隔越大——有些数据将无法正确分类。C 值低，过拟合训练数据的可能性就低，但是分类效果可能会相对较差

SVM（基础形式）局限性之一就是只能用来对线性可分的数据进行分类。如果数据线性不可分，就要用到内核函数，将其置入更高维的空间中，加入更多伪特征直到数据线性可分。常用的内核函数有几种。线性内核最简单，它无外乎两个个体的特征向量的点积、带权重的特征和偏置项。多项式核提高点积的阶数（比如 2）。此外，还有高斯内核（rbf）、Sigmoind 内核

Input:

```python
# -*- coding: utf-8 -*-
# author_test.py
import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC  # 支持向量机
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from ch9 import getdata


# 去掉古藤堡的说明
def clean_book(document):
    lines = document.split("\n")
    start = 0
    end = len(lines)
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("*** START OF THIS PROJECT GUTENBERG"):
            start = i + 1
        elif line.startswith("*** END OF THIS PROJECT GUTENBERG"):
            end = i - 1
    return "\n".join(lines[start:end])


def load_books_data(folder=getdata.data_folder):
    # 存储文档和作者
    documents = []
    authors = []
    # 遍历子文件夹
    subfolders = [subfolder for subfolder in os.listdir(folder)
                  if os.path.isdir(os.path.join(folder, subfolder))]
    for author_number, subfolder in enumerate(subfolders):
        full_subfolder_path = os.path.join(folder, subfolder)
        for document_name in os.listdir(full_subfolder_path):
            # 跳过目录下的 getdata.py 文件
            if document_name == 'getdata.cpython-38.pyc':
                continue
            with open(os.path.join(full_subfolder_path, document_name), 'r') as inf:
                documents.append(clean_book(inf.read()))
                authors.append(author_number)
    return documents, np.array(authors, 'int')

# 功能词
function_words = ["a", "able", "aboard", "about", "above", "absent",
                  "according", "accordingly", "across", "after", "against",
                  "ahead", "albeit", "all", "along", "alongside", "although",
                  "am", "amid", "amidst", "among", "amongst", "amount", "an",
                  "and", "another", "anti", "any", "anybody", "anyone",
                  "anything", "are", "around", "as", "aside", "astraddle",
                  "astride", "at", "away", "bar", "barring", "be", "because",
                  "been", "before", "behind", "being", "below", "beneath",
                  "beside", "besides", "better", "between", "beyond", "bit",
                  "both", "but", "by", "can", "certain", "circa", "close",
                  "concerning", "consequently", "considering", "could",
                  "couple", "dare", "deal", "despite", "down", "due", "during",
                  "each", "eight", "eighth", "either", "enough", "every",
                  "everybody", "everyone", "everything", "except", "excepting",
                  "excluding", "failing", "few", "fewer", "fifth", "first",
                  "five", "following", "for", "four", "fourth", "from", "front",
                  "given", "good", "great", "had", "half", "have", "he",
                  "heaps", "hence", "her", "hers", "herself", "him", "himself",
                  "his", "however", "i", "if", "in", "including", "inside",
                  "instead", "into", "is", "it", "its", "itself", "keeping",
                  "lack", "less", "like", "little", "loads", "lots", "majority",
                  "many", "masses", "may", "me", "might", "mine", "minority",
                  "minus", "more", "most", "much", "must", "my", "myself",
                  "near", "need", "neither", "nevertheless", "next", "nine",
                  "ninth", "no", "nobody", "none", "nor", "nothing",
                  "notwithstanding", "number", "numbers", "of", "off", "on",
                  "once", "one", "onto", "opposite", "or", "other", "ought",
                  "our", "ours", "ourselves", "out", "outside", "over", "part",
                  "past", "pending", "per", "pertaining", "place", "plenty",
                  "plethora", "plus", "quantities", "quantity", "quarter",
                  "regarding", "remainder", "respecting", "rest", "round",
                  "save", "saving", "second", "seven", "seventh", "several",
                  "shall", "she", "should", "similar", "since", "six", "sixth",
                  "so", "some", "somebody", "someone", "something", "spite",
                  "such", "ten", "tenth", "than", "thanks", "that", "the",
                  "their", "theirs", "them", "themselves", "then", "thence",
                  "therefore", "these", "they", "third", "this", "those",
                  "though", "three", "through", "throughout", "thru", "thus",
                  "till", "time", "to", "tons", "top", "toward", "towards",
                  "two", "under", "underneath", "unless", "unlike", "until",
                  "unto", "up", "upon", "us", "used", "various", "versus",
                  "via", "view", "wanting", "was", "we", "were", "what",
                  "whatever", "when", "whenever", "where", "whereas",
                  "wherever", "whether", "which", "whichever", "while",
                  "whilst", "who", "whoever", "whole", "whom", "whomever",
                  "whose", "will", "with", "within", "without", "would", "yet",
                  "you", "your", "yours", "yourself", "yourselves"]


if __name__ == '__main__':
    # 获取数据
    documents, classes = load_books_data(getdata.data_folder)
    # 提取特征词
    extractor = CountVectorizer(vocabulary=function_words)
    # 参数字典
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svr = SVC()
    # 使用网格搜索最优参数值
    grid = GridSearchCV(svr, parameters)
    # 使用功能词分类
    pipeline1 = Pipeline([('feature_extraction', extractor),
                          ('clf', grid)])
    scores = cross_val_score(pipeline1, documents, classes, scoring='f1_macro')
    print(np.mean(scores))
```

Output:

    0.7738985477640941
    Score: 0.813

### N 元语法

N 元语法由一系列的 N 个为一组的对象组成，N 为每组对象的个数

Input:

```python
    # 用 N 元语法分类
    pipeline = Pipeline([('feature_extraction', CountVectorizer(analyzer='char', ngram_range=(3, 3))),  # 长度为 3 的 N 元语法
                         ('classifier', grid)
                         ])
    scores = cross_val_score(pipeline, documents, classes, scoring='f1_macro')
    print("Score: {:.3f}".format(np.mean(scores)))
```

Output:

    Score: 0.813

### 安然邮件数据集

- 读取数据集
- 清洗数据
- 组装流水线
- 使用 F 值评估

```python
# -*- coding: utf-8 -*-
import os
from email.parser import Parser  # 邮件解析器
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import check_random_state  # 随机状态实例
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import quotequail

enron_data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maildir")
if __name__ == '__main__':
    p = Parser()


    def get_enron_corpus(num_authors=10, data_folder=enron_data_folder,
                         min_docs_author=10, max_docs_author=100,
                         random_state=None):
        random_state = check_random_state(random_state)
        # 随机对得到的邮箱列表进行排序
        # os.listdir 函数每次返回结果不一定相同，在使用该函数前先排序，从而保持返回结果的一致性
        email_addresses = sorted(os.listdir(data_folder))
        random_state.shuffle(email_addresses)

        documents = []
        classes = []
        author_num = 0
        authors = {}

        # 遍历邮箱文件夹，查找它下面名字中含有“sent”的表示发件箱的子文件夹
        for user in email_addresses:
            users_email_folder = os.path.join(data_folder, user)
            mail_folders = [os.path.join(users_email_folder, subfolder) for subfolder in os.listdir(users_email_folder)
                            if "sent" in subfolder]
            try:
                # 获取子文件夹中的每一封邮件，跳过其中的子文件夹
                authored_emails = [open(os.path.join(mail_folder, email_filename), encoding='cp1252').read()
                                   for mail_folder in mail_folders
                                   for email_filename in os.listdir(mail_folder)]
            except IsADirectoryError:
                continue
            # 获得至少十封邮件
            if len(authored_emails) < min_docs_author:
                continue
            # 最多获取前 100 封邮件
            if len(authored_emails) > max_docs_author:
                authored_emails = authored_emails[:max_docs_author]
            # 解析邮件，获取邮件内容
            contents = [p.parsestr(email)._payload for email in authored_emails]
            documents.extend(contents)
            # 将发件人添加到类列表中，每封邮件添加一次
            classes.extend([author_num] * len(authored_emails))
            # 记录收件人编号，再把编号 +1
            authors[user] = author_num
            author_num += 1
            # 收件人数量达到设置的值跳出循环
            if author_num >= num_authors or author_num >= len(email_addresses):
                break
        # 返回邮件数据集以及收件人字典
        return documents, np.array(classes), authors

    documents, classes, authors = get_enron_corpus(data_folder=enron_data_folder, random_state=14)

    # 移除邮件的回复信息
    def remove_replies(email_contents):
        r = quotequail.unwrap(email_contents)
        if r is None:
            return email_contents
        if 'text_top' in r:
            return r['text_top']  # 字典 r 中存在 text_top，返回它的值
        elif 'text' in r:
            return r['text']
        return email_contents

    documents = [remove_replies(document) for document in documents]

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svr = SVC()
    grid = GridSearchCV(svr, parameters)
    pipeline = Pipeline([('feature_extraction', CountVectorizer(analyzer='char', ngram_range=(3, 3))),
                         ('classifier', grid)
                         ])
    scores = cross_val_score(pipeline, documents, classes, scoring='f1_macro')
    print("Score: {:.3f}".format(np.mean(scores)))
```

Output:

    Score: 0.664

---

从流水线中获得最好的参数组合

Input:

```python
    training_documents, test_documents, y_train, y_test = train_test_split(documents, classes, random_state=14)
    pipeline.fit(training_documents, y_train)
    y_pred = pipeline.predict(test_documents)
    print(pipeline.named_steps['classifier'].best_params_)
```

Output:

    {'C': 10, 'kernel': 'rbf'}

---

绘制混淆矩阵查看分类情况

Input:

```python
    cm = confusion_matrix(y_test, y_pred)
    cm = cm / cm.astype(np.float).sum(axis=1)

    sorted_authors = sorted(authors.keys(), key=lambda x: authors[x])
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, cmap='Blues')
    tick_marks = np.arange(len(sorted_authors))
    plt.xticks(tick_marks, sorted_authors)
    plt.yticks(tick_marks, sorted_authors)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
```

Output:

![9.1](/img/in-post/data-mining/ch9/myplot9.1.png)

## 第十章

这两天在鼓捣 jupyterlab，一开始在服务器上建了一个 lab 环境，可每次连接都要登上几分钟，不知道是服务器 CPU 不行还是网络不行。然后又在本地鼓捣，在 debian 装 nodejs 和 npm 的时候把系统依赖搞崩了，于是狠下心来重装了电脑。。。发生的事情太多，心累。。

昨天重装了 Ubuntu，搞了下美化，安装了必须的软件（别说 Ubuntu 还挺好用，真香）

我保证这是最后一句吐槽了，一定

本章介绍如何对新闻语料进行聚类，以发现其中的趋势和主题。

### 获取新闻文章

这一章的数据集是从 reddit 获得的网页链接，reddit 的 app 审核机制不是很严格 (?) 因此我终于拿到了墙外的 api，使用 requests 下载又费了一番功夫，使用书上源码的 url 下载总是 403 错误，研究了好半天 reddit 的 api，发现 reddit 的 url 改成了 (new, top, ...)，修改之后总算完成了链接的索引

Input:

```python
# get_links.py
# -*- coding: utf-8 -*-
import json
import os
import requests
import getpass
import time

# 需要的一些凭证
CLIENT_ID = "xxxxxxxxxxx"
CLIENT_SECRET = "xxxxxxxxxxx"
USER_AGENT = "python:xxxxxxxxx (by /u/xxxxxxxxx)"
USERNAME = "xxxxxxxx"
PASSWORD = "xxxxxxxxxxxxxx"

# requests 使用代理
proxies = {"http": "socks5://xxxxxx", "https": "socks5://xxxxxx"}


def login(username, password):
    if password is None:
        password = getpass.getpass(
            "Enter reddit password for user {}: ".format(username)
        )
    headers = {"User-Agent": USER_AGENT}
    # 使用凭据设置身份验证对象
    client_auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    post_data = {"grant_type": "password", "username": username, "password": password}
    response = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        proxies=proxies,
        auth=client_auth,
        data=post_data,
        headers=headers,
    )
    return response.json()


if __name__ == "__main__":
    # 调用 login 获取 token
    # token = login(USERNAME, PASSWORD)
    # print(token)

    token = {
        "access_token": "xxxxxxxxxxxxxxxxxxxxxxxx",
        "token_type": "xxxxx",
        "expires_in": 3600,
        "scope": "*",
    }

    def get_links(subreddit, token, n_pages=5):
        # 存放链接信息
        stories = []
        after = None
        for page_number in range(n_pages):
            # 进行调用之前等待，以避免超过 API 限制
            print("等待 2s...")
            time.sleep(2)
            # 设置标头进行调用
            headers = {
                "Authorization": "bearer {}".format(token["access_token"]),
                "User-Agent": USER_AGENT,
            }
            # top 为最热链接，这里也可以换成 new
            url = "https://oauth.reddit.com/r/{}/top?limit=100".format(subreddit)
            if after:
                url += "&after={}".format(after)
            while True:
                try:
                    response = requests.get(
                        url, proxies=proxies, headers=headers, timeout=10
                    )
                    result = response.json()
                    # 获取下一个循环的 cursor
                    after = result["data"]["after"]
                except:
                    print("requests 出错等待...")
                    time.sleep(2)
                else:
                    break
            # 将所有新闻项添加到 story 列表中
            for story in result["data"]["children"]:
                stories.append(
                    (
                        story["data"]["title"],
                        story["data"]["url"],
                        story["data"]["score"],
                    )
                )
        return stories

    stories = get_links("worldnews", token)
    base_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_folder, "raw")
    # 这里我将所有的链接都存在了文件里，因为获取这些网站的内容要很久
    with open(os.path.join(base_folder, "stories2.txt"), "w") as f:
        for link in stories:
            f.write(json.dumps(list(link)))
            f.write("\n")
```

### 从网站抽取文本

api/top 总共有 500 个网站，我又获取了 api/new 的 490 个，总共下载了半个小时，失败了 300。。。

最后成功下载的网站数为 365

```python
# get_data.py
# -*- coding: utf-8 -*-
import hashlib
import os
import requests
import json


proxies = {"http": "socks5://xxxxxxxxxxxx", "https": "socks5://xxxxxxxxxxxxx"}


if __name__ == "__main__":
    base_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_folder, "raw")
    # 读取链接数据
    with open(os.path.join(base_folder, "stories1.txt"), "r") as f:
        temp = f.readlines()
    stories = []
    for l in temp:
        stories.append(json.loads(l))

    # 获取网页内容
    number_errors = 0
    for title, url, score in stories:
        print(url)
        output_filename = hashlib.md5(url.encode()).hexdigest()
        fullpath = os.path.join(data_folder, output_filename + ".txt")
        try:
            response = requests.get(url, proxies=proxies, timeout=10)
            data = response.text
            with open(fullpath, "w") as outf:
                outf.write(data)
        except Exception as e:
            number_errors += 1
            # 输出出错数量
            print("出错：{}".format(number_errors))
```

---

下载下来的网页全是 html 文件，要从中提取出有用的信息，这里使用较为通用的 lxml 库，其它处理 html 的库还有 BeautifulSoup 等。

```python
# get_content.py
# -*- coding: utf-8 -*-
import os
from lxml import html, etree

if __name__ == "__main__":
    base_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_folder, "raw")
    # 输出变成纯文本文件的路径
    text_output_folder = os.path.join(base_folder, "textonly")
    filenames = [
        os.path.join(data_folder, filename) for filename in os.listdir(data_folder)
    ]
    # 存放不可能包含新闻内容的节点
    skip_node_types = ["script", "head", "style", etree.Comment]
    # 把 html 文件解析成 lxml 对象
    def get_text_from_file(filename):
        with open(filename, "r") as inf:
            html_tree = html.parse(inf)
        return get_text_from_node(html_tree.getroot())

    # 抽取子节点中的文本内容，最后返回拼接在一起的所有子节点的文本
    def get_text_from_node(node):
        if len(node) == 0:
            # 没有子节点，直接返回内容
            if node.text:
                return node.text
            else:
                return ""
        else:
            # 有子节点，递归调用得到内容
            results = (
                get_text_from_node(child)
                for child in node
                if child.tag not in skip_node_types
            )
        result = str.join("\n", (r for r in results if len(r) > 1))
        # 检查文本长度
        if len(result) >= 100:
            return result
        else:
            return ""

    for filename in os.listdir(data_folder):
        text = get_text_from_file(os.path.join(data_folder, filename))
        with open(os.path.join(text_output_folder, filename), "w") as outf:
            outf.write(text)
```

### 新闻语料聚类

k-means 算法

k-means 聚类算法迭代寻找最能够代表数据的聚类质心点。算法开始时使用从训练数据中随机选取的几个数据点作为质心点。k-means 中的 k 表示寻找多少个质心点，同时也是算法将会找到的簇的数量。步骤：

- 为每一个数据点分配簇标签  
  为每个个体设置一个标签，将它和最近的质心点联系起来，标签相同的个体属于同一个簇
- 更新各簇的质心点

每次更新质心点时，所有质心点将会小范围移动，这会轻微改变每个数据点在簇内的位置，从而引发下一次迭代时质心点的变动

```python
# -*- coding: utf-8 -*-
import os
from sklearn.cluster import KMeans
# TfidfVectorizer 向量化工具，根据词语出现在多少篇文章中，对词语计数进行加权
# 出现在较多文档中的词语权重较低（用文档集数量除以词语出现在的文档的数量，然后取对数）
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from collections import Counter
from scipy.sparse import csr_matrix  # 稀疏矩阵
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree  # 计算最小生成树 MST
from scipy.sparse.csgraph import connected_components  # 连通分支
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer


base_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_folder, "raw")
text_output_folder = os.path.join(base_folder, "textonly")

if __name__ == "__main__":
    # 分簇的数量
    n_clusters = 10
    pipeline = Pipeline(
        [
            ("feature_extraction", TfidfVectorizer(max_df=0.4)),  # 特征抽取，忽略出现在 40% 文档中的词语（删除功能词）
            ("clusterer", KMeans(n_clusters=n_clusters)),  # 调用 k-means 算法
        ]
    )
    documents = [
        open(os.path.join(text_output_folder, filename)).read()
        for filename in os.listdir(text_output_folder)
    ]
    # 不为 fit 函数指定目标类别，进行训练
    pipeline.fit(documents)
    # 使用训练过的算法预测
    # labels 包含每个数据点的簇标签，标签相同的数据点属于同一个簇，标签本身没有含义
    labels = pipeline.predict(documents)

    # 使用 Counter 类查看每个簇的数据点数量
    c = Counter(labels)
    for cluster_number in range(n_clusters):
        print(
            "Cluster {} contains {} samples".format(cluster_number, c[cluster_number])
        )
```

Output:

    Cluster 0 contains 2 samples
    Cluster 1 contains 4 samples
    Cluster 2 contains 1 samples
    Cluster 3 contains 2 samples
    Cluster 4 contains 329 samples
    Cluster 5 contains 7 samples
    Cluster 6 contains 2 samples
    Cluster 7 contains 13 samples
    Cluster 8 contains 3 samples
    Cluster 9 contains 2 samples

---

聚类分析主要是探索性分析，因此很难有效地评估结果的好坏，如果有测试集，可以对其分析来评价效果

对于 k-means 算法，寻找新质心点的标准是，最小化每个数据点到最近质心点的距离。这叫作算法的惯性权重（inertia），任何经过训练的 KMeans 实例都有该属性

下面将 n_clusters 依次取 2 到 20 之间的值，每取一个值，k-means 算法运行 10 次。每次运行算法都记录惯性权重。

Input:

```python
    # 惯性权重，这个值没有意义，但是可以用来确定 n_clusters
    print(pipeline.named_steps["clusterer"].inertia_)
    print()
    inertia_scores = []
    n_clusters_values = list(range(2, 20))
    for n_clusters in n_clusters_values:
        # 当前的惯性权重组
        cur_inertia_scores = []
        X = TfidfVectorizer(max_df=0.4).fit_transform(documents)
        for i in range(10):
            km = KMeans(n_clusters=n_clusters).fit(X)
            cur_inertia_scores.append(km.inertia_)
        inertia_scores.append(cur_inertia_scores)
        print("{} : {}".format(n_clusters, np.mean(cur_inertia_scores)))
```

Output:

    291.45747555507467

    2 : 310.72961350285766
    3 : 305.7904332223444
    4 : 302.18859768191396
    5 : 300.28785590112705
    6 : 297.48005120447067
    7 : 294.226862724111
    8 : 292.340968109182
    9 : 291.18707107605024
    10 : 289.46981977256536
    11 : 287.9333326469133
    12 : 285.0561596766078
    13 : 284.33745019948356
    14 : 282.71178879028537
    15 : 280.94991762471807
    16 : 279.9555799316599
    17 : 278.3825941905214
    18 : 274.94616060558434
    19 : 275.0297854253871

---

将上表作图

Input:

```python
    import plotly
    data = plotly.graph_objs.Scatter(
        x=list(range(18)),
        y=[
            310.73,
            305.79,
            302.18,
            300.28,
            297.48,
            294.22,
            292.34,
            291.18,
            289.46,
            287.93,
            285.05,
            284.33,
            282.71,
            280.94,
            279.95,
            278.38,
            274.94,
            275.02,
        ],
    )
    fig = plotly.graph_objs.Figure(data)
    fig.show()
```

Output:

![10.1](/img/in-post/data-mining/ch10/10.1.png)

---

根据上图可以发现在 n_clusters=9 和 15 时拐点比较明显，这里为了方便计算，我们按照书上选择 6

Input:

```python
    # 设置 n_clusters 值为 6，重新运行算法
    n_clusters = 6
    pipeline = Pipeline(
        [
            ("feature_extraction", TfidfVectorizer(max_df=0.4)),
            ("clusterer", KMeans(n_clusters=n_clusters)),
        ]
    )
    pipeline.fit(documents)
    labels = pipeline.predict(documents)
    # 获取特征的所对应的词
    terms = pipeline.named_steps["feature_extraction"].get_feature_names()
    # 统计 6 个簇中每个簇的元素个数
    c = Counter(labels)
    for cluster_number in range(n_clusters):
        print(
            "Cluster {} contains {} samples".format(cluster_number, c[cluster_number])
        )
        print(" Most important terms")
        centroid = pipeline.named_steps["clusterer"].cluster_centers_[cluster_number]
        most_important = centroid.argsort()
        for i in range(5):
            # 排列是非降序排列
            term_index = most_important[-(i + 1)]
            # 输出序号，词语，得分
            print(
                " {0} {1} (score: {2:.4f})".format(
                    i + 1, terms[term_index], centroid[term_index]
                )
            )
```

Output:

    Cluster 0 contains 15 samples
    Most important terms
    1 games (score: 0.2351)
    2 olympic (score: 0.1921)
    3 athletes (score: 0.1555)
    4 ioc (score: 0.1383)
    5 tokyo (score: 0.1365)
    Cluster 1 contains 48 samples
    Most important terms
    1 she (score: 0.0442)
    2 her (score: 0.0409)
    3 masks (score: 0.0381)
    4 monday (score: 0.0298)
    5 23 (score: 0.0294)
    Cluster 2 contains 150 samples
    Most important terms
    1 you (score: 0.0342)
    2 measures (score: 0.0246)
    3 would (score: 0.0246)
    4 country (score: 0.0233)
    5 our (score: 0.0231)
    Cluster 3 contains 14 samples
    Most important terms
    1 your (score: 0.1922)
    2 you (score: 0.1833)
    3 robot (score: 0.1644)
    4 unusual (score: 0.1505)
    5 box (score: 0.1478)
    Cluster 4 contains 128 samples
    Most important terms
    1 india (score: 0.0222)
    2 et (score: 0.0189)
    3 tablet (score: 0.0156)
    4 app (score: 0.0140)
    5 2020 (score: 0.0129)
    Cluster 5 contains 10 samples
    Most important terms
    1 cache (score: 0.2858)
    2 found (score: 0.2672)
    3 server (score: 0.2484)
    4 error (score: 0.2358)
    5 mod_security (score: 0.1450)

---

上面代码在流水线最后一步的 k-means 实例上调用转换方法。得到的矩阵有六个特征，数据量跟文档的长度相同，shape=(365,6)

Input:

```python
    # 用 K-means 算法转化特征
    X = pipeline.transform(documents)
```

### 聚类融合

聚类算法也可以进行融合，这样做的主要原因是，融合后得到的算法能够平滑算法多次运行所得到的不同结果。多次运行 k-means 算法得到的结果因最初选择的质心点不同而不同。多次运行算法，综合考虑所得到的多个结果，可以减少波动。聚类融合方法还可以降低参数选择对最终结果的影响。大多数聚类算法对参数选择很敏感，参数稍有不同将带来不同的聚类结果

最基本的融合方法是对数据进行多次聚类，每次都记录各个数据点的簇标签。然后计算每两个数据点被分到同一个簇的次数。这就是*证据累积*算法（Evidence Accumulation Clustering，EAC）的精髓

- 第一步，使用 k-means 等低水平的聚类算法对数据集进行多次聚类，记录每一次迭代两个数据点出现在同一簇的频率，将结果保存到共协矩阵（coassociation）中
- 第二步，使用另外一种聚类算法——分级聚类对第一步得到的共协矩阵进行聚类分析。分级聚类一个比较有趣的特性是，它等价于寻找一棵把所有节点连接到一起的树，并把权重低的边去掉。

Input:

```python
    # 遍历所有标签，记录具有相同标签的两个数据点的位置，创建共协矩阵
    def create_coassociation_matrix(labels):
        rows = []
        cols = []
        # labels 种类
        unique_labels = set(labels)
        for label in unique_labels:
            # 找出 label 值相同的数据点
            indices = np.where(labels == label)[0]
            # 记录他们的位置：如 1、3 点的数据均为 1，即 1 和 1 相同，1 和 3 相同，3 和 1 相同，3 和 3 相同
            # 行和列均增加了 4 个 indices*indices 个数字
            for index1 in indices:
                for index2 in indices:
                    rows.append(index1)
                    cols.append(index2)
        # 返回给定 shape 和 type 的值全为 1 的矩阵
        data = np.ones((len(rows),))
        # 创建稀疏矩阵满足：a[rows[k], cols[k]] = data[k]
        return csr_matrix((data, (rows, cols)), dtype="float")

    # 使用标签生成共协矩阵
    C = create_coassociation_matrix(labels)
    # 这里书上说多输入几次 C 看看结果，我没有用 notebook，但是使用 print 输出是一样的，因此没有搞懂书上的含义
    print(C)
    print((365 ** 2 - create_coassociation_matrix(labels).nnz) / 365 ** 2)

    mst = minimum_spanning_tree(C)
    mst = minimum_spanning_tree(-C)

    pipeline.fit(documents)
    labels2 = pipeline.predict(documents)
    C2 = create_coassociation_matrix(labels2)
    C_sum = (C + C2) / 2
    mst = minimum_spanning_tree(-C_sum)
    # 删除低于阈值的边
    mst.data[mst.data > -1] = 0
    number_of_clusters, labels = connected_components(mst)
```

Output:

    (0, 0)	1.0
    (0, 1)	1.0
    (0, 2)	1.0
    (0, 3)	1.0
    (0, 4)	1.0
    (0, 5)	1.0
    (0, 6)	1.0
    (0, 7)	1.0
    (0, 8)	1.0
    (0, 9)	1.0
    (0, 10)	1.0
    (0, 11)	1.0
    (0, 12)	1.0
    (0, 13)	1.0
    :	:
    (364, 350)	1.0
    (364, 351)	1.0
    (364, 352)	1.0
    (364, 353)	1.0
    (364, 354)	1.0
    (364, 355)	1.0
    (364, 356)	1.0
    (364, 357)	1.0
    (364, 358)	1.0
    (364, 359)	1.0
    (364, 360)	1.0
    (364, 361)	1.0
    (364, 362)	1.0
    (364, 363)	1.0
    (364, 364)	1.0
    0.11092512666541565

---

从图的理论角度看，生成树为所有节点都连接到一起的图。_最小生成树_（Minimum Spanning Tree，MST）即总权重最低的生成树。结合我们的应用来讲，图中的节点对应数据集中的个体，边的权重对应两个顶点被分到同一簇的次数——也就是共协矩阵所记录的值。

矩阵 C 中，值越高表示一组数据点被分到同一簇的次数越多——这个值表示相似度。相反，minimum_spanning_tree 函数的输入为距离，高的值反而表示相似度越小。这里又用到了一次取反

Input:

```python
    mst = minimum_spanning_tree(C)
    # 对 C 取反再计算最小生成树
    mst = minimum_spanning_tree(-C)
    # 创建额外的标签
    pipeline.fit(documents)
    labels2 = pipeline.predict(documents)
    C2 = create_coassociation_matrix(labels2)
    C_sum = (C + C2) / 2
    # 生成阈值不全为 1 和 0 的最小生成树
    mst = minimum_spanning_tree(-C_sum)
    # 删除低于阈值的边
    mst.data[mst.data > -1] = 0
    number_of_clusters, labels = connected_components(mst)
    print(number_of_clusters)
    print(labels)
```

Output:

    2
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0, 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0, 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

---

k-means 算法不考虑特征的权重，它寻找的是圆形簇（circular clusters）

证据累积算法的工作原理为重新把特征映射到新空间，每次运行 k-means 算法都相当于使用转换器对特征进行一次转换。

证据累积算法只关心数据点之间的距离而不是它们在原来特征空间的位置。对于没有规范化过的特征，仍然存在问题。因此，特征规范很重要，无论如何都要做（我们用 tf-idf 规范特征值，从而使特征具有相同的值域）

Input:

```python
    # 创建证据累积算法类
    class EAC(BaseEstimator, ClusterMixin):
        def __init__(
            self, n_clusterings=10, cut_threshold=0.5, n_clusters_range=(3, 10)
        ):
            self.n_clusterings = n_clusterings  # k-means 算法运行次数
            self.cut_threshold = cut_threshold  # 用来删除边的阈值
            self.n_clusters_range = n_clusters_range  # 每次运行 k-means 算法要找到的簇的数量

        def fit(self, X, y=None):
            # 进行指定次数的共协矩阵累加
            C = sum(
                (
                    create_coassociation_matrix(self._single_clustering(X))
                    for _ in range(self.n_clusterings)
                )
            )
            mst = minimum_spanning_tree(-C)
            mst.data[mst.data > -self.cut_threshold] = 0
            self.n_components, self.labels_ = connected_components(mst)
            return self

        # 进行一次集群
        def _single_clustering(self, X):
            # 在给定范围中随机选择一个集群数
            n_clusters = np.random.randint(*self.n_clusters_range)
            km = KMeans(n_clusters=n_clusters)
            # 返回由 k-means 计算得到的簇标签
            return km.fit_predict(X)

    pipeline = Pipeline(
        [("feature_extraction", TfidfVectorizer(max_df=0.4)), ("clusterer", EAC())]
    )
    pipeline.fit(documents)
    number_of_clusters, labels = (
        pipeline["clusterer"].n_components,
        pipeline["clusterer"].labels_,
    )
    print(number_of_clusters)
    print(labels)

```

Output:

    1
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

总感觉有什么问题。。。 `(￣ε(#￣)☆╰╮o(￣皿￣///))`

### 线上学习

线上学习是指用新数据增量地改进模型。支持线上学习的算法可以先用一条或少量数据进行训练，随着更多新数据的添加，更新模型。

线上学习与流式学习（streaming-based learning）有关，但有几个重要的不同点。线上学习能够重新评估先前创建模型时所用到的数据，而对于后者，所有数据都只使用一次。

scikit-learn 提供了 MiniBatchKMeans 算法，可以用它来实现线上学习功能。这个类实现了 `partial_fit` 函数，接收一组数据，更新模型。调用`fit()`将会删除之前的训练结果，重新根据新数据进行训练。

Input:

```python
    # 使用 TfIDFVectorizer 从数据集中抽取特征，创建矩阵 X
    n_clusters = 6
    vec = TfidfVectorizer(max_df=0.4)
    X = vec.fit_transform(documents)
    mbkm = MiniBatchKMeans(random_state=14, n_clusters=3)
    batch_size = 10
    # 随机从 X 矩阵中选择数据，模拟来自外部的新数据
    for iteration in range(int(X.shape[0] / batch_size)):
        start = batch_size * iteration
        end = batch_size * (iteration + 1)
        mbkm.partial_fit(X[start:end])
    # 获取数据集聚类结果
    labels = mbkm.predict(X)
    c = Counter(labels)
    for cluster_number in range(n_clusters):
        print(
            "Cluster {} contains {} samples".format(cluster_number, c[cluster_number])
        )
```

Output:

    Cluster 0 contains 2 samples
    Cluster 1 contains 362 samples
    Cluster 2 contains 1 samples
    Cluster 3 contains 0 samples
    Cluster 4 contains 0 samples
    Cluster 5 contains 0 samples

---

由于 TfIDFVectorizer 不是在线算法，因此无法在流水线中使用

为了解决这个问题，我们使用 HashingVectorizer 类，它巧妙地使用散列算法极大地降低了计算词袋模型所需的内存开销，将数据的内容转换成散列值

Input:

```python
    class PartialFitPipeline(Pipeline):
        def partial_fit(self, X, y=None):
            Xt = X
            # 经过最后一步之前的所有步转换
            for name, transform in self.steps[:-1]:
                Xt = transform.transform(Xt)
            # 调用 MiniBatchKMeans 的 partial_fit 函数
            return self.steps[-1][1].partial_fit(Xt, y=y)

    pipeline = PartialFitPipeline(
        [
            ("feature_extraction", HashingVectorizer()),
            ("clusterer", MiniBatchKMeans(random_state=14, n_clusters=3)),
        ]
    )
    batch_size = 10
    for iteration in range(int(len(documents) / batch_size)):
        start = batch_size * iteration
        end = batch_size * (iteration + 1)
        pipeline.partial_fit(documents[start:end])
    labels = pipeline.predict(documents)
    c = Counter(labels)
    for cluster_number in range(n_clusters):
        print(
            "Cluster {} contains {} samples".format(cluster_number, c[cluster_number])
        )
```

Output:

    Cluster 0 contains 4 samples
    Cluster 1 contains 76 samples
    Cluster 2 contains 285 samples
    Cluster 3 contains 0 samples
    Cluster 4 contains 0 samples
    Cluster 5 contains 0 samples

这一章的内容比较多，也学了挺久，虽然中间结果跟书上的差的有点多。。可能是因为最近新冠肺炎吧 (￣\_,￣ )

## 第十一章

本章介绍如何使用深度神经网络识别图像中的物体

### 深度神经网络

深度神经网络和第 8 章中的基本神经网络的差别在于规模大小。至少包含两层隐含层的神经网络被称为深度神经网络。神经网络的核心其实就是一系列矩阵运算，两个网络之间连接的权重可以用矩阵来表示。其中行表示前一层神经元，列表示后一层神经元，一个神经网络就可以用一组这样的矩阵来表示。除了神经元外，每层增加一个偏置项，它是一个特殊的神经元，永远处于激活状态，并且跟下一层的每一个神经元都有连接。

神经网络使用卷积层（一般来说，仅卷积神经网络包含该层）和池化层（pooling layer），池化层接收某个区域最大输出值，可以降低图像中的微小变动带来的噪音，减少（down-sample，降采样）信息量，这样后续各层所需工作量也会相应减少。

使用 Iris 数据集进行对比实验

Input:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt


iris = load_iris()
X = iris.data.astype(np.float32)
y_true = iris.target.astype(np.int32)

# 预处理数据集
y_onehot = OneHotEncoder().fit_transform(y_true.reshape(-1, 1))
y_onehot = y_onehot.astype(np.int64).todense()

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, random_state=14)
input_layer_size, hidden_layer_size, output_layer_size = 4, 6, 3
# 隐含层
hidden_layer = Dense(output_dim=hidden_layer_size, input_dim=input_layer_size, activation='relu')
# 输出层
output_layer = Dense(output_layer_size, activation='sigmoid')
# 创建顺序模型
model = Sequential(layers=[hidden_layer, output_layer])
# 为训练神经网络配置模型
# 损失函数设置为均方误差，优化器设置为 adam(亚当) 即遵循原始文件中的默认参数，指定精度衡量标准
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# 当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次 epoch
# 为模型训练固定的 epoch（数据集上的迭代）
# 输出模式。0 不输出，1 每个 epoch 一个进度条，2 一行每个 epoch。
history = model.fit(X_train, y_train, nb_epoch=100, verbose=2)
# 记录了连续几个 epoch 的训练损失值和度量值，以及验证损失值和验证度量值 (如果适用的话)
history.history
# 作图，绘制出 epoch 和 loss 关系图
plt.figure(figsize=(10, 10))
plt.plot(history.epoch, history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 为输入样本生成输出预测，计算是分批进行的
# 返回的是数值 [0.9356668, 0.20588416, 0.00021186471],代表样本属于每个类别的概率
y_pred = model.predict(X_test)
# 返回一串预测结果，样本属于哪一个类别
y_pred = model.predict_classes(X_test)
y_pred = model.predict_classes(X_test)
print(classification_report(y_true=y_test.argmax(axis=1), y_pred=y_pred))
```

Output:

![iris_1](/img/in-post/data-mining/ch11/iris_100_epoch.png)

                  precision    recall  f1-score   support

               0       1.00      1.00      1.00        17
               1       1.00      0.08      0.14        13
               2       0.40      1.00      0.57         8

        accuracy                           0.68        38
       macro avg       0.80      0.69      0.57        38
    weighted avg       0.87      0.68      0.62        38

---

重复上面的操作，这次运行 1000 步，对比实验结果

Input:

```python
hidden_layer = Dense(output_dim=hidden_layer_size, input_dim=input_layer_size, activation='relu')
output_layer = Dense(output_layer_size, activation='sigmoid')
model = Sequential(layers=[hidden_layer, output_layer])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, nb_epoch=1000, verbose=False)
plt.figure(figsize=(12, 8))
plt.plot(history.epoch, history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show("keras_on_iris_2.png")
y_pred = model.predict_classes(X_test)
print(classification_report(y_true=y_test.argmax(axis=1), y_pred=y_pred))
```

Output:

![iris_2](/img/in-post/data-mining/ch11/iris_1000_epoch.png)

                  precision    recall  f1-score   support

               0       1.00      1.00      1.00        17
               1       1.00      1.00      1.00        13
               2       1.00      1.00      1.00         8

        accuracy                           1.00        38
       macro avg       1.00      1.00      1.00        38
    weighted avg       1.00      1.00      1.00        38

从结果可以看出，经过 100 步训练的神经网络正确率达到了 68%，经过 1000 步训练后正确率达到了 100%

---

验证码识别实验

Input:

```python
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from skimage.measure import label, regionprops


def create_captcha(text, shear=0, size=(100, 30), scale=1):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(
        "/home/saltfish/Programming/Python/data_mining/ch11/FiraCode-Medium.otf", 22
    )
    draw.text((0, 0), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    image = image / image.max()
    shape = image.shape
    # Apply scale
    shapex, shapey = (shape[0] * scale, shape[1] * scale)
    image = tf.resize(image, (shapex, shapey))
    return image

image = create_captcha("FISH", shear=0.5, scale=0.6)
plt.imshow(image, cmap="Greys")
```

Output:

![captcha_1](/img/in-post/data-mining/ch11/captcha_1.png)

---

```python
def segment_image(image):
    # 标记找到连通的非黑色像素的子图像
    labeled_image = label(image > 0.2, connectivity=1, background=0)
    subimages = []
    # 拆分子图
    for region in regionprops(labeled_image):
        # 提取子图
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])
    if len(subimages) == 0:
        # 未找到子图，返回这个图片本身
        return [
            image,
        ]
    return subimages

subimages = segment_image(image)
# 选出四张小图片
f, axes = plt.subplots(1, len(subimages), figsize=(10, 3))
for i in range(len(subimages)):
    axes[i].imshow(subimages[i], cmap="gray")
plt.show()
```

Output:

![captcha_2](/img/in-post/data-mining/ch11/captcha_2.png)

---

```python
random_state = check_random_state(14)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
assert len(letters) == 26
shear_values = np.arange(0, 0.8, 0.05)
scale_values = np.arange(0.9, 1.1, 0.1)

# 随机生成一个字母的图片
def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    scale = random_state.choice(scale_values)
    return (
        create_captcha(letter, shear=shear, size=(30, 30), scale=scale),
        letters.index(letter),
    )

image, target = generate_sample(random_state)
plt.imshow(image, cmap="Greys")
print("The target for this image is: {0}".format(letters[target]))
```

Output:

    The target for this image is: L

![captcha_3](/img/in-post/data-mining/ch11/captcha_3.png)

---

Input:

```python
# 生成数据集
dataset, targets = zip(*(generate_sample(random_state) for i in range(1000)))
dataset = np.array(
    [tf.resize(segment_image(sample)[0], (20, 20)) for sample in dataset]
)
dataset = np.array(dataset, dtype="float")
targets = np.array(targets)

onehot = OneHotEncoder()
y = onehot.fit_transform(targets.reshape(targets.shape[0], 1))
y = y.todense()
X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

hidden_layer = Dense(100, input_dim=X_train.shape[1])
output_layer = Dense(y_train.shape[1])

model = Sequential(layers=[hidden_layer, output_layer])
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=1000, verbose=False)
y_pred = model.predict(X_test)
print(f1_score(y_pred=y_pred.argmax(axis=1), y_true=y_test.argmax(axis=1), average="macro"))
print(classification_report(y_pred=y_pred.argmax(axis=1), y_true=y_test.argmax(axis=1)))
```

Output:

    1.0

                  precision    recall  f1-score   support

               0       1.00      1.00      1.00         4
               1       1.00      1.00      1.00         4
               2       1.00      1.00      1.00         3
               3       1.00      1.00      1.00        10
               4       1.00      1.00      1.00         3
               5       1.00      1.00      1.00         3
               6       1.00      1.00      1.00         1
               7       1.00      1.00      1.00         3
               8       1.00      1.00      1.00         5
               9       1.00      1.00      1.00         3
              10       1.00      1.00      1.00         3
              11       1.00      1.00      1.00         6
              12       1.00      1.00      1.00         3
              13       1.00      1.00      1.00         5
              14       1.00      1.00      1.00         4
              15       1.00      1.00      1.00         6
              16       1.00      1.00      1.00         1
              17       1.00      1.00      1.00         3
              18       1.00      1.00      1.00         2
              19       1.00      1.00      1.00         3
              20       1.00      1.00      1.00         5
              21       1.00      1.00      1.00         7
              22       1.00      1.00      1.00         4
              23       1.00      1.00      1.00         2
              24       1.00      1.00      1.00         2
              25       1.00      1.00      1.00         5

        accuracy                           1.00       100
       macro avg       1.00      1.00      1.00       100
    weighted avg       1.00      1.00      1.00       100

### 使用 GPU 优化

为了让我的 GPU 能跑程序，可费了我好大功夫，结果我这 960M 的 2G 内存还跑不了太大的程序/(ㄒ o ㄒ)/~~

第 101 次想念我的台式机，可恶的病毒

配置的过程跟 [TensorFlow](https://tensorflow.google.cn/install/gpu) 官网给的方法没啥区别，在这就不多说了（官网给出的 NVIDIA 显卡驱动版本是 430，我这里是 440，CUDA 版本是 10.2，依然能运行程序，可能只需要 `development and runtime libraries` 正确安装就行？）

使用 tensorflow 在执行 `modle.compile()` 的时候需要较长的时间，运行时的速度还是很快的

初次接触神经网络，不了解的东西太多了，还是先多做几个训练再说吧。。

### 应用

书上使用 CIFAR 图像数据集的代码太老了（原谅我太菜了解决不了依赖问题），因此我跟着 Tensorflow 官网的代码做完了这个实验

服装识别

Input:

```python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # --------加载、了解、预处理数据集--------
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    # 查看数据集
    print(
        train_images.shape,  # (60000，28，28)
        len(train_labels),  # 60000
        train_labels,  # [9 0 0 ... 3 0 5]
        test_images.shape,  # (10000, 28, 28)
        len(test_labels),  # 10000
    )
```

Output:

    (60000, 28, 28) 60000 [9 0 0 ... 3 0 5] (10000, 28, 28) 10000

---

Input:

```python
    # 查看图像
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    # 预处理标准化
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 查看数据集
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
```

Output:

<table align="left">
  <td align="center">
    <img src="/img/in-post/data-mining/ch11/tensorflow11.1.png"
        alt="Fashion MNIST sprite"  width="600">
  </td>
  <td align="center">
    <img src="/img/in-post/data-mining/ch11/tensorflow11.2.png"
        alt="Fashion MNIST sprite"  width="600">
  </td>
</table>

---

Input:

```python
    # --------建立模型--------
    # 建立神经网络所需要模型的各层
    # tf.keras.layers.Flatten 将图像的格式从二维数组 (28 * 28) 转换为一维数组 (28 * 28 = 784)
    # 可以将这个图层看作是图像中取消堆叠的像素行，并将它们排列起来
    # 这个层没有参数需要学习; 它只是重新格式化数据。
    #
    # 然后是两个稠密层（完全连接的层），中间一层有 128 个节点，
    # 最后一层返回长度为 10 的对数数组。每个神经元包含一个得分，指示当前图像对这一类的评分
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10),
        ]
    )
    # --------编译模型--------
    # 损失函数：这可以衡量训练期间模型的准确程度，希望最小化这个函数，以便将模型“引导”到正确的方向
    # 优化器：如何基于它看到的数据和它的损失函数更新模型
    # 指标：用于检测训练和测试步骤。下面的例子使用精确度，即正确分类的图像的分数
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # --------训练模型--------
    model.fit(train_images, train_labels, epochs=10)
    # --------评估表现--------
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("\nTest accuracy:", test_acc)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # prediction 是由 10 个数字组成的数组。它们表示模型对图像对应于 10 种不同衣服各自的置信度
    predictions = probability_model.predict(test_images)
    print(predictions[0])
```

Output:

    pciBusID: 0000:02:00.0 name: GeForce GTX 960M computeCapability: 5.0
    coreClock: 1.176GHz coreCount: 5 deviceMemorySize: 1.96GiB deviceMemoryBandwidth: 74.65GiB/s
    Epoch 1/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.4919 - accuracy: 0.8271
    Epoch 2/10
    1875/1875 [==============================] - 3s 1ms/step - loss: 0.3758 - accuracy: 0.8648
    Epoch 3/10
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.3346 - accuracy: 0.8770
    Epoch 4/10
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.3099 - accuracy: 0.8860
    Epoch 5/10
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.2927 - accuracy: 0.8927
    Epoch 6/10
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.2807 - accuracy: 0.8962
    Epoch 7/10
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.2655 - accuracy: 0.9010
    Epoch 8/10
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.2548 - accuracy: 0.9044
    Epoch 9/10
    1875/1875 [==============================] - 3s 1ms/step - loss: 0.2440 - accuracy: 0.9095
    Epoch 10/10
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.2373 - accuracy: 0.9113
    313/313 - 0s - loss: 0.3479 - accuracy: 0.8798

    Test accuracy: 0.879800021648407

    [1.3496768e-07 1.5826453e-10 1.7375668e-09 2.1999605e-10 5.5648923e-07
    1.9829762e-03 1.9957926e-07 1.8424643e-04 9.3086570e-09 9.9783188e-01]

从输出可以看出 loss 函数正在逐渐减小，训练的准确率在不断的增加，这正是我们所要的

在训练集中的准确率为 91.1%，而在测试集中只有 88%，这是出现了过拟合 (overfitting)，关于过拟合的证明和避免过拟合的方法，等过几天单独写一个 post 学习一下

---

定义两个函数用来绘图，更直观地看出预测结果

Input:

```python
    # --------验证模型--------
    # 制作图表来观察十个类别预测的完整集合
    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = "blue"
        else:
            color = "red"
        # 置信度百分比
        plt.xlabel(
            "{} {:2.0f}% ({})".format(
                class_names[predicted_label],
                100 * np.max(predictions_array),
                class_names[true_label],
            ),
            color=color,
        )

    # 绘制置信度柱状图
    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        # 错误的预测标签为红色
        thisplot[predicted_label].set_color("red")
        # 正确的标签为蓝色
        thisplot[true_label].set_color("blue")

    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    i = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()
```

Output:

<table class="tfo-notebook-buttons" align="center">
  <tr align="center">
    <img src="/img/in-post/data-mining/ch11/tensorflow11.3.png" />
  </tr>
  <tr align="center">
    <img src="/img/in-post/data-mining/ch11/tensorflow11.4.png" />
  </tr>
  <tr align="center">
    <img src="/img/in-post/data-mining/ch11/tensorflow11.5.png" />
  </tr>
</table>

---

示范如何使用模型来得到预测结果

Input:

```python
    # --------使用模型--------
    img = test_images[1]
    print(img.shape)
    # 转换成 keras 支持的格式
    img = np.expand_dims(img, 0)
    print(img.shape)
    # 为该图像预测
    predictions_single = probability_model.predict(img)
    print(predictions_single)
    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()
    # 返回预测的种类
    print("result: ", np.argmax(predictions_single[0]))
```

Output:

    (28, 28)
    (1, 28, 28)
    [[3.3822223e-05 3.9569712e-13 9.9579656e-01 1.2699689e-10 3.9967773e-03
    1.0960948e-12 1.7281482e-04 5.0896191e-17 7.9589724e-11 1.4832706e-12]]
    result: 2

    # 这张图片忘了保存了

---

试着增加步数观察是否能得到更好的结果

Input:

```python
    # 仅修改这一行代码，其它不变，重新运行
    model.fit(train_images, train_labels, epochs=100, verbose=0)
```

Output:

    313/313 - 0s - loss: 0.8228 - accuracy: 0.8824
    Test accuracy: 0.8823999762535095

    [3.4202448e-25 0.0000000e+00 1.0021874e-24 0.0000000e+00 5.4433387e-31
    1.8526166e-17 5.1352536e-34 4.5777732e-13 1.4344803e-28 1.0000000e+00]

    (28, 28)
    (1, 28, 28)
    [[2.1121348e-16 0.0000000e+00 1.0000000e+00 1.0520916e-32 5.9910987e-09
    1.4628578e-34 1.1571613e-14 0.0000000e+00 6.4996830e-38 0.0000000e+00]]
    result: 2

<table class="tfo-notebook-buttons" align="center">
  <tr align="center">
    <img src="/img/in-post/data-mining/ch11/tensorflow11.6.png" />
  </tr>
  <tr align="center">
    <img src="/img/in-post/data-mining/ch11/tensorflow11.7.png" />
  </tr>
  <tr align="center">
    <img src="/img/in-post/data-mining/ch11/tensorflow11.8.png" />
  </tr>
</table>

在增加到 100 步后，最后一步的输出为

    Epoch 100/100
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.0570 - accuracy: 0.9790

在训练集上的精确度达到了 97.9%，而在测试集中也只达到了 88.2%，有微小的进步

这个实验条理清晰地展示了深度学习的基本步骤：

- 加载、了解、预处理数据集
- 建立模型
- 建立模型
- 训练模型
- 评估表现
- 验证模型
- 使用模型做预测

---

这本书就快看完了，正愁不知道下本书看啥的我又发现了一个学习宝库`TensorFlow`。就决定是你了！

笔记本好难用，还是 pycharm 适合我。。。但还是得学用笔记本啊～～

## 第十二章

本章主要介绍了 python 使用 MapReduce 来进行大数据处理

### MapReduce 例子

MapReduce 主要分为两步：映射（Map）和规约（Reduce）

Input:

```python
# -*- coding: utf-8 -*-
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups

from joblib import Parallel, delayed

import timeit


# 计算 documents 中的单词出现词素
def map_word_count(document_id, document):
    counts = defaultdict(int)
    for word in document.split():
        counts[word] += 1
    for word in counts:
        yield word, counts[word]


# 将 map 得到的结果，即每篇文章中单词出现的次数整合起来
# 如文章 1 中单词"apple"出现了 2 次，文章 2 中单词"apple"出现了 5 次，则返回结果为 ["apple":[2,5],...]
def shuffle_words(results_generators):
    records = defaultdict(list)
    # 遍历每一篇文章
    for results in results_generators:
        # 遍历每个单词
        for word, count in results:
            records[word].append(count)
    # 每次生成一个单词
    for word in records:
        yield word, records[word]


# 将单词所对应的列表叠加起来得到单词出现次数
def reduce_counts(word, list_of_counts):
    return word, sum(list_of_counts)


if __name__ == "__main__":
    dataset = fetch_20newsgroups(subset="train")
    documents = dataset.data
    start = timeit.default_timer()
    # 生成器，输出 (单词，出现次数的键值对)
    map_results = map(map_word_count, range(len(documents)), documents)
    shuffle_results = shuffle_words(map_results)
    reduce_results = [
        reduce_counts(word, list_of_counts) for word, list_of_counts in shuffle_results
    ]
    end = timeit.default_timer()
    print(reduce_results[:5])
    print(len(reduce_results))
    print("----------", str(end - start))
```

Output:

    pydev debugger: process 7540 is connecting
    [('From:', 11536), ('lerxst@wam.umd.edu', 2), ("(where's", 3), ('my', 7679), ('thing)', 9)]
    280308
    ---------- 4.087287616999674

---

接下来导入 joblib 库，将 map 工作分配出去，使用 4 个进程进行计算

Input:

```python
    def map_word_count(document_id, document):
        counts = defaultdict(int)
        for word in document.split():
            counts[word] += 1
        return list(counts.items())

    start = timeit.default_timer()

    map_results = Parallel(n_jobs=4)(
        delayed(map_word_count)(i, document) for i, document in enumerate(documents)
    )

    shuffle_results = shuffle_words(map_results)
    reduce_results = [
        reduce_counts(word, list_of_counts) for word, list_of_counts in shuffle_results
    ]

    end = timeit.default_timer()

    print(reduce_results[:5])
    print(len(reduce_results))
    print("----------", str(end - start))
```

Output:

    pydev debugger: process 7566 is connecting
    pydev debugger: process 7556 is connecting
    pydev debugger: process 7552 is connecting
    pydev debugger: process 7561 is connecting
    [('From:', 11536), ('lerxst@wam.umd.edu', 2), ("(where's", 3), ('my', 7679), ('thing)', 9)]
    280308
    ---------- 3.5958340090001

可以看到运行时间确实减少了（数据集太少了效果不怎么样）

### MapReduce 应用

书中使用 blogs 的数据集，有 19320 个人的 blog 信息

手头上的电脑配置有点不行了，跑的属实费劲，就放在这了（其实是迫不及待想去做做 tensorflow 的练习了嘿嘿）

## 接下来的方向

书中根据每一章的内容，都有更进一步的实践，我会选几个单独做一下练习

Done

## 参考链接

1. [python-3.8.2-doc](https://docs.python.org/zh-cn/3.8/index.html)
2. [pandas-doc](https://pandas.pydata.org/docs/user_guide/index.html)
3. [numpy-doc](https://numpy.org/devdocs/)
4. [scikit-learn](https://scikit-learn.org/stable/index.html)
5. [tensorflow](https://tensorflow.google.cn/)
