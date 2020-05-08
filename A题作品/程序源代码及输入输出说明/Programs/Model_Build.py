#导入库
#导入分类树，可视化库
from sklearn.tree import DecisionTreeClassifier,export_graphviz
#导入分类报告库
from sklearn.metrics import classification_report
#导入可视化库
import graphviz
#导入pandas库
import pandas as pd
#导入numpy
import numpy as np
#导入样本集拆分相关库
from sklearn.model_selection import train_test_split
#导入KNN相关库
from sklearn.neighbors import KNeighborsClassifier
#导入lasso回归相关库
from sklearn.linear_model import Lasso
#导入逻辑回归相关库
from sklearn.linear_model import LogisticRegression
#导入投票相关库
from sklearn.ensemble import VotingClassifier
# 导入模型保存库
import joblib
# 随机重复采样
from imblearn.over_sampling import RandomOverSampler

# Lasso回归验证因子相关性，确保模型得到的数据是恰当的
def Lass():
    inputFile = 'data.csv'
    #读取文件
    data = pd.read_csv(inputFile,encoding = 'GBK')
    #相关系数矩阵
    pearson = np.round(data.corr(method= 'pearson'),2)
    print('----------相关系数矩阵为----------')
    #打印相关系数矩阵
    print(pearson)
    #惩罚力度
    lasso = Lasso(10)   #系数可调
    lasso.fit(data.iloc[:,3:15],data['是否高转送'])
    print('相关系数为：',np.round(lasso.coef_,5))
    #打印出相关系数非0的自变量的个数
    print('相关系数非零个数为：',np.sum(lasso.coef_!= 0))

# 决策树
def Decision():
    print('------------决策树------------')
    # 读取文件
    data1 = pd.read_csv('data.csv', encoding='GBK')
    data = data1.loc[data1["年份（年末）"] != 7]
    # 处理数据不均衡问题
    ros = RandomOverSampler(random_state=0, sampling_strategy=0.3)
    X_resampled, y_resampled = ros.fit_sample(data.iloc[:, 3:15], data['是否高转送'])
    # 拆分专家样本集
    data_tr, data_te, label_tr, label_te = train_test_split(X_resampled, y_resampled)
    #模型构建
    Model = DecisionTreeClassifier(max_depth=20,random_state=8,splitter='random',min_samples_split=3,min_samples_leaf=1,)
    #模型训练
    Model.fit(data_tr, label_tr)

    #模型预测
    dt_pre = Model.predict(data_te)
    print('预测结果为：',dt_pre)
    print('---------模型预测值与真实值比较------------')
    print(dt_pre==label_te)      #比较模型预测值与真实值
    # 分类报告
    dt_reports= classification_report(label_te,dt_pre)
    print('---------分类报告------------')
    #打印分类报告
    print(dt_reports)

    # 决策树可视化
    dot_data = export_graphviz(Model,feature_names=['年份（年末）','交易日平均价','预增或预减','超涨或超跌','次新股','每股资本公积(元/股)+每股未分配利润(元/股)','每股现金流量净额(元/股)','实收资本(或股本)','每股收益(期末摊薄，元/股)','每股净资产(元/股)','营业总收入同必增长(%)','近两年送转比例','上市时间'],class_names='是否高转送')
    #可视化结果保存到“dt.dot”
    #打开dot文件，需要在node属性中添加“fontname = FangSong”，否则会出现乱码
    f = open('dt.dot','w')
    f.write(dot_data)
    f.close()
    graph = graphviz.Source(dot_data)


#KNN预测
def Knn():
    print('------------KNN------------')
    #读取数据
    data1 = pd.read_csv('data.csv', encoding='GBK')
    data = data1.loc[data1["年份（年末）"] != 7]
    # 处理数据不均衡问题
    ros = RandomOverSampler(random_state=0, sampling_strategy=0.3)
    X_resampled, y_resampled = ros.fit_sample(data.iloc[:, 3:15], data['是否高转送'])
    # 拆分专家样本集
    data_tr, data_te, label_tr, label_te = train_test_split(X_resampled, y_resampled)
    #模型构建
    model = KNeighborsClassifier(n_neighbors=4)
    #模型训练
    model.fit(data_tr,label_tr)
    #模型预测
    KN_pre = model.predict(data_te)
    #打印结果
    print('预测结果为：',KN_pre)
    #打印预测精度
    KN_acc = model.score(data_te,label_te)
    print('正确率：',KN_acc)


# 逻辑回归
def Logistic():
    print('------------逻辑回归------------')
    # 读取数据
    data1 = pd.read_csv('data.csv', encoding='GBK')
    data = data1.loc[data1["年份（年末）"] != 7]
    # 处理数据不均衡问题
    ros = RandomOverSampler(random_state=0, sampling_strategy=0.3)
    X_resampled, y_resampled = ros.fit_sample(data.iloc[:, 3:15], data['是否高转送'])
    # 拆分专家样本集
    data_tr, data_te, label_tr, label_te = train_test_split(X_resampled, y_resampled)
    #模型构建
    clf = LogisticRegression(max_iter=1000)
    #模型训练
    clf.fit(data_tr, label_tr)
    #模型预测
    log_pre = clf.predict(data_te)
    #预测结果打印
    print('预测结果为：',log_pre)
    #打印分类报告
    log_res = classification_report(label_te, log_pre)
    print('---------分类报告------------')
    print(log_res)


#硬投票
def Voting():
    #读取文件
    data1 = pd.read_csv('data.csv',encoding = 'GBK')
    data = data1.loc[data1["年份（年末）"]!=7]
    print('------------投票------------')
    #处理数据不均衡问题
    ros = RandomOverSampler(random_state=0,sampling_strategy=0.3)
    X_resampled, y_resampled = ros.fit_sample(data.iloc[:, 3:15], data['是否高转送'])
    #拆分专家样本集
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,test_size=0.15)
    #模型构建
    voting_clf = VotingClassifier(estimators=[
        ('log_pre',LogisticRegression(max_iter=5000)),
        ('KN_pre',KNeighborsClassifier(n_neighbors=4)),
        ('dt_pre',DecisionTreeClassifier(max_depth=22,random_state=8,splitter='random',min_samples_split=3,min_samples_leaf=1,))
    ],voting='hard')
    #模型训练
    voting_model = voting_clf.fit(X_train, y_train)
    #模型在测试集得分
    voting_score = voting_model.score(X_test,y_test)
    #模型预测
    voting_pre = voting_model.predict(X_test)
    print('正确率：',voting_score)
    #模型预测值与真实值比较
    print('---------模型预测值与真实值比较------------')
    # print(voting_clf.predict(data.iloc[:,3:15]) == data['是否高转送'])
    #打印分类报告
    voting_res = classification_report(y_test,voting_pre)
    print(voting_res)
    #训练好的模型保存到本地
    joblib.dump(voting_clf,'model.pkl')
    print('模型保存成功')
    # 预测第八年高送转情况
    print('---------预测第八年数据--------')
    data1 = pd.read_csv('data.csv',encoding = 'GBK')
    data2 = data1.loc[data1["年份（年末）"]==7]
    data3 = data2.iloc[:, 3:15]
    eight_pre = voting_model.predict(data3)
    number = data2['股票编号'].values
    dataframe = pd.DataFrame({'股票编号':number,'是否高转送':eight_pre})
    dataframe.to_csv("eight_pre.csv", index=False, encoding='GBK')
    print("第八年预测数据保存成功！")
#模型调用并输出预测结果到csv文件
def Model_load():
    #加载训练好的模型
    clf = joblib.load('model.pkl')
    print('模型加载成功')
    #打开预处理过的测试数据‘pre_test_data.csv’
    data = pd.read_csv('pre_test_data.csv', encoding='GBK')
    #预测结果
    pre = clf.predict(data)
    #打印预测结果
    print(pre)
    #结果导出到“pre.csv”
    dataframe = pd.DataFrame(data=pre, columns=['是否高转送'])
    dataframe.to_csv("pre.csv", index=False, encoding='GBK',)




