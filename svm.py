from dataset import Dataset
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    ## SVM分类
    datatypes_select = ["n01514668","n01531178","n01534433","n01622779","n01664065","n01682714","n01694178","n01748264","n01774384","n01824575"]
    labelmap= {"n01514668":0,"n01531178":1,"n01534433":2,"n01622779":3,"n01664065":4,"n01682714":5,"n01694178":6,"n01748264":7,"n01774384":8,"n01824575":9}
    data = Dataset(r"imagenet_mini",dataset='SVM',datatypes_select=datatypes_select,labelmap=labelmap)
    data.load()
    data.get_labels()
    train_data, train_labels, test_data, test_labels = data.get_data()
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    sc = StandardScaler()               # 定义一个标准缩放器
    sc.fit(train_data)                  # 计算均值、标准差
    X_train_std = sc.transform(train_data) # 使用计算出的均值和标准差进行标准化
    X_test_std  = sc.transform(test_data)  # 使用计算出的均值和标准差进行标准化



    svm = SVC(C=0.1,kernel='linear')                         # 定义线性支持向量分类器 (linear为线性核函数)
    svm.fit(X_train_std, train_labels)       # 根据给定的训练数据拟合训练SVM模型
    # 使用测试集进行数据预测

    Y_pred = svm.predict(X_test_std) # 用训练好的分类器svm预测数据X_test_std的标签
    print('Misclassified samples: %d' % (test_labels != Y_pred).sum())   # 输出错误分类的样本数
    print('Accuracy: %.2f' % svm.score(X_test_std, test_labels)) 