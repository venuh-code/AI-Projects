# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import operator as opt

#考虑到不同特征值范围差别很大的影响，可对这类数据进行最大最小值标准化数据集
def normData(dataset):
    maxVals = dataset.max(axis=0) #求列的最大值
    minVals = dataset.min(axis=0) #求列的最小值
    ranges = maxVals - minVals
    retData = (dataset - minVals) / ranges
    return retData, minVals, ranges

#KNN,欧式距离
def kNN(dataset, labels, testdata, k):
    distSquareMat = (dataset - testdata) ** 2 #计算差值的平方
    distSquareSums = distSquareMat.sum(axis=1) #求每一行的差值平方和
    distances = distSquareSums ** 0.5 #开根号，得出每个样本到测试点的距离
    sortedIndices = distances.argsort()  #array.argsort(),默认axis=0从小到大排序，得到排序后的下标位置
    indices = sortedIndices[:k] #取距离最小的k个值对应的小标位置
    labelCount = {} #存储每个label的出现次数
    for i in indices:
        label = labels[i]
        labelCount[label] = labelCount.get(label, 0) + 1 #次数加1,dict.get(k, val)获取字典中k对应的值,没有k,则返回val

    sortedCount = sorted(labelCount.items(), key=opt.itemgetter(1), reverse=True) #operator.itemgetter(),结合sorted使用,可按不同的区域进行排序
    return sortedCount[0][0] #返回最多的一个label

#主函数
if __name__ == "__main__":
    dataSet = np.array([[2, 3], [6, 8]])
    normDataSet, ranges, minVals = normData(dataSet)
    labels = ['a', 'b']
    testData = np.array([[3.9, 5.5], [9, 10]])
    for i in range(len(testData)):
        normTestData = (testData[i] - minVals) / ranges
        result = kNN(normDataSet, labels, normTestData, 1)
        print(result)