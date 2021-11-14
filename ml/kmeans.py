import numpy as np
import sklearn.datasets             #加载原数据
import matplotlib.pyplot as plt
import random

#点到各点距离
def PointToData(point,dataset):
    #print('point.shape, datase.shape', point.shape, dataset.shape)
    a = np.multiply(dataset - point,dataset - point)
    #print('a',a)
    distence = np.sqrt(a[:,0]+a[:,1])
    #print('distence', distence)
    return distence
   
 #选择初始的k个中心簇
def startpoint(k,dataset):
    m, n = np.shape(dataset)
    index1 = random.randint(0,len(dataset) - 1)
    A = []  # 初始的k个中心簇
    A_dit = []  # 初始所有点到中心簇的距离
    A.append(dataset[index1])
    #print('A=========', A)
    sum_dis = np.zeros((m, 1))
    flag_mat = np.ones((m,1))
    flag_mat[index1] = 0
    for i in range(0, k - 1):
        #print(i,'A[i]', A[i])
        A_dit.append((PointToData(A[i], dataset)).reshape(-1,1) )
        #print('A_dit[{}]:{}'.format(i,A_dit[i]))
        sum_dis =(sum_dis  + A_dit[i]) * flag_mat
        #print('sum_dis[{}]:{}'.format(i,sum_dis))
        Index = np.argmax(sum_dis)
        flag_mat[Index] = 0
        print('选的Index：',Index)
        A.append(dataset[Index])
    return A
    
#加载数据
print("===")
Data = sklearn.datasets.load_iris()
dataset = Data.data[:,0:2]

test = dataset[0:15,:]
testm,testn = np.shape(test)
print(test)

#初始点测试函数
k = 4
Apoint = startpoint(k,test)
print('Apoint',Apoint)


def classfy(dataset,Apoint):
    m,n = np.shape(dataset)
    dis_li = []
    num = 0
    for point in Apoint:
        distence = PointToData(point,dataset)
        dis_li.append(distence)
        if num == 0:
            dis_li_mat = dis_li[num]
        else:
            dis_li_mat = np.column_stack((dis_li_mat,dis_li[num]))
        num += 1
    result = np.argmin(dis_li_mat,axis=1)
    #print('dis_li:',dis_li)
    #print('dis_li_mat:\n', dis_li_mat)
    #print('classfy:',result)
    return result
label2 = classfy(test,Apoint)
print('label2:',label2)


#求分类的新中心
def Center(dataset,label,k):
    i = 0
    newpoint = []
    for index in range(k):
        flag = (label==index)
        #print('flag,i:',flag,i)
        num = sum(flag)
        #print('num:',num,index)
        a = flag.reshape(-1,1) * dataset
        #print('a', a)
        newpoint.append(np.sum(a,axis = 0)/num)
        i += 1
        #print(newpoint)
    return newpoint
print("==== get new cneter ====")
testcenter = Center(test,label2,k)
print('testcenter:',testcenter)

#K-means主体函数
def myK(k,dataset):
    Startpoint = startpoint(k,dataset)
    m,n = np.shape(Startpoint)
    print('n', n)
    centerpoint = Startpoint
    labelset = classfy(dataset,Startpoint)
    newcenter = Center(dataset,labelset,k)
    print('外:cecnterpoint', centerpoint)
    print('外:newcenter', newcenter)
    flag = 0
    for i in range(k):
        for j in range(n):
            if centerpoint[i][j] != newcenter[i][j]:
                flag = 1
    while flag:
        print('循环')
        # print('里:cecnterpoint', centerpoint)
        # print('里:newcenter', newcenter)
        flag = 0
        for i in range(k):
            for j in range(n):
                if centerpoint[i][j] != newcenter[i][j]:
                    flag = 1
        # print('flag:',flag)
        centerpoint = newcenter[:]
        labelset = classfy(dataset,centerpoint)
        newcenter = Center(dataset, labelset, k)
    #print('final_resultlabel:',labelset)
    #print('cenerpoint:', centerpoint)
    return labelset,centerpoint

#测试
k=4
final_label,centerpoint = myK(k,dataset)
print('centerpoint:',centerpoint)
mat_center = np.mat(centerpoint)

#画图
# plt.scatter(test[:,0],test[:,1],40,10*(labelset+1))
plt.scatter(dataset[:, 0], dataset[:, 1],40,1*(final_label+0))
plt.show()

   