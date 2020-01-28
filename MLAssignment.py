import numpy as np
import matplotlib.pyplot as plt


def genGaussianSamples(N, m, C):
    A = np.linalg.cholesky(C)
    U = np.random.randn(N, 2)
    return (U @ A.T + m)


def gauss2D(x, m, C):
    Ci = np.linalg.inv(C)
    dC = np.linalg.det(C)
    num = np.exp(-0.5 * np.dot((x - m).T, np.dot(Ci, (x - m))))
    den = 2 * np.pi * (dC ** 0.5)

    return num / den


def twoDGaussianPlot(nx, ny, m, C):
    x = np.linspace(-7, 7, nx)
    y = np.linspace(-7, 7, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i, j], Y[i, j]])
            Z[i, j] = gauss2D(xvec, m, C)

    return X, Y, Z


def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def randCent(dataSet, k):
    m, n = dataSet.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))  #
        centroids[i, :] = dataSet[index, :]
    return centroids


def MyKMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChange = True

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    # for i in range(k):
    #     plt.plot(centroids[i, 0], centroids[i, 1], 'o', c='g')
    while clusterChange:
        clusterChange = False
        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
            # for i in range(k):
            #     plt.plot(centroids[i, 0], centroids[i, 1], 'o', c='g', )

    # for i in range(3):
    #     plt.plot(centroids[i, 0], centroids[i, 1], 'o', c='m')
    return centroids, clusterAssment


# Define three means
#
Means = np.array([[0, 3], [3, 0], [4, 4]])

# Define three covariance matrices ensuring
# they are positive definite
#

from sklearn.datasets import make_spd_matrix

CovMatrices = np.zeros((3, 2, 2))
for j in range(3):
    CovMatrices[j, :, :] = make_spd_matrix(2)
# Priors
#
w = np.random.rand(3)
w = w / np.sum(w)

# How many data in each component (1000 in total)
#
nData = np.floor(w * 1000).astype(int)
# Draw samples from each component
#
X0 = genGaussianSamples(nData[0], Means[0, :], CovMatrices[0, :, :])
X1 = genGaussianSamples(nData[1], Means[1, :], CovMatrices[1, :, :])
X2 = genGaussianSamples(nData[2], Means[2, :], CovMatrices[2, :, :])

# Append into an array for the data we need
#
X = np.append(np.append(X0, X1, axis=0), X2, axis=0)

plt.scatter(X0[:, 0], X0[:, 1], s=3, c='m')
plt.scatter(X1[:, 0], X1[:, 1], s=3, c='b')
plt.scatter(X2[:, 0], X2[:, 1], s=3, c='c')

nx, ny = 50, 40

m1 = np.array([0, 3])
m2 = np.array([3, 0])
m3 = np.array([4, 4])

C1 = CovMatrices[0, :, :]
C2 = CovMatrices[1, :, :]
C3 = CovMatrices[1, :, :]

Xp1, Yp1, Zp1 = twoDGaussianPlot(nx, ny, m1, C1)
Xp2, Yp2, Zp2 = twoDGaussianPlot(nx, ny, m2, C2)
Xp3, Yp3, Zp3 = twoDGaussianPlot(nx, ny, m3, C3)
plt.contour(Xp1, Yp1, Zp1, 3)
plt.contour(Xp2, Yp2, Zp2, 3)
plt.contour(Xp3, Yp3, Zp3, 3)

plt.grid(True)
plt.title('Sample data from mixture Gaussian density')
plt.show()

plt.scatter(X0[:, 0], X0[:, 1], s=3, c='k')
plt.scatter(X1[:, 0], X1[:, 1], s=3, c='k')
plt.scatter(X2[:, 0], X2[:, 1], s=3, c='k')
# centroids, clusterAssment = MyKMeans(X, 3)

import cmath

SSE = []
for k in range(1, 9):
    centroids, clusterAssment = MyKMeans(X, k)
    error = np.sum(clusterAssment[:, 1])
    SSE.append(error)

print("SSE is: ", SSE)

X = [1, 2, 3, 4, 5, 6, 7, 8]
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()

print("centroids are: ", centroids)
# def column(matrix, i):
#     return [row[i] for row in matrix]

plt.grid(True)
plt.title('K means clustering implementation')
plt.show()

from sklearn.cluster import KMeans

estimator = KMeans(n_clusters=3, random_state=9)
estimator.fit(X)

print("SSE is: ", estimator.inertia_)
C = estimator.cluster_centers_
print("Centroids for sklearn are: ", C)
plt.scatter(X[:, 0], X[:, 1], s=3, c='k')
for i in range(3):
    plt.plot(C[i, 0], C[i, 1], 'o', c='m')
plt.grid(True)
plt.title('Sklearn K means clustering implementation')
plt.show()

import pandas as pd

file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(file, header=None)
df = df.values

irisData = df[:, :4]
irisTarget = df[:, 4]

XSetosa = irisData[:50, :]
XVersicolour = irisData[50:100, :]
XVirginica = irisData[100:150, :]

XSetosa = XSetosa[:, [2, 3]]
XVersicolour = XVersicolour[:, [2, 3]]
XVirginica = XVirginica[:, [2, 3]]

plt.scatter(XSetosa[:, 0], XSetosa[:, 1], s=3, c='m', label='Iris Setosa')
plt.scatter(XVersicolour[:, 0], XVersicolour[:, 1], s=3, c='b', label='Iris Versicolour')
plt.scatter(XVirginica[:, 0], XVirginica[:, 1], s=3, c='c', label='Iris Virginica')
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title('Iris data set with last two dimensions')
plt.legend()
plt.grid(True)
plt.show()

centroids, clusterAssment = MyKMeans(irisData, 3)
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
irisData = irisData[:, [2, 3]]
m, n = irisData.shape
for i in range(m):
    markIndex = int(clusterAssment[i, 0])
    plt.plot(irisData[i, 0], irisData[i, 1], mark[markIndex])

for i in range(3):
    plt.plot(centroids[i, 2], centroids[i, 3], 'x', c='#0A122A')
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend()
plt.title('Kmeans clustering on Iris dataset on two dimensions')
plt.grid(True)
plt.show()
