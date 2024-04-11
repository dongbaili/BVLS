from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from toy import gen_spurious_feature

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

def k_means():
    shift_degree = 0.8
    X_train_s, X_test_s, g_train, g_test = gen_spurious_feature(X_train, X_test, y_train, y_test, shift_degree)

    print(X_train_s.shape)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X_train_s)
    kmeans_labels = kmeans.predict(X_train_s)

    # PCA降维，方便可视化
    pca = PCA(n_components=2)
    X = pca.fit_transform(X_train_s)

    indexs = [[],[],[],[]]
    for group_id in range(4):
        for i in range(len(g_train)):
            if g_train[i] == group_id:
                indexs[group_id].append(i)

    kmeans_indexs = [[],[],[],[]]
    for group_id in range(4):
        for i in range(len(kmeans_labels)):
            if kmeans_labels[i] == group_id:
                kmeans_indexs[group_id].append(i)

    # group by original group
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '2d'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    colors = ['red','blue','yellow',"green"]
    labels = ["0 0", "0 1", "1 0", "1 1"]
    for group_id in range(4):
        ax1.scatter(X[indexs[group_id], 0], X[indexs[group_id], 1], c=colors[group_id], label=labels[group_id])
    ax1.legend()
    ax1.set_title("Original Group")

    # group by k-means
    for group_id in range(4):
        ax2.scatter(X[kmeans_indexs[group_id], 0], X[kmeans_indexs[group_id], 1], c=colors[group_id], label=group_id)

    ax2.set_title("K-means")
    ax2.legend()
    plt.savefig("K-means_train.png")

k_means()