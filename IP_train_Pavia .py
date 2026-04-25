
# IMPORTS
import numpy as np
import scipy.io as sio
import pandas as pd
import time
import random

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv

import get_cls_map
import SSFTTnet


#LOAD DATA 
def loadData():
    data = sio.loadmat('../data/PaviaU.mat')['paviaU']
    labels = sio.loadmat('../data/PaviaU_gt.mat')['paviaU_gt']
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    return data, labels


def getRGBImage(X):
    r, g, b = X[:, :, 30], X[:, :, 20], X[:, :, 10]
    rgb = np.stack((r, g, b), axis=2).astype(np.float32)
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-8)
    return rgb


#  PCA 
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    return np.reshape(newX, (X.shape[0], X.shape[1], numComponents))


# PATCH CREATION
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0]+2*margin, X.shape[1]+2*margin, X.shape[2]))
    newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X
    return newX


def createImageCubes(X, y, windowSize=5):
    margin = (windowSize - 1) // 2
    zeroPaddedX = padWithZeros(X, margin)
    patchesData = []
    patchesLabels = []

    for r in range(margin, zeroPaddedX.shape[0]-margin):
        for c in range(margin, zeroPaddedX.shape[1]-margin):
            patch = zeroPaddedX[r-margin:r+margin+1, c-margin:c+margin+1]
            label = y[r-margin, c-margin]
            if label > 0:
                patchesData.append(patch)
                patchesLabels.append(label - 1)

    return np.array(patchesData), np.array(patchesLabels)


# DATASET 
class TrainDS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.x_data = torch.FloatTensor(X)
        self.y_data = torch.LongTensor(y)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return len(self.x_data)


class TestDS(TrainDS):
    pass


#  DATA LOADER 
def create_data_loader():
    X, y = loadData()

    test_ratio = 0.95
    patch_small, patch_large = 9, 13
    pca_components = 15

    X_pca = applyPCA(X, pca_components)

    X_small, y_all = createImageCubes(X_pca, y, patch_small)
    X_large, _ = createImageCubes(X_pca, y, patch_large)

    Xtrain_s, Xtest_s, ytrain, ytest = train_test_split(
        X_small, y_all, test_size=test_ratio, stratify=y_all)

    Xtrain_l, Xtest_l, _, _ = train_test_split(
        X_large, y_all, test_size=test_ratio, stratify=y_all)

    # reshape
    Xtrain_s = Xtrain_s.reshape(-1, patch_small, patch_small, pca_components, 1)
    Xtrain_l = Xtrain_l.reshape(-1, patch_large, patch_large, pca_components, 1)

    Xtest_s = Xtest_s.reshape(-1, patch_small, patch_small, pca_components, 1)
    Xtest_l = Xtest_l.reshape(-1, patch_large, patch_large, pca_components, 1)

    # crop large
    center, half = patch_large//2, patch_small//2
    Xtrain_l = Xtrain_l[:, center-half:center+half+1, center-half:center+half+1]
    Xtest_l = Xtest_l[:, center-half:center+half+1, center-half:center+half+1]

    # fusion
    Xtrain = np.concatenate([Xtrain_s, Xtrain_l], axis=3)
    Xtest = np.concatenate([Xtest_s, Xtest_l], axis=3)

    # reshape for pytorch
    Xtrain = Xtrain.reshape(-1, patch_small, patch_small, pca_components*2, 1)
    Xtest = Xtest.reshape(-1, patch_small, patch_small, pca_components*2, 1)

    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)

    train_loader = torch.utils.data.DataLoader(
        TrainDS(Xtrain, ytrain), batch_size=16, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        TestDS(Xtest, ytest), batch_size=16, shuffle=False)

    return train_loader, test_loader, y_all


# TRAIN 
def train(train_loader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SSFTTnet.SSFTTnet().to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            output = net(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return net, device


#  TEST 
def test(device, net, test_loader):
    net.eval()
    y_pred, y_true = [], []

    for data, target in test_loader:
        data = data.to(device)
        output = net(data)
        pred = np.argmax(output.detach().cpu().numpy(), axis=1)

        y_pred.extend(pred)
        y_true.extend(target.numpy())

    return np.array(y_pred), np.array(y_true)


# METRICS 
def acc_reports(y_true, y_pred):
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    each_acc = np.diag(confusion) / np.sum(confusion, axis=1)
    aa = np.mean(each_acc)

    return oa*100, aa*100, kappa*100


# MAIN
if __name__ == "__main__":

    train_loader, test_loader, y_all = create_data_loader()

    net, device = train(train_loader, epochs=50)

    y_pred, y_test = test(device, net, test_loader)

    oa, aa, kappa = acc_reports(y_test, y_pred)

    print("OA:", oa)
    print("AA:", aa)
    print("Kappa:", kappa)

