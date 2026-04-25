```python
# IMPORTS
import numpy as np
import scipy.io as sio
import time

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv

import SSFTTnet
import get_cls_map


# LOAD DATA
def loadData():
    data = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
    labels = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']
    return data, labels


# PCA
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    return np.reshape(newX, (X.shape[0], X.shape[1], numComponents))


# PATCH
def padWithZeros(X, margin):
    newX = np.zeros((X.shape[0]+2*margin, X.shape[1]+2*margin, X.shape[2]))
    newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X
    return newX


def createImageCubes(X, y, windowSize):
    margin = (windowSize - 1) // 2
    zeroPaddedX = padWithZeros(X, margin)

    patchesData, patchesLabels = [], []

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
        self.x = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class TestDS(TrainDS):
    pass


# DATA LOADER
def create_data_loader():
    X, y = loadData()
    test_ratio = 0.90

    patch_small = 13
    patch_large = 21
    pca_components = 30

    X_pca = applyPCA(X, pca_components)

    X_small, y_all = createImageCubes(X_pca, y, patch_small)
    X_large, _ = createImageCubes(X_pca, y, patch_large)

    Xtrain_s, Xtest_s, ytrain, ytest = train_test_split(
        X_small, y_all, test_size=test_ratio, stratify=y_all)

    Xtrain_l, Xtest_l, _, _ = train_test_split(
        X_large, y_all, test_size=test_ratio, stratify=y_all)

    Xtrain_s = Xtrain_s.reshape(-1, patch_small, patch_small, pca_components, 1)
    Xtrain_l = Xtrain_l.reshape(-1, patch_large, patch_large, pca_components, 1)

    Xtest_s = Xtest_s.reshape(-1, patch_small, patch_small, pca_components, 1)
    Xtest_l = Xtest_l.reshape(-1, patch_large, patch_large, pca_components, 1)

    center, half = patch_large//2, patch_small//2

    Xtrain_l = Xtrain_l[:, center-half:center+half+1, center-half:center+half+1]
    Xtest_l = Xtest_l[:, center-half:center+half+1, center-half:center+half+1]

    Xtrain = np.concatenate([Xtrain_s, Xtrain_l], axis=3)
    Xtest = np.concatenate([Xtest_s, Xtest_l], axis=3)

    Xtrain = Xtrain.reshape(-1, patch_small, patch_small, pca_components*2, 1)
    Xtest = Xtest.reshape(-1, patch_small, patch_small, pca_components*2, 1)

    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)

    train_loader = torch.utils.data.DataLoader(
        TrainDS(Xtrain, ytrain), batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        TestDS(Xtest, ytest), batch_size=64, shuffle=False)

    return train_loader, test_loader


# TRAIN
def train(train_loader, epochs=100):
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


# TEST
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

    train_loader, test_loader = create_data_loader()

    net, device = train(train_loader, epochs=100)

    y_pred, y_test = test(device, net, test_loader)

    oa, aa, kappa = acc_reports(y_test, y_pred)

    print("OA:", oa)
    print("AA:", aa)
    print("Kappa:", kappa)
```
