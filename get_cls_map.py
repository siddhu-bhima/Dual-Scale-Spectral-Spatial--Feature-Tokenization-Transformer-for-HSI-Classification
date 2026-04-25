get_cls_map.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import os


# ------------------ MAP RECONSTRUCTION ------------------ #
def get_classification_map(y_pred, y):

    height, width = y.shape
    cls_labels = np.zeros((height, width))

    k = 0
    for i in range(height):
        for j in range(width):

            if y[i, j] == 0:
                continue   # keep background

            cls_labels[i, j] = y_pred[k] + 1
            k += 1

    return cls_labels


# ------------------ COLOR MAP (7-CLASS HOUSTON) ------------------ #
def list_to_colormap(x_list):

    y = np.zeros((x_list.shape[0], 3))

    for index, item in enumerate(x_list):

        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.

        elif item == 1:
            y[index] = np.array([255, 0, 0]) / 255.
        elif item == 2:
            y[index] = np.array([0, 255, 0]) / 255.
        elif item == 3:
            y[index] = np.array([0, 0, 255]) / 255.
        elif item == 4:
            y[index] = np.array([255, 255, 0]) / 255.
        elif item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        elif item == 6:
            y[index] = np.array([0, 255, 255]) / 255.
        elif item == 7:
            y[index] = np.array([128, 128, 128]) / 255.

    return y


# ------------------ SAVE IMAGE ------------------ #
def classification_map(map, ground_truth, dpi, save_path):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(
        ground_truth.shape[1] * 2.0 / dpi,
        ground_truth.shape[0] * 2.0 / dpi
    )

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)   # 🔥 prevent memory issues


# ------------------ TEST FUNCTION ------------------ #
def test(device, net, loader):

    net.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in loader:

            inputs = inputs.to(device)
            outputs = net(inputs)

            preds = torch.argmax(outputs, dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    return np.array(y_pred), np.array(y_true)


# ------------------ MAIN FUNCTION ------------------ #
def get_cls_map(net, device, all_data_loader, y):

    # 🔥 Create folder if not exists
    os.makedirs('classification_maps', exist_ok=True)

    # 🔥 Predict all pixels
    y_pred, _ = test(device, net, all_data_loader)

    # 🔥 Reconstruct full map
    cls_labels = get_classification_map(y_pred, y)

    # Flatten
    x = np.ravel(cls_labels)
    gt = y.flatten()

    # Color mapping
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    # Reshape
    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))

    # 🔥 Save Houston outputs
    classification_map(y_re, y, 300,
                       'classification_maps/Houston_predictions.png')

    classification_map(gt_re, y, 300,
                       'classification_maps/Houston_gt.png')

    print('------ Houston classification maps generated successfully -------')