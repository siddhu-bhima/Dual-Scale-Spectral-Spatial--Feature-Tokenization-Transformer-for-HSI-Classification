

import numpy as np
import matplotlib.pyplot as plt
import torch

# ------------------ MAP RECONSTRUCTION ------------------ #
def get_classification_map(y_pred, y):

    height, width = y.shape
    cls_labels = np.zeros((height, width))

    k = 0
    for i in range(height):
        for j in range(width):

            if y[i, j] == 0:
                continue
            else:
                cls_labels[i, j] = y_pred[k] + 1
                k += 1

    return cls_labels


# ------------------ COLOR MAP (PAVIA - 9 CLASSES) ------------------ #
def list_to_colormap(x_list):

    y = np.zeros((x_list.shape[0], 3))

    for index, item in enumerate(x_list):

        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.  # background

        elif item == 1:
            y[index] = np.array([255, 0, 0]) / 255.      # Asphalt
        elif item == 2:
            y[index] = np.array([0, 255, 0]) / 255.      # Meadows
        elif item == 3:
            y[index] = np.array([0, 0, 255]) / 255.      # Gravel
        elif item == 4:
            y[index] = np.array([255, 255, 0]) / 255.    # Trees
        elif item == 5:
            y[index] = np.array([255, 0, 255]) / 255.    # Metal sheets
        elif item == 6:
            y[index] = np.array([0, 255, 255]) / 255.    # Bare soil
        elif item == 7:
            y[index] = np.array([128, 128, 128]) / 255.  # Bitumen
        elif item == 8:
            y[index] = np.array([255, 165, 0]) / 255.    # Bricks
        elif item == 9:
            y[index] = np.array([75, 0, 130]) / 255.     # Shadows

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

    return 0


# ------------------ TEST FUNCTION ------------------ #
def test(device, net, test_loader):

    net.eval()

    y_pred_test = []
    y_test = []

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            outputs = net(inputs)

            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)

            y_pred_test.extend(preds)
            y_test.extend(labels.numpy())

    return np.array(y_pred_test), np.array(y_test)


# ------------------ MAIN FUNCTION ------------------ #
def get_cls_map(net, device, all_data_loader, y):

    # Predict all pixels
    y_pred, _ = test(device, net, all_data_loader)

    # Reconstruct map
    cls_labels = get_classification_map(y_pred, y)

    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))

    # 🔥 FIXED FILE NAMES (PAVIA)
    classification_map(y_re, y, 300,
                       'classification_maps/Pavia_predictions.png')

    classification_map(gt_re, y, 300,
                       'classification_maps/Pavia_gt.png')

    print('------ Pavia classification maps generated successfully -------')
