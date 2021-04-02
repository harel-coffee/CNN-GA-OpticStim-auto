from models import *

import pickle
import numpy as np

from sklearn.feature_selection import mutual_info_classif

def shift_2d_replace(data, dx, dy, constant=False):
    """
    Shifts the array in two dimensions while setting rolled values to constant
    :param data: The 2d numpy array to be shifted
    :param dx: The shift in x
    :param dy: The shift in y
    :param constant: The constant to replace rolled values with
    :return: The shifted array with "constant" where roll occurs
    """
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data

def extract_from_categories(x, y, cat_list):
    """Extract samples from dataset according to their category."""
    categories = np.array([int(np.where(el==1)[0][0]) for el in y])
    idx = [ind for ind, val in enumerate(categories) if val in cat_list]
    y_cat = y[idx, :]
    x_cat = x[idx, :, :]
    return x_cat, y_cat


def separate_testset3(x_test, y_test, n_samples_per_cat, save=False, filename=None):
    categories = np.array([int(np.where(el==1)[0][0]) for el in y_test])
    n_cat = np.unique(categories).size
    y1 = [None for i in range(n_cat)]
    x1 = [None for i in range(n_cat)]
    y2 = [None for i in range(n_cat)]
    x2 = [None for i in range(n_cat)]
    y3 = [None for i in range(n_cat)]
    x3 = [None for i in range(n_cat)]
    for i in range(n_cat):
        idx = np.arange(x_test.shape[0])
        idx = idx[np.where(categories == i)]
        np.random.shuffle(idx)
        idx1 = idx[:n_samples_per_cat[0]]
        idx2 = idx[n_samples_per_cat[0]:(n_samples_per_cat[0] + n_samples_per_cat[1])]
        idx3 = idx[(n_samples_per_cat[0] + n_samples_per_cat[1]):]
        y1[i] = y_test[idx1, :]
        x1[i] = x_test[idx1, :, :]
        y2[i] = y_test[idx2, :]
        x2[i] = x_test[idx2, :, :]
        y3[i] = y_test[idx3, :]
        x3[i] = x_test[idx3, :, :]
    y_exp = np.concatenate(y1, axis=0)
    x_exp = np.concatenate(x1, axis=0)
    y_cal = np.concatenate(y2, axis=0)
    x_cal = np.concatenate(x2, axis=0)
    y_left = np.concatenate(y3, axis=0)
    x_left = np.concatenate(x3, axis=0)

    idx = np.arange(x_cal.shape[0])
    np.random.shuffle(idx)
    x_cal = x_cal[idx, :, :, :]
    y_cal = y_cal[idx, :]

    idx = np.arange(x_exp.shape[0])
    np.random.shuffle(idx)
    x_exp = x_exp[idx, :, :, :]
    y_exp = y_exp[idx, :]

    idx = np.arange(x_left.shape[0])
    np.random.shuffle(idx)
    x_left = x_left[idx, :, :, :]
    y_left = y_left[idx, :]

    if save:
        with open(filename, 'wb') as f:
            pickle.dump([x_cal, y_cal, x_exp, y_exp, x_left, y_left], f)
    return (x_cal, y_cal), (x_exp, y_exp), (x_left, y_left)


def generate_shifted_data(input_image, input_to_layer_model, model):
    target_list = []
    target_out_list = []

    target_list.append(input_to_layer_model.predict(input_image))
    target_out = model.predict(input_image)
    target_out_list.append(target_out)
    cat = np.argmax(target_out[0, :])

    acc_disp = np.array(
        [
            [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
            [-1, -1], [-2, -2], [-3, -3], [-4, -4], [-5, -5],
            [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
            [0, -1], [0, -2], [0, -3], [0, -4], [0, -5],
            [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
            [-1, 0], [-2, 0], [-3, 0], [-4, 0], [-5, 0]
        ]
    )

    same_class_idx = []
    for i in range(acc_disp.shape[0]):
        new_input = shift_2d_replace(input_image[0, :, :, 0], dx=acc_disp[i, 0], dy=acc_disp[i, 1], constant=0)
        new_input = np.expand_dims(np.expand_dims(new_input, axis=0), axis=3)
        target_out = model.predict(new_input)
        target_cat = np.argmax(target_out[0, :])
        if target_cat == cat:
            same_class_idx.append(i)
            target_list.append(input_to_layer_model.predict(new_input))
            target_out_list.append(target_out)

    acc_disp = acc_disp[same_class_idx, :]
    return acc_disp, target_list, target_out_list

def select_random_stimulus(x_test, input_to_layer_model, model):
    sel = np.random.randint(low=0, high=x_test.shape[0])
    input = np.expand_dims(x_test[sel, :, :, :], axis=0)
    target = input_to_layer_model.predict(input)
    target_out = model.predict(input)
    target_cat = np.argmax(target_out[0, :])
    return sel, input, target, target_out, target_cat
