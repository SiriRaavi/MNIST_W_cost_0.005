import numpy as np
from scipy.spatial import distance
import os


def count_mislabel(gen, img_names):
    """
    :param gen: generator instance
    :param img_names: image names in the current batch
    :return:
    """
    mislab = np.zeros((gen.batch_size, gen.nb_samples_per_class * gen.nb_classes)).astype('int32')
    for w in range(gen.batch_size):
        for v in range(1):
            if int(img_names[w, v][2:]) > 399:
                mislab[w, v] = 1
    return mislab


# def count_mislabel(path):
#     """
#     :param gen: generator instance
#     :param img_names: image names in the current batch
#     :return:
#     """
#     count = 0
#     for i in os.listdir(path):
#         if int(i.replace('.png', '')[2:]) > 399:
#             count += 1
#     return count


def update_dict(d, acc, img_names, fb):
    # Taking only the name of the training examples (not the validations)
    names = img_names[:, :fb * 2]
    # Adding the new names to the dictionary keys
    for name in np.unique(names):
        if name not in d:
            d[name] = 0
    # searching for the decreased accuracies
    if acc[fb] <= 0.5:
        # if the prediction accuracy of the first validation samples (from both classes) is dropped
        for batch in names:
            for name in batch:
                d[name] += 1
    return d


def prepare_dict_to_save(d):
    keys = list(d.keys())
    vals = list(d.values())
    cls_label = np.array([int(k[0]) for k in keys])
    sample_name = np.array([int(k[2:]) for k in keys])
    count = np.array([int(k) for k in vals])
    return np.concatenate((cls_label.reshape(-1, 1), sample_name.reshape(-1, 1), count.reshape(-1, 1)), axis=1)


def prepare_img_names_to_save(img_names):
    cls_label = np.array([int(k[0]) for k in img_names[0]])
    img_num = np.array([int(k[2:]) for k in img_names[0]])
    return cls_label, img_num



def uniqueness_test(labels):
    """
    To test if all rows of labels array is unique or not
    :param labels: array of labels of size [nb_classes, len(vals)*num]
    :return: boolean
    """
    num_cls = labels.shape[0]
    a = np.zeros((num_cls, num_cls))
    for ii in range(num_cls):
        for jj in range(num_cls):
            if np.all(labels[ii, :] == labels[jj, :]):
                a[ii, jj] += 1
    return np.sum(a) == num_cls


def generate_labels(values, nb_class, how_many):
    """
    :param values: list of values to be picked from
    :param nb_class: number of classes (number of rows of the final label array)
    :param how_many: number of times randomly selecting from values array
    :return: matrix of one-hot-encoded labels
    """
    labels = np.zeros((0, len(values) * how_many))
    a = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    for i in range(len(a)):
        for j in range(len(values)):
            for k in range(how_many):
                labels = np.concatenate((labels, np.array(a[i] + a[j] + a[k]).reshape(-1, len(values) * how_many)))
    return np.random.permutation(labels)[:nb_class]


def probs_to_ohe(mat, letter_oh_len):
    """
    Converts the matrix of predictions into matrix of one-hot-encoded values
    :param letter_oh_len: length of one-hot-encoded vectors associated to each single letter (e.g. 'a':[0,0,1] )
    :param mat: matrix of prediction values of size [batch_size*nb_classes*nb_samples_per_class, ohe_len] (1600, 9)
    :return: matrix of one-hot-encoded values of the same size as mat
    """
    mat_cat = np.argmax(mat.reshape((-1, letter_oh_len)), axis=1)   # (1600, 9) -> (4800, 3) -> (4800)
    mat_ = (np.arange(letter_oh_len) == mat_cat[:, None]).astype(np.int32)   # one-hot-encoding  -> (4800, 3)
    return mat_.reshape((-1, mat.shape[1]))   #  -> (1600, 9)


def OEH_to_categorical(unique_label, preds, y_ohe, batch_size, letter_o_len):
    """
    Changes the one-hot-encoded labels to categorical values
    :param letter_o_len: length of one-hot-encoded vectors associated to each single letter (e.g. 'a':[0,0,1] )
    :param batch_size: batch size
    :param unique_label: matrix of unique labels of size [nb_classes, ohe_len]  (10, 9)
    :param preds: matrix of predictions to be used when computing the euclidean distance, (1600, 9)
    :param y_ohe: numpy array of one-hot-encoded labels of size [batch_size*nb_classes*nb_samples_per_class, ohe_len] (1600, 9)
    :return: array of size [batch_size, nb_classes*nb_samples_per_class] (16, 100)
    """
    y_cat = [np.squeeze(np.asarray(list(np.where((ohe == unique_label).all(axis=1))))) for ohe in y_ohe]
    # y_cat is the array of length batch_size*nb_classes*nb_samples_per_class  (1600)
    for n in range(len(y_cat)):
        if np.array(y_cat[n]).size == 0:    #if a OHE-label was not in the set of unique labels selected for training
            y_cat[n] = closest_label(unique_label, preds[n], letter_o_len)
    return np.array(y_cat).reshape((batch_size, -1))


def closest_label(unique_label, pred_, letter_len):
    """
    Finds the closest label from the set of unique labels
    :param unique_label: matrix of unique labels of size [nb_classes, ohe_len]  (10, 9)
    :param pred_: single vector of length [ohe_len]  (1, 9)
    :param letter_len: length of one-hot-encoded vectors associated to each single letter (e.g. 'a':[0,0,1] )
    :return: single integer number (row number of closest label in unique labels matrix)
    """
    pred_ = pred_.reshape((letter_len, -1))     # -> 3x3
    dist = np.array([])
    for row in unique_label:
        dist = np.append(dist,
                         np.sum([distance.euclidean(row.reshape((letter_len, -1))[m],
                                                    pred_[m]) for m in range(letter_len)]))
    return np.argmin(dist)
