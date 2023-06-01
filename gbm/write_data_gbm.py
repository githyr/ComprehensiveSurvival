#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function

import pickle as pickle
import numpy as np
import scipy.io as scio
from sklearn.model_selection import StratifiedShuffleSplit

path = '../data_set/'



def read_gbm():
    data = scio.loadmat(path + 'GBMData.mat')
    view1 = data['DNA']
    view2 = data['mRNA']   # miRNA  DNA mRNA
    view3 = data['miRNA']
    class_label = data['label']
    return view1, view2, view3, class_label


def write_gbm(data_set_name='gbm', train_size=0.1, test_size=0.9, validation_size=0.1, seed=3):
    if train_size + test_size != 1:
        print("Error !!! The sum of train_size, test_size should be 1")
        return
    print("Start write data, train size = ", train_size * 100, "%")

    view1, view2, view3, label = read_gbm()
    size_list = []
    class_size = 2
    for i in range(class_size):
        indexes = np.where(label == i)[0]
        size_list.append(len(indexes))
    print("Sample number in each class: ", size_list)
    view1 = np.asarray(view1, dtype=np.float32)
    view2 = np.asarray(view2, dtype=np.float32)
    view3 = np.asarray(view3, dtype=np.float32)
    label = np.asarray(label, dtype=np.int32)
    seed = seed
    # Split half of the data as test set ===========================================================================
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size, random_state=seed)
    for train_idx, test_idx in stratified_split.split(view1, label):
        view1_train, view1_test = view1[train_idx], view1[test_idx]
        view2_train, view2_test = view2[train_idx], view2[test_idx]
        view3_train, view3_test = view3[train_idx], view3[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]

    size_list = []
    for i in range(class_size):
        indexes = np.where(y_test == i)[0]
        size_list.append(len(indexes))
    print(size_list, "Test size")

    with open(path + data_set_name + '/test.pkl', 'wb') as f_test:
        pickle.dump((view1_test, view2_test, view3_test, y_test), f_test, -1)

    # Train and validation ===========================================================================
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
    for train_idx, test_idx in stratified_split.split(view1_train, y_train):
        view1_train, view1_validation = view1_train[train_idx], view1_train[test_idx]
        view2_train, view2_validation = view2_train[train_idx], view2_train[test_idx]
        view3_train, view3_validation = view3_train[train_idx], view3_train[test_idx]
        train_label, validation_label = y_train[train_idx], y_train[test_idx]

    with open(path + data_set_name + '/train.pkl', 'wb') as f_train:
        pickle.dump((view1_train, view2_train, view3_train, train_label), f_train, -1)
    with open(path + data_set_name + '/validation.pkl', 'wb') as f_train:
        pickle.dump((view1_validation, view2_validation, view3_validation, validation_label), f_train, -1)
    size_list = []
    for i in range(class_size):
        indexes = np.where(train_label == i)[0]
        size_list.append(len(indexes))
    print(size_list, "train size")
    size_list = []
    for i in range(class_size):
        indexes = np.where(validation_label == i)[0]
        size_list.append(len(indexes))
    print(size_list, "validation size")


def test_write(train_size=0.4):
    name = 'gbm'
    print("=============================== Train size = ", train_size, "=========================================")
    test_size = 1 - train_size
    write_gbm(train_size=train_size, test_size=test_size, validation_size=0.3)
    print("\n\n\nRead data ===========================================================================")
    with open(path + name + '/' + 'test.pkl', 'rb') as fp:
        train_view1_data, train_view2_data, train_view3_data, train_labels = pickle.load(fp)
    print("test size = ", len(train_labels), train_view1_data.shape, train_view2_data.shape, train_view3_data.shape)
    with open(path + name + '/' + '/train.pkl', 'rb') as fp:
        train_view1_data, train_view2_data, train_view3_data, train_labels = pickle.load(fp)
        print("train size = ", len(train_labels), train_view1_data.shape, train_view2_data.shape, train_view3_data.shape)
    with open(path + name + '/' + '/validation.pkl', 'rb') as fp:
        train_view1_data, train_view2_data, train_view3_data, train_labels = pickle.load(fp)
    print("validation size = ", len(train_labels), train_view1_data.shape, train_view2_data.shape, train_view3_data.shape)


if __name__ == "__main__":
    # Test
    test_write(train_size=0.5)
