#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import torch.utils.data as data
import pickle
import numpy as np

path = '../data_set/'


# Without ID

class WebKBValidation(data.Dataset):    #两模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'webkb/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        page, link, target = self.train_page_data[index], self.train_link_data[index], self.labels[index]
        return page, link, target

    def __len__(self):
        return len(self.train_link_data)

class AdvValidation(data.Dataset):   #三模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'adv/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, target = self.train_view1_data[index], self.train_view2_data[index], self.train_view3_data[index], self.labels[index]
        return view1, view2, view3, target

    def __len__(self):
        return len(self.train_view2_data)

class BrcaValidation(data.Dataset):   #三模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'brca/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, target = self.train_view1_data[index], self.train_view2_data[index], self.train_view3_data[index], self.labels[index]
        return view1, view2, view3, target

    def __len__(self):
        return len(self.train_view2_data)

class BICValidation(data.Dataset):   #三模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'bic/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, target = self.train_view1_data[index], self.train_view2_data[index], self.train_view3_data[index], self.labels[index]
        return view1, view2, view3, target

    def __len__(self):
        return len(self.train_view2_data)

class GBMValidation(data.Dataset):   #三模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'gbm/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, target = self.train_view1_data[index], self.train_view2_data[index], self.train_view3_data[index], self.labels[index]
        return view1, view2, view3, target

    def __len__(self):
        return len(self.train_view2_data)

class ProteinValidation(data.Dataset):  #三模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'protein/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)

        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)

        length = self.__len__()
        # print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, target = self.train_view1_data[index], self.train_view2_data[index], \
                                      self.train_view3_data[index], self.labels[index]

        return view1, view2, view3, target

    def __len__(self):
        return len(self.train_view1_data)

class LSCCValidation(data.Dataset):  #三模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'lscc/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)

        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)

        length = self.__len__()
        # print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, target = self.train_view1_data[index], self.train_view2_data[index], \
                                      self.train_view3_data[index], self.labels[index]

        return view1, view2, view3, target

    def __len__(self):
        return len(self.train_view1_data)

class NoisymnistValidation(data.Dataset):    #两模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'noisymnist/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        page, link, target = self.train_page_data[index], self.train_link_data[index], self.labels[index]
        return page, link, target

    def __len__(self):
        return len(self.train_link_data)

class KRCCCValidation(data.Dataset):   #三模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'krccc/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, target = self.train_view1_data[index], self.train_view2_data[index], self.train_view3_data[index], self.labels[index]
        return view1, view2, view3, target

    def __len__(self):
        return len(self.train_view2_data)

class Caltech204Validation(data.Dataset):   #四模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'Caltech204/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, view4, target = self.train_view1_data[index], self.train_view2_data[index], self.train_view3_data[index], self.train_view4_data[index], self.labels[index]
        return view1, view2, view3, view4, target

    def __len__(self):
        return len(self.train_view2_data)

class Caltech20Validation(data.Dataset):   #六模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'Caltech20/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.train_view6_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.train_view6_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.train_view6_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, view4, view5, view6, target = self.train_view1_data[index], self.train_view2_data[index], self.train_view3_data[index], self.train_view4_data[index], self.train_view5_data[index], self.train_view6_data[index], self.labels[index]
        return view1, view2, view3, view4, view5, view6, target

    def __len__(self):
        return len(self.train_view2_data)

class CaltechallValidation(data.Dataset):   #六模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'Caltechall/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.train_view6_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.train_view6_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.train_view6_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, view4, view5, view6, target = self.train_view1_data[index], self.train_view2_data[index], self.train_view3_data[index], self.train_view4_data[index], self.train_view5_data[index], self.train_view6_data[index], self.labels[index]
        return view1, view2, view3, view4, view5, view6, target

    def __len__(self):
        return len(self.train_view2_data)

class ReutersValidation(data.Dataset):   #五模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'Reuters/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, view4, view5, target = self.train_view1_data[index], self.train_view2_data[index], self.train_view3_data[index], self.train_view4_data[index], self.train_view5_data[index], self.labels[index]
        return view1, view2, view3, view4, view5, target

    def __len__(self):
        return len(self.train_view2_data)

class Reuters2Validation(data.Dataset):   #两模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'Reuters2/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, target = self.train_view1_data[index], self.train_view2_data[index], self.labels[index]
        return view1, view2, target

    def __len__(self):
        return len(self.train_view2_data)

class BUSOBJValidation(data.Dataset):   #五模态
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'BUSOBJ/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_view1_data, self.train_view2_data, self.train_view3_data, self.train_view4_data, self.train_view5_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, view3, view4, view5, target = self.train_view1_data[index], self.train_view2_data[index], self.train_view3_data[index], self.train_view4_data[index], self.train_view5_data[index], self.labels[index]
        return view1, view2, view3, view4, view5, target

    def __len__(self):
        return len(self.train_view2_data)

