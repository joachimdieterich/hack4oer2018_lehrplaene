import tensorflow as tf
import tensorflow_hub as hub
import json

from argparse import ArgumentParser
import os
from os import listdir
from os.path import isfile, join

import os
import sys
import argparse
import json

import pickle

import itertools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-i', '--input_path', required=True, help='input path')
    parser.add_argument('-t', '--target', choices=['year', 'school', 'course'], default='course', help='input path')
    parser.add_argument('-m', '--model', required=True, help='output path')
    args = parser.parse_args()
    return args


def gen_mapping(data):
    data_set = set(data)
    return {x: y for x, y in zip(data_set, range(len(data_set)))}


def run_mapping(mapping, data):
    return [mapping[x] for x in data]


def load(path):
    with open(path + '_svm.bin', 'rb') as f:
        clf = pickle.loads(f.read())

    with open(path + '_mapping.bin', 'rb') as f:
        mapping = pickle.loads(f.read())

    return clf, mapping


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(
    #         j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    args = parse_args()

    with open(args.input_path, 'rb') as f:
        texts, school, year, course, ems = pickle.loads(f.read())

    feature_test = np.squeeze(np.asarray(ems))
    print(np.squeeze(np.asarray(ems)).shape)

    target = year
    if args.target == 'school':
        target = school
    if args.target == 'course':
        target = course

    clf, mapping = load(args.model)

    print('Num classes: {}'.format(len(mapping)))

    target = run_mapping(mapping, target)

    print('eval SVM')
    #print('Accuracy: {}'.format(clf.score(feature_test, target)))

    prediction = clf.predict(feature_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target, prediction)
    np.set_printoptions(precision=2)

    inv_mapping = {v: k for k, v in mapping.items()}

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(12, 12))
    plot_confusion_matrix(cnf_matrix, classes=mapping.keys(), title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure(figsize=(12, 12))
    plot_confusion_matrix(cnf_matrix, classes=mapping.keys(), normalize=True, title='Normalized confusion matrix')

    plt.figure(figsize=(12, 6))

    plt.title('prediction')
    label_map = [inv_mapping[x] for x in range(len(mapping))]

    tick_marks = np.arange(len(label_map))

    plt.hist(prediction, bins=tick_marks)
    plt.xticks(tick_marks, label_map, rotation=45)

    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.tick_params(axis='both', which='minor', labelsize=6)

    plt.figure(figsize=(12, 6))
    plt.title('target')
    label_map = [inv_mapping[x] for x in range(len(mapping))]
    tick_marks = np.arange(len(label_map))

    plt.hist(target, bins=tick_marks)
    plt.xticks(tick_marks, label_map, rotation=45)

    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.tick_params(axis='both', which='minor', labelsize=6)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
