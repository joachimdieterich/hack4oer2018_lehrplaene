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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-i', '--input_path', required=True, help='input path')
    parser.add_argument('-t', '--target', choices=['year', 'school', 'course'], default='course', help='input path')
    parser.add_argument('-o', '--output_path', required=True, help='output path')
    args = parser.parse_args()
    return args


def gen_mapping(data):
    data_set = set(data)
    return {x: y for x, y in zip(data_set, range(len(data_set)))}


def run_mapping(mapping, data):
    return [mapping[x] for x in data]


def save(path, clf, mapping):
    with open(path + '_svm.bin', 'wb') as f:
        f.write(pickle.dumps(clf))

    with open(path + '_mapping.bin', 'wb') as f:
        f.write(pickle.dumps(mapping))


def main():
    args = parse_args()

    with open(args.input_path, 'rb') as f:
        texts, school, year, course, ems = pickle.loads(f.read())

    feature_train = np.squeeze(np.asarray(ems))
    print(np.squeeze(np.asarray(ems)).shape)

    target = year
    if args.target == 'school':
        target = school
    if args.target == 'course':
        target = course

    target_mapping = gen_mapping(target)
    target = run_mapping(target_mapping, target)

    print('train SVM')
    clf = svm.SVC()
    clf.fit(feature_train, target)

    save(args.output_path, clf, target_mapping)


if __name__ == '__main__':
    sys.exit(main())
