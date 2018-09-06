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


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-i', '--input_path', required=True, help='input path')
    parser.add_argument('-t', '--text_field', choices=['text', 'text_origin'], default='text_origin', help='input path')
    parser.add_argument('-o', '--output_path', required=True, help='output path')
    args = parser.parse_args()
    return args


def gen_mapping(data):
    data_set = set(data)
    return {x: y for x, y in zip(data_set, range(len(data_set)))}


def run_mapping(mapping, data):
    return [mapping[x] for x in data]


def save(path, clf, year_mapping, tfidf_transformer, count_transformer):
    with open(path + '_svm.bin', 'wb') as f:
        f.write(pickle.dumps(clf))

    with open(path + '_mapping.bin', 'wb') as f:
        f.write(pickle.dumps(year_mapping))

    with open(path + '_tfidf.bin', 'wb') as f:
        f.write(pickle.dumps(tfidf_transformer))

    with open(path + '_count.bin', 'wb') as f:
        f.write(pickle.dumps(count_transformer))


# 33 Matthias
def main():
    args = parse_args()

    texts = []
    school = []
    year = []
    course = []

    with open(args.input_path, 'r') as json_f:
        for i, line in enumerate(json_f):

            data_dict = json.loads(line)

            texts.append(data_dict[args.text_field])
            school.append(data_dict['school'])
            year.append(int(data_dict['year']))
            course.append(data_dict['course'])

    print(texts[:1])

    with tf.Graph().as_default():
        module_url = "https://tfhub.dev/google/nnlm-de-dim128/1"
        embed = hub.Module(module_url)
        ph_string = tf.placeholder(dtype=tf.string)
        embeddings = embed(ph_string)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            ems = []
            for i, x in enumerate(texts):
                print(i)
                em = sess.run(embeddings, feed_dict={ph_string: [x]})
                ems.append(em)

        with open(args.output_path, 'wb') as f:
            f.write(pickle.dumps((texts, school, year, course, ems)))


if __name__ == '__main__':
    sys.exit(main())
