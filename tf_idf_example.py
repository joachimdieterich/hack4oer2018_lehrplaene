#!/usr/bin/env python3

import os
import sys
import argparse
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-i', '--input_path', required=True, help='input path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    text = []
    school = []
    year = []
    course = []

    with open(args.input_path, 'r') as json_f:
        for line in json_f:
            data_dict = json.loads(line)

            print(data_dict['text_origin'])
            text.append(data_dict['text_origin'])
            school.append(data_dict['school'])
            year.append(data_dict['year'])
            course.append(data_dict['course'])

    count_transformer = CountVectorizer()
    feature_count = count_transformer.fit_transform(text)
    tfidf_transformer = TfidfTransformer().fit(feature_count)
    feature_tfidf = tfidf_transformer.transform(feature_count)
    print(feature_tfidf.shape)
    return 0


if __name__ == '__main__':
    sys.exit(main())
