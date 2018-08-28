import os
import sys
import argparse
import re
import json


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-i', '--input_path', required=True, help='input path')
    parser.add_argument('-o', '--output_path', required=True, help='input path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.output_path + '.txt', 'w') as f, open(args.output_path + '.hash', 'w') as f2:


        json_re = re.compile(r'(.*?)\.json')
        if os.path.isdir(args.input_path):
            for root, dirs, files in os.walk(args.input_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    match = re.match(json_re, file)
                    if not match:
                        continue
                    hash_id = match.group(1)

                    with open(file_path, 'r') as json_f:
                        data_dict = json.load(json_f)
                        f.write(data_dict['text'] + '\n')
                        f2.write(hash_id + '\n')

        else:
            with open(args.input_path, 'r') as json_f:
                for line in json_f:
                    data_dict = json.loads(line)
                    f.write(data_dict['text'] + '\n')
                    f2.write(hash_id + '\n')

    return 0


if __name__ == '__main__':
    sys.exit(main())
