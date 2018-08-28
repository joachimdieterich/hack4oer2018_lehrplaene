import os
import sys
import argparse
import re
import pdf_renderer.libpdf_python as pdf_python
import json


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-i', '--input_path', required=True, help='input path')
    parser.add_argument('-o', '--output_path', required=True, help='input path')
    args = parser.parse_args()
    return args


# type = content_match.group(1)
# school = content_match.group(2)
# year = content_match.group(3)
# course = content_match.group(4)


def main():
    args = parse_args()

    pdf_re = re.compile('(.*?)\.pdf')

    path_re = re.compile(r'^([^/]*?)/([^/]*?)/([^/]*?)(/.*?)?/[^/]+\.pdf$')

    existing = []
    paper_text_map = []

    for root, dirs, files in os.walk(args.input_path):
        for file in files:
            match = re.match(pdf_re, file)
            if not match:
                continue
            if match.group(1) in existing:
                continue
            existing.append(match.group(1))

            file_path = os.path.join(root, file)

            path_match = re.match(path_re, file_path)
            if not path_match:
                continue

            type = path_match.group(4)
            school = path_match.group(1)
            year = path_match.group(2)
            course = path_match.group(3)

            with open(file_path, 'rb') as f:
                data = f.read()
                document = pdf_python.Document(data)
                text = ''
                for page_id in range(document.size()):
                    page_text = document.page(page_id).text()
                    text += page_text
                paper_text_map.append({
                    'text_origin': text,
                    'hash': match.group(0),
                    'type': type,
                    'school': school,
                    'year': year,
                    'course': course
                })

    with open(args.output_path, 'w') as f:
        for x in paper_text_map:
            f.write(json.dumps(x))

    return 0


if __name__ == '__main__':
    sys.exit(main())
