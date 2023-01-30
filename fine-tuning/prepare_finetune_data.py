import os
import json


src_dir = 'final_emails/'


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


if __name__ == '__main__':
    files = os.listdir(src_dir)
    data = list()
    for file in files:
        text = open_file(src_dir + file)
        chunks = text.split('EMAIL:')
        info = {'prompt': chunks[0].strip() + '\n\nEMAIL:', 'completion': '\n\n' + chunks[1].strip()}
        data.append(info)
    with open('emails.jsonl', 'w') as outfile:
        for i in data:
            json.dump(i, outfile)
            outfile.write('\n')
