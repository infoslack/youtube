#!/usr/bin/env python
import os
import config
import openai
import argparse
from time import time,sleep
from uuid import uuid4
openai.api_key = config.OPENAI_API_KEY

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

scenarios = open_file('baseline.txt').splitlines()

def gpt_completion(prompt, engine='text-davinci-003', temp=1.0, top_p=1.0, tokens=2048, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdf', 'asdasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='utf-8', errors='ignore').decode()

    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            #text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            save_file('logs/%s' % filename, prompt + '\n\n----------\n\n' + text)
            return text
        except Exception as fail:
            retry += 1
            if retry >= max_retry:
                return "GPT-3 error: %s" % fail
            print('Error communicating with OpenAI:', fail)
            sleep(1)

def create_scenario():
    for scenario in scenarios:
        prompt = open_file('prompt_bulletpoints.txt').replace('<<SCENARIO>>', scenario).replace('<<UUID>>', str(uuid4()))
        print('\n\n----------\n\n', prompt)
        completion = gpt_completion(prompt)
        filename = scenario.replace(' ','').lower()[0:10] + str(time()) + '.txt'
        save_file('bulletpoints/%s' % filename, completion)
        print('\n\n', completion)

def final_prompt():
    src_dir = 'bulletpoints/'
    for file in os.listdir(src_dir):
        scenario = open_file(src_dir + file)
        prompt = open_file('email_bulletpoints.txt').replace('<<SCENARIO>>', scenario)
        print('\n\n==========\n\n', prompt)
        completion = gpt_completion(prompt)
        print('\n\n', completion)
        filename = file.replace(' ','').lower()[0:10] + str(time()) + '.txt'
        output = scenario.strip() + '\n\nEMAIL:\n\n' + completion
        save_file('final_emails/%s' % filename, output)

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', action='store_true', help='Create scenarios')
parser.add_argument('--final', action='store_true', help='Create emails')
args = parser.parse_args()

if __name__ == '__main__':
    if args.scenario:
        create_scenario()
    if args.final:
        final_prompt()
