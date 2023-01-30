#!/usr/bin/env python
import requests
import openai
import config
import argparse
from pprint import pprint
openai.api_key = config.OPENAI_API_KEY

def file_upload(filename, purpose='fine-tune'):
    resp = openai.File.create(purpose=purpose, file=open(filename))
    pprint(resp)
    return resp

def file_list():
    resp = openai.File.list()
    pprint(resp)

def finetune_model(fileid, suffix, model='davinci'):
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % openai.api_key}
    payload = {'training_file': fileid, 'model': model, 'suffix': suffix}
    resp = requests.request(method='POST', url='https://api.openai.com/v1/fine-tunes', json=payload, headers=header, timeout=45)
    pprint(resp.json())

def finetune_list():
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % openai.api_key}
    resp = requests.request(method='GET', url='https://api.openai.com/v1/fine-tunes', headers=header, timeout=45)
    pprint(resp.json())

def finetune_events(ftid):
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % openai.api_key}
    resp = requests.request(method='GET', url='https://api.openai.com/v1/fine-tunes/%s/events' % ftid, headers=header, timeout=45)
    pprint(resp.json())

def finetune_get(ftid):
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % openai.api_key}
    resp = requests.request(method='GET', url='https://api.openai.com/v1/fine-tunes/%s' % ftid, headers=header, timeout=45)
    pprint(resp.json())

parser = argparse.ArgumentParser()
parser.add_argument('--build', action='store_true', help='Upload and train new model')
parser.add_argument('--list', action='store_true', help='List models')
args = parser.parse_args()

if __name__ == '__main__':
    if args.build:
        resp = file_upload('emails.jsonl')
        finetune_model(resp['id'], 'gerador_de_emails_pt_BR', 'davinci')
    if args.list:
        finetune_list()
