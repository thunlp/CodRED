import argparse
import json
import random
from functools import partial
from multiprocessing import Pool

import redis
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


def process(line):
    article = json.loads(line)
    tokens = list()
    mapping = dict()
    doc_id = int(article['id'])
    for para_id, para in enumerate(article['tokens']):
        for sent_id, sentence in enumerate(para):
            for word_id, word in enumerate(sentence):
                subwords = tokenizer.tokenize(word)
                mapping[(para_id, sent_id, word_id)] = list(range(len(tokens), len(tokens) + len(subwords)))
                tokens.extend(subwords)
    qs = list()
    for entity in article['vertexSet']:
        spans = list()
        for mention in entity:
            if 'Q' in mention:
                subwords = list()
                for position in range(mention['pos'][2], mention['pos'][3]):
                    subwords.extend(mapping[(mention['pos'][0], mention['pos'][1], position)])
                span = [min(subwords), max(subwords) + 1]
                spans.append(span)
        if len(spans) == len(entity):
            qs.append({
                'Q': entity[0]['Q'],
                'spans': spans
            })
        else:
            qs.append(None)
    instances = list()
    kset = set()
    for edge in article['edgeSet']:
        h = edge['h']
        t = edge['t']
        kset.add((h, t))
        if qs[h] is None or qs[t] is None:
            continue
        for r in edge['rs']:
            if 'P' + str(r) in relations:
                span_h = qs[h]['spans'][0]
                span_t = qs[t]['spans'][0]
                instances.append([doc_id, span_h[0], span_h[1], span_t[0], span_t[1], 'P' + str(r)])
    no_relations = list()
    for i in range(len(qs)):
        if qs[i] is None:
            continue
        for j in range(len(qs)):
            if qs[j] is None:
                continue
            if i != j and (i, j) not in kset:
                no_relations.append((i, j))
    if len(no_relations) > len(instances):
        no_relations = random.choices(no_relations, k=len(instances))
    for i, j in no_relations:
        instances.append([doc_id, qs[i]['spans'][0][0], qs[i]['spans'][0][1], qs[j]['spans'][0][0], qs[j]['spans'][0][1], 'n/a'])
    redisd.set(f'dsre-doc-{doc_id}', json.dumps(tokens))
    return instances, article['title'] in dev_docs


def initializer(base_model, _relations, t_docs):
    global redisd
    global tokenizer
    global relations
    global dev_docs
    redisd = redis.Redis(host='localhost', port=6379, decode_responses=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    relations = set(_relations)
    dev_docs = t_docs


def main(base_model):
    redisd = redis.Redis(host='localhost', port=6379, decode_responses=True)
    dev_dataset = json.load(open('rawdata/dev_dataset.json'))
    dev_docs = set(map(lambda x: x[1], dev_dataset)) | set(map(lambda x: x[2], dev_dataset))
    relations = json.load(open('rawdata/relations.json'))
    lines = list()
    with open('rawdata/distant_documents.jsonl') as f:
        for line in tqdm(f):
            lines.append(line.strip())
    train_examples = list()
    dev_examples = list()
    with Pool(48, initializer=initializer, initargs=(base_model, relations, dev_docs)) as p:
        for instances, is_dev in tqdm(p.imap(process, lines)):
            if is_dev:
                dev_examples.extend(instances)
            else:
                train_examples.extend(instances)
    json.dump(train_examples, open('dsre_train_examples.json', 'w'))
    json.dump(dev_examples, open('dsre_dev_examples.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='bert-base-cased')
    args = parser.parse_args()
    main(args.base_model)
