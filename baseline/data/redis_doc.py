import argparse
import json
from multiprocessing import Pool

import redis
from tqdm import tqdm
from transformers import AutoTokenizer


def process(line):
    line = line.strip()
    if len(line) == 0:
        return None
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
        assert len(entity) > 0
        spans = list()
        for mention in entity:
            subwords = list()
            for position in range(mention['pos'][2], mention['pos'][3]):
                k = (mention['pos'][0], mention['pos'][1], position)
                if k in mapping:
                    subwords.extend(mapping[k])
            if len(subwords) > 0:
                span = [min(subwords), max(subwords) + 1]
                spans.append(span)
        if len(spans) > 0:
            k = dict()
            for key in entity[0]:
                if key != 'pos':
                    k[key] = entity[0][key]
                    k['spans'] = spans
            qs.append(k)
    obj = dict()
    obj['tokens'] = tokens
    obj['entities'] = qs
    obj['id'] = article['id']
    obj['title'] = article['title']
    redisd.set(f'codred-doc-{obj["title"]}', json.dumps(obj))
    return doc_id, article['title']


def initializer(base_model):
    global redisd
    global tokenizer
    redisd = redis.Redis(host='localhost', port=6379, decode_responses=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)


def main(base_model):
    redisd = redis.Redis(host='localhost', port=6379, decode_responses=True)
    all_ids = list()
    with open('rawdata/wiki_ent_link.jsonl') as f:
        with Pool(48, initializer=initializer, initargs=(base_model,)) as p:
            for doc_id, title in tqdm(p.imap_unordered(process, f)):
                all_ids.append([doc_id, title])
    json.dump(all_ids, open('all_docs.json', 'w'))
    popular_ids = list()
    with open('rawdata/popular_page_ent_link.jsonl') as f:
        with Pool(48, initializer=initializer, initargs=(base_model,)) as p:
            for doc_id, title in tqdm(p.imap_unordered(process, f)):
                popular_ids.append([doc_id, title])
    json.dump(popular_ids, open('popular_docs.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='bert-base-cased')
    args = parser.parse_args()
    main(args.base_model)
