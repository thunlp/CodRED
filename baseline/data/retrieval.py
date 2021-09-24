import json
import math
import random
import sys
from itertools import product

import redis
from tqdm import tqdm


def count_candidates(q2t, t2q, h, t):
    d1 = q2t[h]
    d2 = q2t[t]
    candidates = list()
    for _1, _2 in product(d1, d2):
        e1s = t2q[_1]
        e2s = t2q[_2]
        if h in e1s and t in e2s:
            candidates.append([_1, _2, e1s[h] * e2s[t]])
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def place_data(dataset):
    epr2d = dict()
    key2docs = dict()
    for key, doc1, doc2, label in dataset:
        if key not in epr2d:
            epr2d[key] = set()
            key2docs[key] = set()
        epr2d[key].add(label)
        if label != 'n/a':
            key2docs[key].add((doc1, doc2))
    bags = list()
    for key, labels in epr2d.items():
        rs = list(labels)
        if 'n/a' in rs and len(rs) > 1:
            rs.remove('n/a')
        bags.append([key, rs, key2docs[key]])
    return bags


def main():
    dev_dataset = json.load(open('rawdata/dev_dataset.json'))
    q2t = json.load(open('q2t.json'))
    t2q = json.load(open('t2q.json'))
    dev_bags = place_data(dev_dataset)
    ret = dict()
    dev_ranks = list()
    for key, rs, docs in tqdm(dev_bags):
        ground_doc_pairs = set([tuple(c) for c in docs])
        h, t = key.split('#')
        docpairs = count_candidates(q2t, t2q, h, t)
        if len(ground_doc_pairs) > 0:
            rank = list()
            for i, c in enumerate(docpairs):
                if (c[0], c[1]) in ground_doc_pairs:
                    rank.append(i + 1)
            while len(rank) < len(ground_doc_pairs):
                rank.append(1000000)
            dev_ranks.append(rank)
        docpairs = [[d[0], d[1]] for d in docpairs[0:16]]
        ret[key] = docpairs
    json.dump(ret, open(f'result-count.json', 'w'))
    json.dump(dev_ranks, open(f'dev-rank-count.json', 'w'))


if __name__ == '__main__':
    main()
