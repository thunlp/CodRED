import json
from collections import defaultdict


def main():
    overlap = json.load(open('result-count.json'))
    dev_dataset = json.load(open('rawdata/dev_dataset.json'))
    dev_ep2r = defaultdict(list)
    for ep, _, _, r in dev_dataset:
        dev_ep2r[ep].append(r)
    dev_data = list()
    for ep, r in dev_ep2r.items():
        if 'n/a' in r and len(r) > 1:
            r.remove('n/a')
        for i, dps in enumerate(overlap[ep]):
            dev_data.append([ep, dps[0], dps[1], r[i % len(r)]])
    json.dump(dev_data, open('open_dev_data.json', 'w'))


if __name__ == '__main__':
    main()
