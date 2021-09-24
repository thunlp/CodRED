import json

import redis
from tqdm import tqdm


def main():
    q2t = dict()
    t2q = dict()
    redisd = redis.Redis(host='localhost', port=6379, decode_responses=True)
    titles = set()
    popular_docs = json.load(open('popular_docs.json'))
    for _, title in popular_docs:
        titles.add(title)
    all_docs = json.load(open('all_docs.json'))
    for _, title in popular_docs:
        if title not in titles:
            titles.add(title)
    for title in tqdm(titles):
        doc = json.loads(redisd.get(f'codred-doc-{title}'))
        for entity in doc['entities']:
            if 'Q' in entity:
                name = 'Q' + str(entity['Q'])
                if name not in q2t:
                    q2t[name] = dict()
                q2t[name][title] = len(entity)
                if title not in t2q:
                    t2q[title] = dict()
                t2q[title][name] = len(entity)
    json.dump(q2t, open('q2t.json', 'w'))
    json.dump(t2q, open('t2q.json', 'w'))


if __name__ == '__main__':
    main()
