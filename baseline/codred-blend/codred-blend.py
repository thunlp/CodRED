import json
import random
from functools import partial

import numpy as np
import redis
import sklearn
import torch
from eveliver import (Logger, Trainer, TrainerCallback, load_model,
                      tensor_to_obj)
from transformers import AutoTokenizer, BertModel


def eval_performance(facts, pred_result):
    sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
    prec = []
    rec = []
    correct = 0
    total = len(facts)
    for i, item in enumerate(sorted_pred_result):
        if (item['entpair'][0], item['entpair'][1], item['relation']) in facts:
            correct += 1
        prec.append(float(correct) / float(i + 1))
        rec.append(float(correct) / float(total))
    auc = sklearn.metrics.auc(x=rec, y=prec)
    np_prec = np.array(prec)
    np_rec = np.array(rec)
    f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
    mean_prec = np_prec.mean()
    return {'prec': np_prec.tolist(), 'rec': np_rec.tolist(), 'mean_prec': mean_prec, 'f1': f1, 'auc': auc}


def expand(start, end, total_len, max_size):
    e_size = max_size - (end - start)
    _1 = start - (e_size // 2)
    _2 = end + (e_size - e_size // 2)
    if _2 - _1 <= total_len:
        if _1 < 0:
            _2 -= -1
            _1 = 0
        elif _2 > total_len:
            _1 -= (_2 - total_len)
            _2 = total_len
    else:
        _1 = 0
        _2 = total_len
    return _1, _2


def place_train_data(dataset):
    ep2d = dict()
    for key, doc1, doc2, label in dataset:
        if key not in ep2d:
            ep2d[key] = dict()
        if label not in ep2d[key]:
            ep2d[key][label] = list()
        ep2d[key][label].append([doc1, doc2, label])
    bags = list()
    for key, l2docs in ep2d.items():
        if len(l2docs) == 1 and 'n/a' in l2docs:
            bags.append([key, 'n/a', l2docs['n/a'], 'o'])
        else:
            labels = list(l2docs.keys())
            for label in labels:
                if label != 'n/a':
                    ds = l2docs[label]
                    if 'n/a' in l2docs:
                        ds.extend(l2docs['n/a'])
                    bags.append([key, label, ds, 'o'])
    bags.sort(key=lambda x: x[0] + '#' + x[1])
    return bags


def place_dev_data(dataset, single_path):
    ep2d = dict()
    for key, doc1, doc2, label in dataset:
        if key not in ep2d:
            ep2d[key] = dict()
        if label not in ep2d[key]:
            ep2d[key][label] = list()
        ep2d[key][label].append([doc1, doc2, label])
    bags = list()
    for key, l2docs in ep2d.items():
        if len(l2docs) == 1 and 'n/a' in l2docs:
            bags.append([key, ['n/a'], l2docs['n/a'], 'o'])
        else:
            labels = list(l2docs.keys())
            ds = list()
            for label in labels:
                if single_path and label != 'n/a':
                    ds.append(random.choice(l2docs[label]))
                else:
                    ds.extend(l2docs[label])
            if 'n/a' in labels:
                labels.remove('n/a')
            bags.append([key, labels, ds, 'o'])
    bags.sort(key=lambda x: x[0] + '#' + '#'.join(x[1]))
    return bags


def gen_c(tokenizer, passage, span, max_len, bound_tokens, d_start, d_end, no_additional_marker, mask_entity):
    ret = list()
    ret.append(bound_tokens[0])
    for i in range(span[0], span[1]):
        if mask_entity:
            ret.append('[MASK]')
        else:
            ret.append(passage[i])
    ret.append(bound_tokens[1])
    prev = list()
    prev_ptr = span[0] - 1
    while len(prev) < max_len:
        if prev_ptr < 0:
            break
        if not no_additional_marker and prev_ptr in d_end:
            prev.append(f'[UNUSED{(d_end[prev_ptr] + 2) * 2 + 2}]')
        prev.append(passage[prev_ptr])
        if not no_additional_marker and prev_ptr in d_start:
            prev.append(f'[UNUSED{(d_start[prev_ptr] + 2) * 2 + 1}]')
        prev_ptr -= 1
    nex = list()
    nex_ptr = span[1]
    while len(nex) < max_len:
        if nex_ptr >= len(passage):
            break
        if not no_additional_marker and nex_ptr in d_start:
            nex.append(f'[UNUSED{(d_start[nex_ptr] + 2) * 2 + 1}]')
        nex.append(passage[nex_ptr])
        if not no_additional_marker and nex_ptr in d_end:
            nex.append(f'[UNUSED{(d_end[nex_ptr] + 2) * 2 + 2}]')
        nex_ptr += 1
    pn = max_len - len(ret)
    if len(prev) + len(nex) > pn:
        if len(prev) > pn / 2 and len(nex) > pn / 2:
            prev = prev[0:pn // 2]
            nex = nex[0:pn - pn // 2]
        elif len(prev) <= len(nex):
            nex = nex[0:pn - len(prev)]
        elif len(nex) < len(prev):
            prev = prev[0:pn - len(nex)]
    prev.reverse()
    ret = prev + ret + nex
    return ret


def process_example(h, t, doc1, doc2, tokenizer, max_len, redisd, no_additional_marker, mask_entity):
    doc1 = json.loads(redisd.get('codred-doc-' + doc1))
    doc2 = json.loads(redisd.get('codred-doc-' + doc2))
    v_h = None
    for entity in doc1['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == h and v_h is None:
            v_h = entity
    assert v_h is not None
    v_t = None
    for entity in doc2['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == t and v_t is None:
            v_t = entity
    assert v_t is not None
    d1_v = dict()
    for entity in doc1['entities']:
        if 'Q' in entity:
            d1_v[entity['Q']] = entity
    d2_v = dict()
    for entity in doc2['entities']:
        if 'Q' in entity:
            d2_v[entity['Q']] = entity
    ov = set(d1_v.keys()) & set(d2_v.keys())
    if len(ov) > 40:
        ov = set(random.choices(list(ov), k=40))
    ov = list(ov)
    ma = dict()
    for e in ov:
        ma[e] = len(ma)
    d1_start = dict()
    d1_end = dict()
    for entity in doc1['entities']:
        if 'Q' in entity and entity['Q'] in ma:
            for span in entity['spans']:
                d1_start[span[0]] = ma[entity['Q']]
                d1_end[span[1] - 1] = ma[entity['Q']]
    d2_start = dict()
    d2_end = dict()
    for entity in doc2['entities']:
        if 'Q' in entity and entity['Q'] in ma:
            for span in entity['spans']:
                d2_start[span[0]] = ma[entity['Q']]
                d2_end[span[1] - 1] = ma[entity['Q']]
    k1 = gen_c(tokenizer, doc1['tokens'], v_h['spans'][0], max_len // 2 - 2, ['[UNUSED1]', '[UNUSED2]'], d1_start, d1_end, no_additional_marker, mask_entity)
    k2 = gen_c(tokenizer, doc2['tokens'], v_t['spans'][0], max_len // 2 - 1, ['[UNUSED3]', '[UNUSED4]'], d2_start, d2_end, no_additional_marker, mask_entity)
    tokens = ['[CLS]'] + k1 + ['[SEP]'] + k2 + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(token_ids) < max_len:
        token_ids = token_ids + [0] * (max_len - len(tokens))
    attention_mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
    token_type_id = [0] * (len(k1) + 2) + [1] * (len(k2) + 1) + [0] * (max_len - len(tokens))
    return tokens, token_ids, token_type_id, attention_mask


def collate_fn(batch, args, relation2id, tokenizer, redisd):
    assert len(batch) == 1
    if batch[0][-1] == 'o':
        batch = batch[0]
        h, t = batch[0].split('#')
        r = relation2id[batch[1]]
        dps = batch[2]
        if len(dps) > 8:
            dps = random.choices(dps, k=8)
        input_ids = list()
        token_type_ids = list()
        attention_mask = list()
        dplabel = list()
        for doc1, doc2, l in dps:
            tokens, token_ids, token_type_id, amask = process_example(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity)
            input_ids.append(token_ids)
            token_type_ids.append(token_type_id)
            attention_mask.append(amask)
            dplabel.append(relation2id[l])
        input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
        token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)
        dplabel_t = torch.tensor(dplabel, dtype=torch.int64)
        rs_t = torch.tensor([r], dtype=torch.int64)
    else:
        examples = batch[0]
        h_len = tokenizer.max_len_sentences_pair // 2 - 2
        t_len = tokenizer.max_len_sentences_pair - tokenizer.max_len_sentences_pair // 2 - 2
        _input_ids = list()
        _token_type_ids = list()
        _attention_mask = list()
        _rs = list()
        for idx, example in enumerate(examples):
            doc = json.loads(redisd.get(f'dsre-doc-{example[0]}'))
            _, h_start, h_end, t_start, t_end, r = example
            if r in relation2id:
                r = relation2id[r]
            else:
                r = 'n/a'
            h_1, h_2 = expand(h_start, h_end, len(doc), h_len)
            t_1, t_2 = expand(t_start, t_end, len(doc), t_len)
            h_tokens = doc[h_1:h_start] + ['[UNUSED1]'] + doc[h_start:h_end] + ['[UNUSED2]'] + doc[h_end:h_2]
            t_tokens = doc[t_1:t_start] + ['[UNUSED3]'] + doc[t_start:t_end] + ['[UNUSED4]'] + doc[t_end:t_2]
            h_token_ids = tokenizer.convert_tokens_to_ids(h_tokens)
            t_token_ids = tokenizer.convert_tokens_to_ids(t_tokens)
            input_ids = tokenizer.build_inputs_with_special_tokens(h_token_ids, t_token_ids)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(h_token_ids, t_token_ids)
            obj = tokenizer._pad({'input_ids': input_ids, 'token_type_ids': token_type_ids}, max_length=args.seq_len, padding_strategy='max_length')
            _input_ids.append(obj['input_ids'])
            _token_type_ids.append(obj['token_type_ids'])
            _attention_mask.append(obj['attention_mask'])
            _rs.append(r)
        input_ids_t = torch.tensor(_input_ids, dtype=torch.long)
        token_type_ids_t = torch.tensor(_token_type_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(_attention_mask, dtype=torch.long)
        dplabel_t = torch.tensor(_rs, dtype=torch.long)
        rs_t = None
        r = None
    return input_ids_t, token_type_ids_t, attention_mask_t, dplabel_t, rs_t, [r]


def collate_fn_infer(batch, args, relation2id, tokenizer, redisd):
    assert len(batch) == 1
    batch = batch[0]
    h, t = batch[0].split('#')
    rs = [relation2id[r] for r in batch[1]]
    dps = batch[2]
    input_ids = list()
    token_type_ids = list()
    attention_mask = list()
    for doc1, doc2, l in dps:
        tokens, token_ids, token_type_id, amask = process_example(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity)
        input_ids.append(token_ids)
        token_type_ids.append(token_type_id)
        attention_mask.append(amask)
    input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
    token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
    attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)
    return input_ids_t, token_type_ids_t, attention_mask_t, h, rs, t


class Codred(torch.nn.Module):
    def __init__(self, args, num_relations):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.predictor = torch.nn.Linear(self.bert.config.hidden_size, num_relations)
        weight = torch.ones(num_relations, dtype=torch.float32)
        weight[0] = 0.1
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=weight)
        self.aggregator = args.aggregator
        self.no_doc_pair_supervision = args.no_doc_pair_supervision
    
    def forward(self, input_ids, token_type_ids, attention_mask, dplabel=None, rs=None):
        # input_ids: T(num_sentences, seq_len)
        # token_type_ids: T(num_sentences, seq_len)
        # attention_mask: T(num_sentences, seq_len)
        # rs: T(batch_size)
        # maps: T(batch_size, max_bag_size)
        # embedding: T(num_sentences, seq_len, embedding_size)
        embedding, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        # r_embedding: T(num_sentences, embedding_size)
        r_embedding = embedding[:, 0, :]
        # logit: T(1, num_relations)
        # dp_logit: T(num_sentences, num_relations)
        logit, dp_logit = self.predict_logit(r_embedding, rs=rs)
        # prediction: T(1)
        _, prediction = torch.max(logit, dim=1)
        if dplabel is not None and rs is None:
            loss = self.loss(dp_logit, dplabel)
            # prediction: T(num_sentences)
            _, prediction = torch.max(dp_logit, dim=1)
        elif rs is not None:
            if self.no_doc_pair_supervision:
                loss = self.loss(logit, rs)
            else:
                loss = self.loss(logit, rs) + self.loss(dp_logit, dplabel)
        else:
            loss = None
        return loss, prediction, logit
    
    def predict_logit(self, r_embedding, rs=None):
        # r_embedding: T(num_sentences, embedding_size)
        # weight: T(num_relations, embedding_size)
        weight = self.predictor.weight
        if self.aggregator == 'max':
            # scores: T(num_sentences, num_relations)
            scores = self.predictor(r_embedding)
            # prob: T(num_sentences, num_relations)
            prob = torch.nn.functional.softmax(scores, dim=1)
            if rs is not None:
                _, idx = torch.max(prob[:, rs[0]], dim=0, keepdim=True)
                return scores[idx], scores
            else:
                # max_score: T(1, num_relations)
                max_score, _ = torch.max(scores, dim=0, keepdim=True)
                return max_score, scores
        elif self.aggregator == 'avg':
            # embedding: T(1, embedding_size)
            embedding = torch.sum(r_embedding, dim=1, keepdim=True) / r_embedding.shape[0]
            return self.predictor(embedding), self.predictor(r_embedding)
        elif self.aggregator == 'attention':
            # attention_score: T(num_sentences, num_relations)
            attention_score = torch.matmul(r_embedding, torch.t(weight))
            # attention_weight: T(num_sentences, num_relations)
            attention_weight = torch.nn.functional.softmax(attention_score, dim=0)
            # embedding: T(num_relations, embedding_size)
            embedding = torch.matmul(torch.transpose(attention_weight, 0, 1), r_embedding)
            # logit: T(num_relations, num_relations)
            logit = self.predictor(embedding)
            return torch.diag(logit).unsqueeze(0), self.predictor(r_embedding)
        else:
            assert False


class CodredCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_argument(self, parser):
        parser.add_argument('--seq_len', type=int, default=512)
        parser.add_argument('--aggregator', type=str, default='attention')
        parser.add_argument('--positive_only', action='store_true')
        parser.add_argument('--positive_ep_only', action='store_true')
        parser.add_argument('--no_doc_pair_supervision', action='store_true')
        parser.add_argument('--no_additional_marker', action='store_true')
        parser.add_argument('--mask_entity', action='store_true')
        parser.add_argument('--single_path', action='store_true')
        parser.add_argument('--dsre_only', action='store_true')
        parser.add_argument('--raw_only', action='store_true')
        parser.add_argument('--load_model_path', type=str, default=None)
        parser.add_argument('--train_file', type=str, default='../data/rawdata/train_dataset.json')
        parser.add_argument('--dev_file', type=str, default='../data/rawdata/dev_dataset.json')
        parser.add_argument('--dsre_file', type=str, default='../data/dsre_train_examples.json')

    def load_model(self):
        relations = json.load(open('../../../data/rawdata/relations.json'))
        relations.sort()
        self.relations = ['n/a'] + relations
        self.relation2id = dict()
        for index, relation in enumerate(self.relations):
            self.relation2id[relation] = index
        with self.trainer.cache():
            model = Codred(self.args, len(self.relations))
            if self.args.load_model_path:
                load_model(model, self.args.load_model_path)
            tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
        self.tokenizer = tokenizer
        return model

    def load_data(self):
        train_dataset = json.load(open(self.args.train_file))
        dev_dataset = json.load(open(self.args.dev_file))
        if self.args.positive_only:
            train_dataset = [d for d in train_dataset if d[3] != 'n/a']
            dev_dataset = [d for d in dev_dataset if d[3] != 'n/a']
        train_bags = place_train_data(train_dataset)
        dev_bags = place_dev_data(dev_dataset, self.args.single_path)
        if self.args.positive_ep_only:
            train_bags = [b for b in train_bags if b[1] != 'n/a']
            dev_bags = [b for b in dev_bags if 'n/a' not in b[1]]
        self.dsre_train_dataset = json.load(open(self.args.dsre_file))
        self.dsre_train_dataset = [d for i, d in enumerate(self.dsre_train_dataset) if i % 10 == 0]
        d = list()
        for i in range(len(self.dsre_train_dataset) // 8):
            d.append(self.dsre_train_dataset[8 * i:8 * i + 8])
        if self.args.raw_only:
            pass
        elif self.args.dsre_only:
            train_bags = d
        else:
            d.extend(train_bags)
            train_bags = d
        self.redisd = redis.Redis(host='localhost', port=6379, decode_responses=True)
        with self.trainer.once():
            self.train_logger = Logger(['train_loss', 'train_acc', 'train_pos_acc', 'train_dsre_acc'], self.trainer.writer, self.args.logging_steps, self.args.local_rank)
            self.dev_logger = Logger(['dev_mean_prec', 'dev_f1', 'dev_auc'], self.trainer.writer, 1, self.args.local_rank)
        return train_bags, dev_bags

    def collate_fn(self):
        return partial(collate_fn, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd), partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd), partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd)

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_step(self, step, train_step, inputs, extra, loss, outputs):
        with self.trainer.once():
            self.train_logger.log(train_loss=loss)
            if inputs['rs'] is not None:
                _, prediction, logit = outputs
                rs = extra['rs']
                prediction, logit = tensor_to_obj(prediction, logit)
                for p, score, gold in zip(prediction, logit, rs):
                    self.train_logger.log(train_acc=1 if p == gold else 0)
                    if gold > 0:
                        self.train_logger.log(train_pos_acc=1 if p == gold else 0)
            else:
                _, prediction, logit = outputs
                dplabel = inputs['dplabel']
                prediction, logit, dplabel = tensor_to_obj(prediction, logit, dplabel)
                for p, l in zip(prediction, dplabel):
                    self.train_logger.log(train_dsre_acc=1 if p == l else 0)

    def on_train_epoch_end(self, epoch):
        pass

    def on_dev_epoch_start(self, epoch):
        self._prediction = list()

    def on_dev_step(self, step, inputs, extra, outputs):
        _, prediction, logit = outputs
        h, t, rs = extra['h'], extra['t'], extra['rs']
        prediction, logit = tensor_to_obj(prediction, logit)
        self._prediction.append([prediction[0], logit[0], h, t, rs])

    def on_dev_epoch_end(self, epoch):
        self._prediction = self.trainer.distributed_broadcast(self._prediction)
        results = list()
        pred_result = list()
        facts = dict()
        for p, score, h, t, rs in self._prediction:
            rs = [self.relations[r] for r in rs]
            for i in range(1, len(score)):
                pred_result.append({'entpair': [h, t], 'relation': self.relations[i], 'score': score[i]})
            results.append([h, rs, t, self.relations[p]])
            for r in rs:
                if r != 'n/a':
                    facts[(h, t, r)] = 1
        stat = eval_performance(facts, pred_result)
        with self.trainer.once():
            self.dev_logger.log(dev_mean_prec=stat['mean_prec'], dev_f1=stat['f1'], dev_auc=stat['auc'])
            json.dump(stat, open(f'output/dev-stat-{epoch}.json', 'w'))
            json.dump(results, open(f'output/dev-results-{epoch}.json', 'w'))
        return stat['f1']

    def process_train_data(self, data):
        inputs = {
            'input_ids': data[0],
            'token_type_ids': data[1],
            'attention_mask': data[2],
            'dplabel': data[3],
            'rs': data[4]
        }
        return inputs, {'rs': data[5]}

    def process_dev_data(self, data):
        inputs = {
            'input_ids': data[0],
            'token_type_ids': data[1],
            'attention_mask': data[2]
        }
        return inputs, {'h': data[3], 'rs': data[4], 't': data[5]}


def main():
    trainer = Trainer(CodredCallback())
    trainer.run()


if __name__ == '__main__':
    main()
