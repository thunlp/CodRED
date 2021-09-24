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
    input_ids = list()
    token_type_ids = list()
    attention_mask = list()
    dplabel = list()
    h_len = tokenizer.max_len_sentences_pair // 2 - 2
    t_len = tokenizer.max_len_sentences_pair - tokenizer.max_len_sentences_pair // 2 - 2
    for example in batch:
        if len(example) == 4:
            ht, doc1, doc2, r = example
            h, t = ht.split('#')
            tokens, token_ids, token_type_id, amask = process_example(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity)
            input_ids.append(token_ids)
            token_type_ids.append(token_type_id)
            attention_mask.append(amask)
            dplabel.append(relation2id[r])
        else:
            doc = json.loads(redisd.get(f'dsre-doc-{example[0]}'))
            _, h_start, h_end, t_start, t_end, r = example
            h_1, h_2 = expand(h_start, h_end, len(doc), h_len)
            t_1, t_2 = expand(t_start, t_end, len(doc), t_len)
            h_tokens = doc[h_1:h_start] + ['[UNUSED1]'] + doc[h_start:h_end] + ['[UNUSED2]'] + doc[h_end:h_2]
            t_tokens = doc[t_1:t_start] + ['[UNUSED3]'] + doc[t_start:t_end] + ['[UNUSED4]'] + doc[t_end:t_2]
            h_token_ids = tokenizer.convert_tokens_to_ids(h_tokens)
            t_token_ids = tokenizer.convert_tokens_to_ids(t_tokens)
            _input_ids = tokenizer.build_inputs_with_special_tokens(h_token_ids, t_token_ids)
            _token_type_ids = tokenizer.create_token_type_ids_from_sequences(h_token_ids, t_token_ids)
            obj = tokenizer._pad({'input_ids': _input_ids, 'token_type_ids': _token_type_ids}, max_length=args.seq_len, padding_strategy='max_length')
            input_ids.append(obj['input_ids'])
            token_type_ids.append(obj['token_type_ids'])
            attention_mask.append(obj['attention_mask'])
            dplabel.append(relation2id[r])
    input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
    token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
    attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)
    dplabel_t = torch.tensor(dplabel, dtype=torch.int64)
    return input_ids_t, token_type_ids_t, attention_mask_t, dplabel_t


def collate_fn_infer(batch, args, relation2id, tokenizer, redisd):
    input_ids = list()
    token_type_ids = list()
    attention_mask = list()
    rs = list()
    hs = list()
    ts = list()
    d1 = list()
    d2 = list()
    for example in batch:
        ht, doc1, doc2, r = example
        h, t = ht.split('#')
        tokens, token_ids, token_type_id, amask = process_example(h, t, doc1, doc2, tokenizer, args.seq_len, redisd, args.no_additional_marker, args.mask_entity)
        input_ids.append(token_ids)
        token_type_ids.append(token_type_id)
        attention_mask.append(amask)
        rs.append(relation2id[r])
        hs.append(h)
        ts.append(t)
        d1.append(doc1)
        d2.append(doc2)
    input_ids_t = torch.tensor(input_ids, dtype=torch.int64)
    token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.int64)
    attention_mask_t = torch.tensor(attention_mask, dtype=torch.int64)
    return input_ids_t, token_type_ids_t, attention_mask_t, rs, hs, ts, d1, d2


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
    
    def forward(self, input_ids, token_type_ids, attention_mask, dplabel=None):
        # input_ids: T(num_sentences, seq_len)
        # token_type_ids: T(num_sentences, seq_len)
        # attention_mask: T(num_sentences, seq_len)
        # rs: T(batch_size)
        # maps: T(batch_size, max_bag_size)
        # embedding: T(num_sentences, seq_len, embedding_size)
        embedding, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        # r_embedding: T(num_sentences, embedding_size)
        r_embedding = embedding[:, 0, :]
        # scores: T(num_sentences, num_relations)
        scores = self.predictor(r_embedding)
        # prediction: T(num_sentences)
        _, prediction = torch.max(scores, dim=1)
        if dplabel is not None:
            loss = self.loss(scores, dplabel)
            return loss, prediction, scores
        else:
            return None, prediction, scores


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
        self.dsre_train_dataset = json.load(open(self.args.dsre_file))
        self.dsre_train_dataset = [d for i, d in enumerate(self.dsre_train_dataset) if i % 10 == 0]
        if self.args.raw_only:
            pass
        elif self.args.dsre_only:
            train_dataset = self.dsre_train_dataset
        else:
            self.dsre_train_dataset.extend(train_dataset)
            tmp = list()
            for i in range(min(len(self.dsre_train_dataset), len(train_dataset))):
                tmp.append(self.dsre_train_dataset[i])
                tmp.append(train_dataset[i])
            if len(self.dsre_train_dataset) > len(train_dataset):
                tmp.extend(self.dsre_train_dataset[len(train_dataset):])
            if len(self.dsre_train_dataset) < len(train_dataset):
                tmp.extend(train_dataset[len(self.dsre_train_dataset):])
            train_dataset = tmp
        self.redisd = redis.Redis(host='localhost', port=6379, decode_responses=True)
        with self.trainer.once():
            self.train_logger = Logger(['train_loss', 'train_acc', 'train_pos_acc', 'train_dsre_acc'], self.trainer.writer, self.args.logging_steps, self.args.local_rank)
            self.dev_logger = Logger(['dev_mean_prec', 'dev_f1', 'dev_auc'], self.trainer.writer, 1, self.args.local_rank)
        return train_dataset, dev_dataset

    def collate_fn(self):
        return partial(collate_fn, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd), partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd), partial(collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer, redisd=self.redisd)

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_step(self, step, train_step, inputs, extra, loss, outputs):
        with self.trainer.once():
            self.train_logger.log(train_loss=loss)
            _, prediction, logit = outputs
            dplabel = inputs['dplabel']
            prediction, dplabel = tensor_to_obj(prediction, dplabel)
            for p, l in zip(prediction, dplabel):
                self.train_logger.log(train_acc=1 if p == l else 0)
                if l > 0:
                    self.train_logger.log(train_pos_acc=1 if p == l else 0)

    def on_train_epoch_end(self, epoch):
        pass

    def on_dev_epoch_start(self, epoch):
        self._prediction = list()

    def on_dev_step(self, step, inputs, extra, outputs):
        _, prediction, logit = outputs
        prediction = tensor_to_obj(prediction)
        hs = extra['hs']
        rs = extra['rs']
        ts = extra['ts']
        d1s = extra['d1']
        d2s = extra['d2']
        for p, h, r, t, d1, d2 in zip(prediction, hs, rs, ts, d1s, d2s):
            self._prediction.append([p, h, r, t, d1, d2])

    def on_dev_epoch_end(self, epoch):
        self._prediction = self.trainer.distributed_broadcast(self._prediction)
        correct = 0
        for p, h, r, t, d1, d2 in self._prediction:
            if r == p:
                correct += 1
        with self.trainer.once():
            json.dump(self._prediction, open(f'output/dev-prediction-{epoch}.json', 'w'))
        return correct / len(self._prediction)

    def process_train_data(self, data):
        inputs = {
            'input_ids': data[0],
            'token_type_ids': data[1],
            'attention_mask': data[2],
            'dplabel': data[3],
        }
        return inputs, None

    def process_dev_data(self, data):
        inputs = {
            'input_ids': data[0],
            'token_type_ids': data[1],
            'attention_mask': data[2]
        }
        return inputs, {'rs': data[3], 'hs': data[4], 'ts': data[5], 'd1': data[6], 'd2': data[7]}


def main():
    trainer = Trainer(CodredCallback())
    trainer.run()


if __name__ == '__main__':
    main()
