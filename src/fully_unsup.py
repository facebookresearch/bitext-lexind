# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import collections
import copy
import dotdict
import json
import numpy as np
import os
import random
import regex
import tempfile
import torch
import torch.nn as nn
from chinese_converter import to_traditional, to_simplified
from tqdm import tqdm

from evaluate import evaluate
from models import CRISSWrapper, LexiconInducer


cos = nn.CosineSimilarity(dim=-1)


def setup_configs(configs):
    configs.save_path = configs.save_path.format(src=configs.src_lang, trg=configs.trg_lang)
    configs.stats_path = configs.save_path + '/stats.pt'


def collect_bitext_stats(bitext_path, align_path, save_path, src_lang, trg_lang, is_reversed=False):
    stats_path = save_path + '/stats.pt'
    freq_path = save_path + '/freqs.pt'
    if os.path.exists(stats_path):
        coocc, semi_matched_coocc, matched_coocc = torch.load(stats_path)
    else:
        coocc = collections.defaultdict(collections.Counter)
        semi_matched_coocc = collections.defaultdict(collections.Counter)
        matched_coocc = collections.defaultdict(collections.Counter)
        tmpdir = tempfile.TemporaryDirectory()
        os.system(f'cat {bitext_path} > {tmpdir.name}/bitext.txt')
        os.system(f'cat {align_path} > {tmpdir.name}/aligns.txt')
        bitext = open(f'{tmpdir.name}/bitext.txt').readlines()
        aligns = open(f'{tmpdir.name}/aligns.txt').readlines()
        tmpdir.cleanup()
        assert len(bitext) == len(aligns)
        bar = tqdm(bitext)
        for i, item in enumerate(bar):
            try:
                src_sent, trg_sent = regex.split(r'\|\|\|', item.strip())
                if is_reversed:
                    src_sent, trg_sent = trg_sent, src_sent
                align = [tuple(x if not is_reversed else reversed(x)) for x in json.loads(aligns[i])['inter']]  # only focus on inter based alignment
            except:
                continue
            if src_lang == 'zh_CN':
                src_sent = to_simplified(src_sent)
            if trg_lang == 'zh_CN':
                trg_sent = to_simplified(trg_sent)
            src_words = src_sent.lower().split()
            trg_words = trg_sent.lower().split()
            src_cnt = collections.Counter([x[0] for x in align])
            trg_cnt = collections.Counter([x[1] for x in align])
            for x, sw in enumerate(src_words):
                for y, tw in enumerate(trg_words):
                    if (x, y) in align:
                        semi_matched_coocc[sw][tw] += 1
                        if src_cnt[x] == 1 and trg_cnt[y] == 1:
                            matched_coocc[sw][tw] += 1
                    coocc[sw][tw] += 1
        torch.save((coocc, semi_matched_coocc, matched_coocc), stats_path)
    if os.path.exists(freq_path):
        freq_src, freq_trg = torch.load(freq_path)
    else:
        freq_src = collections.Counter()
        freq_trg = collections.Counter()
        tmpdir = tempfile.TemporaryDirectory()
        os.system(f'cat {bitext_path} > {tmpdir.name}/bitext.txt')
        bitext = open(f'{tmpdir.name}/bitext.txt').readlines()
        tmpdir.cleanup()
        bar = tqdm(bitext)
        for i, item in enumerate(bar):
            try:
                src_sent, trg_sent = regex.split(r'\|\|\|', item.strip())
                if is_reversed:
                    src_sent, trg_sent = trg_sent, src_sent
            except:
                continue
            if src_lang == 'zh_CN':
                src_sent = to_simplified(src_sent)
            if trg_lang == 'zh_CN':
                trg_sent = to_simplified(trg_sent)
            for w in src_sent.split():
                freq_src[w] += 1
            for w in trg_sent.split():
                freq_trg[w] += 1
        torch.save((freq_src, freq_trg), freq_path)
    return coocc, semi_matched_coocc, matched_coocc, freq_src, freq_trg
    

def load_lexicon(path):
    lexicon = [regex.split(r'\t| ', x.strip()) for x in open(path)]
    return set([tuple(x) for x in lexicon])


def get_test_lexicon(test_lexicon, info):
    induced_lexicon = list()
    coocc, semi_matched_coocc, matched_coocc, freq_src, freq_trg = info
    for tsw in tqdm(set([x[0] for x in test_lexicon])):
        ssw = to_simplified(tsw)
        candidates = list()
        for stw in matched_coocc[ssw]:
            ttw = to_traditional(stw)
            candidates.append([tsw, ttw, matched_coocc[ssw][stw] / (coocc[ssw][stw] + 20)])
        if len(candidates) == 0:
            continue
        candidates = sorted(candidates, key=lambda x:-x[-1])
        induced_lexicon.append(candidates[0][:2])
    eval_result = evaluate(induced_lexicon, test_lexicon)
    return induced_lexicon, eval_result


def test(configs, logging_steps=50000):
    setup_configs(configs)
    # prepare feature extractor
    info = collect_bitext_stats(
        configs.bitext_path, configs.align_path, configs.save_path, configs.src_lang, configs.trg_lang, configs.reversed
    ) 
    # dataset
    test_lexicon = load_lexicon(configs.test_set)
    induced_test_lexicon, test_eval = get_test_lexicon(test_lexicon, info)
    with open(configs.save_path + '/induced.fullyunsup.dict', 'w') as fout:
        for item in induced_test_lexicon:
            fout.write('\t'.join([str(x) for x in item]) + '\n')
        fout.close()
    return induced_test_lexicon, test_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--align', type=str, help='path to word alignment')
    parser.add_argument('-b', '--bitext', type=str, help='path to bitext')
    parser.add_argument('-src', '--source', type=str, help='source language code')
    parser.add_argument('-trg', '--target', type=str, help='target language code')
    parser.add_argument('-te', '--test', type=str, help='path to test lexicon')
    parser.add_argument('-o', '--output', type=str, default='./model/', help='path to output folder')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device for training [cuda|cpu]')
    args = parser.parse_args()

    configs = dotdict.DotDict(
        {
            'test_set': args.test, 
            'align_path': args.align,
            'bitext_path': args.bitext,
            'save_path': args.output,
            'batch_size': 128,
            'epochs': 50,
            'device': args.device,
            'hiddens': [8]
        }
    )
    res = test(configs)
    print(res[-1])