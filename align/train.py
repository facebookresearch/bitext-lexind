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
from glob import glob
from chinese_converter import to_traditional, to_simplified
from tqdm import tqdm

from models import CRISSWrapper, WordAligner
from data import BitextAlignmentDataset


cos = torch.nn.CosineSimilarity(dim=-1)

def setup_configs(configs):
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
                align = [tuple(x if not is_reversed else reversed(x)) for x in json.loads(aligns[i])['inter']] 
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


def extract_scores(batch, criss_features, aligner, info, configs):
    coocc, semi_matched_coocc, matched_coocc, freq_src, freq_trg = info
    all_scores = list()
    for i in range(0, len(batch), configs.batch_size):
        subbatch = batch[i:i+configs.batch_size]
        src_words, trg_words = zip(*subbatch)
        features = torch.tensor(
            [
                [
                    matched_coocc[x[0]][x[1]],
                    semi_matched_coocc[x[0]][x[1]],
                    coocc[x[0]][x[1]],
                    freq_src[x[0]],
                    freq_trg[x[1]]
                ] for x in subbatch
            ]
        ).float().to(configs.device).reshape(-1, 5)
        if configs.use_criss:
            subbatch_crissfeat = torch.cat(criss_features[i:i+configs.batch_size], dim=0)
            features = torch.cat((subbatch_crissfeat, features), dim=-1).detach()
        scores = aligner(features).squeeze(-1)
        all_scores.append(scores)
    return torch.cat(all_scores, dim=0)


def train(configs, logging_steps=50000):
    setup_configs(configs)
    os.system(f'mkdir -p {configs.save_path}')
    torch.save(configs, configs.save_path + '/configs.pt')
    info = collect_bitext_stats(
        configs.bitext_path, configs.align_path, configs.save_path, 
        configs.src_lang, configs.trg_lang, configs.reversed
    )
    if configs.use_criss:
        criss = CRISSWrapper(device=configs.device)
    else:
        criss = None
    dataset = BitextAlignmentDataset(configs.bitext_path, configs.align_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    aligner = WordAligner(5 + (2 if configs.use_criss else 0), configs.hiddens, 3, 5).to(configs.device)
    optimizer = torch.optim.Adam(aligner.parameters(), lr=.0005)
    for epoch in range(configs.epochs):
        model_cnt = 0
        total_loss = total_cnt = 0
        bar = tqdm(dataloader)
        for idx, batch in enumerate(bar):
            (ss, ts), edges = batch[0]
            if criss is not None:
                semb = criss.embed(ss, langcode=configs.src_lang)
                temb = criss.embed(ts, langcode=configs.trg_lang)
                cos_matrix = cos(semb.unsqueeze(1), temb.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)
                ip_matrix = (semb.unsqueeze(1) * temb.unsqueeze(0)).sum(-1).unsqueeze(-1).unsqueeze(-1)
                feat_matrix = torch.cat((cos_matrix, ip_matrix), dim=-1)
            # adding contexualized embeddings here
            training_sets = collections.defaultdict(list)
            criss_features = collections.defaultdict(list)
            for i, sw in enumerate(ss):
                for j, tw in enumerate(ts):
                    label = edges[i, j]
                    training_sets[label].append((sw, tw))
                    if criss is not None:
                        criss_features[label].append(feat_matrix[i, j])
            max_len = max(len(training_sets[k]) for k in training_sets)
            training_set = list()
            criss_feats = list()
            targets = list()
            for key in training_sets:
                training_set += training_sets[key] * (max_len // len(training_sets[key]))
                criss_feats += criss_features[key] * (max_len // len(training_sets[key]))
                targets += [key] * len(training_sets[key]) * (max_len // len(training_sets[key]))
            targets = torch.tensor(targets).long().to(configs.device)
            scores = extract_scores(training_set, criss_feats, aligner, info, configs)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(scores, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_cnt += len(batch)
            bar.set_description(f'loss={total_loss / total_cnt:.5f}')
            if (idx + 1) % logging_steps == 0:
                print(f'Epoch {epoch}, step {idx+1}, loss = {total_loss / total_cnt:.5f}', flush=True)
    torch.save(aligner.state_dict(), configs.save_path + f'/model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--align', type=str, help='path to word alignment')
    parser.add_argument('-b', '--bitext', type=str, help='path to bitext')
    parser.add_argument('-src', '--source', type=str, help='source language code')
    parser.add_argument('-trg', '--target', type=str, help='target language code')
    parser.add_argument('-o', '--output', type=str, default='./model/', help='path to output folder')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device for training [cuda|cpu]')
    args = parser.parse_args()

    configs = dotdict.DotDict(
        {
            'align_path': args.align, 
            'bitext_path': args.bitext,
            'save_path': args.output,
            'batch_size': 128,
            'epochs': 100,
            'device': args.device,
            'hiddens': [8],
            'use_criss': True,
            'src_lang': args.source,
            'trg_lang': args.target
        }
    )

    train(configs)
