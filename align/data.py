# Copyright (c) Facebook, Inc. and its affiliates.

from torch.utils.data import DataLoader, Dataset
import regex
import json
import numpy as np
import os


class BitextAlignmentDataset(Dataset):
    def __init__(self, bitext_path, alignment_path):
        super(BitextAlignmentDataset, self).__init__()
        self.bitext_path = bitext_path
        self.alignment_path = alignment_path
        bitext = [regex.split(r'\|\|\|', x.strip()) for x in open(bitext_path)]
        align = open(alignment_path).readlines()
        self.bitext, self.edges = self.filter(bitext, align)
        assert len(self.bitext) == len(self.edges)

    @staticmethod
    def filter(bitext, align):
        real_bitext = list()
        edges = list()
        for i, a in enumerate(align):
            try:
                a = json.loads(a)
                if len(bitext[i]) == 2:
                    bitext[i][0] = bitext[i][0].split()
                    bitext[i][1] = bitext[i][1].split()
                    real_bitext.append(bitext[i])
                    edge_info = np.zeros((len(bitext[i][0]), len(bitext[i][1])))
                    for x, y in a['inter']:
                        edge_info[x, y] = 2
                    for x, y in a['itermax']:
                        if edge_info[x, y] == 0:
                            edge_info[x, y] = 1
                    edges.append(edge_info)
            except:
                continue
        return real_bitext, edges
    
    def __getitem__(self, index):
        return self.bitext[index], self.edges[index]
    
    def __len__(self):
        return len(self.bitext)

    @staticmethod
    def collate_fn(batch):
        return batch
    

class AlignDataset(object):
    def __init__(self, path, langs, split='test'):
        if langs == 'de-en':
            src_sents = [x.strip() for x in open(os.path.join(path, langs, 'de'), encoding='iso-8859-1').readlines()][:-1]
            trg_sents = [x.strip() for x in open(os.path.join(path, langs, 'en'), encoding='iso-8859-1').readlines()][:-1]
            self.ground_truth = self.load_std_file(os.path.join(path, langs, 'alignmentDeEn.talp'))[:-1]
        elif langs == 'ro-en' or langs == 'en-fr':
            src_id2s = dict()
            trg_id2s = dict()
            for fpair in open(os.path.join(path, langs, split, f'FilePairs.{split}')):
                sf, tf = fpair.strip().split()
                for line in open(os.path.join(path, langs, split, sf), encoding='iso-8859-1'):
                    matching = regex.match(r'<s snum=([0-9]*)>(.*)</s>', line.strip())
                    assert matching is not None
                    idx = matching.group(1)
                    sent = matching.group(2).strip()
                    src_id2s[idx] = sent
                for line in open(os.path.join(path, langs, split, tf), encoding='iso-8859-1'):
                    matching = regex.match(r'<s snum=([0-9]*)>(.*)</s>', line.strip())
                    assert matching is not None
                    idx = matching.group(1)
                    sent = matching.group(2).strip()
                    trg_id2s[idx] = sent
            src_sents = [src_id2s[key] for key in sorted(src_id2s.keys())]
            trg_sents = [trg_id2s[key] for key in sorted(trg_id2s.keys())]
            snum2idx = dict([(key, i) for i, key in enumerate(sorted(trg_id2s.keys()))])
            assert len(src_id2s) == len(trg_id2s)
            ground_truth = [list() for _ in src_id2s]
            raw_gt = open(os.path.join(path, langs, split, f'{split}.wa.nonullalign')).readlines()
            for line in raw_gt:
                sid, s, t, sure = line.strip().split()
                idx = snum2idx[sid]
                if sure == 'S':
                    align = '-'.join([s, t])
                else:
                    assert sure == 'P'
                    align = 'p'.join([s, t])
                ground_truth[idx].append(align)
            for i, item in enumerate(ground_truth):
                ground_truth[i] = ' '.join(item)
            self.ground_truth = ground_truth
        elif langs == 'en-hi':
            src_id2s = dict()
            trg_id2s = dict()
            sf = f'{split}.e'
            tf = f'{split}.h'
            for line in open(os.path.join(path, langs, split, sf), encoding='us-ascii'):
                matching = regex.match(r'<s snum=([0-9]*)>(.*)</s>', line.strip())
                assert matching is not None
                idx = matching.group(1)
                sent = matching.group(2).strip()
                src_id2s[idx] = sent
            for line in open(os.path.join(path, langs, split, tf), encoding='utf-8'):
                matching = regex.match(r'<s snum=([0-9]*)>(.*)</s>', line.strip())
                assert matching is not None
                idx = matching.group(1)
                sent = matching.group(2).strip()
                trg_id2s[idx] = sent
            src_sents = [src_id2s[key] for key in sorted(src_id2s.keys())]
            trg_sents = [trg_id2s[key] for key in sorted(trg_id2s.keys())]
            snum2idx = dict([(key, i) for i, key in enumerate(sorted(trg_id2s.keys()))])
            assert len(src_id2s) == len(trg_id2s)
            ground_truth = [list() for _ in src_id2s]
            raw_gt = open(os.path.join(path, langs, split, f'{split}.wa.nonullalign')).readlines()
            for line in raw_gt:
                sid, s, t = line.strip().split()
                idx = snum2idx[sid]
                align = '-'.join([s, t])
                ground_truth[idx].append(align)
            for i, item in enumerate(ground_truth):
                ground_truth[i] = ' '.join(item)
            self.ground_truth = ground_truth
        else:
            raise Exception('language pair not supported.')
        self.sent_pairs = list(zip(src_sents, trg_sents))
        assert len(self.sent_pairs) == len(self.ground_truth)

    @staticmethod
    def load_std_file(path):
        return [x.strip() for x in open(path)]

