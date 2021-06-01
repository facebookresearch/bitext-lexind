# Copyright (c) Facebook, Inc. and its affiliates.

import networkx as nx
import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer
import regex
import collections
from glob import glob


class CRISSAligner(object):
    def __init__(self, path='criss/criss-3rd.pt',
                 args_path='criss/args.pt',
                 tokenizer='facebook/mbart-large-cc25', device='cpu', distortion=0,
                 matching_method='a'
            ):
        from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
        from fairseq.sequence_generator import EnsembleModel
        self.device = device
        args = torch.load(args_path)
        task = tasks.setup_task(args)
        models, _model_args = checkpoint_utils.load_model_ensemble(
            path.split(':'),
            arg_overrides=eval('{}'),
            task=task
        )
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            model = model.to(self.device)
        self.model = EnsembleModel(models).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.distortion = distortion
        self.matching_method = matching_method

    def get_embed(self, bpe_lists, langcodes=('en_XX', 'en_XX')):
        vectors = list()
        for i, bpe_list in enumerate(bpe_lists):
            input_ids = self.tokenizer.convert_tokens_to_ids(bpe_list + ['</s>', langcodes[i]])
            encoder_input = {
                'src_tokens': torch.tensor(input_ids).view(1, -1).to(self.device),
                'src_lengths': torch.tensor([len(input_ids)]).to(self.device)
            }
            encoder_outs = self.model.forward_encoder(encoder_input)
            np_encoder_outs = encoder_outs[0].encoder_out.cpu().squeeze(1).numpy().astype(np.float32)
            vectors.append(np_encoder_outs[:-2, :])
        return vectors

    def get_word_aligns(self, src_sent, trg_sent, langcodes=None, fwd_dict=None, bwd_dict=None, debug=False):
        l1_tokens = [self.tokenizer.tokenize(word) for word in src_sent]
        l2_tokens = [self.tokenizer.tokenize(word) for word in trg_sent]
        bpe_lists = [[bpe for w in sent for bpe in w] for sent in [l1_tokens, l2_tokens]]
        l1_b2w_map = list()
        for i, wlist in enumerate(l1_tokens):
            l1_b2w_map += [i for _ in wlist]
        l2_b2w_map = list()
        for i, wlist in enumerate(l2_tokens):
            l2_b2w_map += [i for _ in wlist]
        vectors = self.get_embed(list(bpe_lists), langcodes)
        sim = (cosine_similarity(vectors[0], vectors[1]) + 1.0) / 2.0
        sim = self.apply_distortion(sim, self.distortion)
        all_mats = dict()
        fwd, bwd = self.get_alignment_matrix(sim)
        if self.matching_method.find('a') != -1:
            all_mats['inter'] = fwd * bwd
        if self.matching_method.find('i') != -1:
            all_mats['itermax'] = self.iter_max(sim)
        if self.matching_method.find('m') != -1:
            all_mats['mwmf'] = self.get_max_weight_match(sim)
        if self.matching_method.find('f') != -1:
            all_mats['fixed'] = fwd * bwd
        aligns = {k: set() for k in all_mats}
        for key in aligns:
            for i in range(vectors[0].shape[0]):
                for j in range(vectors[1].shape[0]):
                    if all_mats[key][i, j] > 1e-10:
                        aligns[key].add((l1_b2w_map[i], l2_b2w_map[j]))
        if 'fixed' in aligns:
            src_aligned = set([x[0] for x in aligns['fixed']])
            trg_aligned = set([x[1] for x in aligns['fixed']])
            candidate_alignment = list()
            for i, sw in enumerate(src_sent):
                sw = sw.lower()
                if i not in src_aligned:
                    for j, tw in enumerate(trg_sent):
                        tw = tw.lower()
                        if tw in fwd_dict[sw]:
                            ri = i / len(src_sent)
                            rj = j / len(trg_sent)
                            if -0.2 < ri - rj < 0.2:
                                candidate_alignment.append((sw, tw, i, j, fwd_dict[sw][tw], 0))
            for j, tw in enumerate(trg_sent):
                tw = tw.lower()
                if j not in trg_aligned:
                    for i, sw in enumerate(src_sent):
                        sw = sw.lower()
                        if sw in bwd_dict[tw]:
                            ri = i / len(src_sent)
                            rj = j / len(trg_sent)
                            if -0.2 < ri - rj < 0.2:
                                candidate_alignment.append((sw, tw, i, j, bwd_dict[tw][sw], 1))
            candidate_alignment = sorted(candidate_alignment, key=lambda x: -x[-2])
            for sw, tw, i, j, val, d in candidate_alignment:
                if regex.match(r'\p{P}', sw) or regex.match(r'\p{P}', tw):
                    continue
                if val < 0.05:
                    break
                if d == 0:
                    if i in src_aligned:
                        continue
                    if (j not in trg_aligned) or ((i-1, j) in aligns['fixed']) or ((i+1, j) in aligns['fixed']):
                        aligns['fixed'].add((i, j))
                        src_aligned.add(i)
                        trg_aligned.add(j)
                        if debug:
                            print(sw, tw, i, j, val, d)
                else:
                    if j in trg_aligned:
                        continue
                    if (i not in src_aligned) or ((i, j+1) in aligns['fixed']) or ((i, j-1) in aligns['fixed']):
                        aligns['fixed'].add((i, j))
                        src_aligned.add(i)
                        trg_aligned.add(j)
                        if debug:
                            print(sw, tw, i, j, val, d)
        for ext in aligns:
            aligns[ext] = sorted(aligns[ext])
        return aligns

    @staticmethod
    def get_max_weight_match(sim):
        if nx is None:
            raise ValueError("networkx must be installed to use match algorithm.")

        def permute(edge):
            if edge[0] < sim.shape[0]:
                return edge[0], edge[1] - sim.shape[0]
            else:
                return edge[1], edge[0] - sim.shape[0]

        G = from_biadjacency_matrix(csr_matrix(sim))
        matching = nx.max_weight_matching(G, maxcardinality=True)
        matching = [permute(x) for x in matching]
        matching = sorted(matching, key=lambda x: x[0])
        res_matrix = np.zeros_like(sim)
        for edge in matching:
            res_matrix[edge[0], edge[1]] = 1
        return res_matrix

    @staticmethod
    def iter_max(sim_matrix, max_count=2):
        alpha_ratio = 0.9
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
        inter = forward * backward.transpose()

        if min(m, n) <= 2:
            return inter

        new_inter = np.zeros((m, n))
        count = 1
        while count < max_count:
            mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
            mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
            mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
            mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
            if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
                mask *= 0.0
                mask_zeros *= 0.0

            new_sim = sim_matrix * mask
            fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
            bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
            new_inter = fwd * bac

            if np.array_equal(inter + new_inter, inter):
                break
            inter = inter + new_inter
            count += 1
        return inter

    @staticmethod
    def get_alignment_matrix(sim_matrix):
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
        return forward, backward.transpose()

    @staticmethod
    def apply_distortion(sim_matrix, ratio=0.5):
        shape = sim_matrix.shape
        if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
            return sim_matrix

        pos_x = np.array([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])])
        pos_y = np.array([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])])
        distortion_mask = 1.0 - ((pos_x - np.transpose(pos_y)) ** 2) * ratio

        return np.multiply(sim_matrix, distortion_mask)


class Aligner(object):
    def __init__(self, aligner_type, **kwargs):
        self.aligner_type = aligner_type
        if aligner_type == 'simalign':
            from simalign import SentenceAligner
            d = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.aligner = SentenceAligner('xlm-roberta-base', device=d, **kwargs)
        elif aligner_type in ['fastalign', 'giza++']:
            pass
        elif aligner_type == 'criss-align':
            self.aligner = CRISSAligner(**kwargs)
        else:
            raise Exception('Aligner type not supported.')

    def align_sents(self, sent_pairs, train_file=None, **kwargs):
        aligns = list()
        if self.aligner_type in ['simalign', 'criss-align']:
            for src, trg in tqdm(sent_pairs):
                src = src.strip().split()
                trg = trg.strip().split()
                align_info = self.aligner.get_word_aligns(src, trg, **kwargs)
                result = None
                for key in align_info:
                    if result is None:
                        result = set(align_info[key])
                    else:
                        result = result.intersection(align_info[key])
                aligns.append(' '.join(['-'.join([str(x) for x in item]) for item in sorted(result)]))
        elif self.aligner_type == 'fastalign':
            temp_dir = tempfile.TemporaryDirectory(prefix='fast-align')
            with open(os.path.join(temp_dir.name, 'bitext.txt'), 'w') as fout:
                for ss, ts in sent_pairs:
                    fout.write(ss + ' ||| ' + ts + '\n')
                fout.close()
            if train_file is not None:
                assert os.path.exists(train_file)
                os.system(f'cat {train_file} >> {temp_dir.name}/bitext.txt')
            os.system(f'fast_align -d -o -v -i {temp_dir.name}/bitext.txt > {temp_dir.name}/fwd.align')
            os.system(f'fast_align -d -o -v -r -i {temp_dir.name}/bitext.txt > {temp_dir.name}/bwd.align')
            os.system(f'atools -i {temp_dir.name}/fwd.align -j {temp_dir.name}/bwd.align -c grow-diag-final-and > {temp_dir.name}/final.align')
            aligns = [x.strip() for x in open(f'{temp_dir.name}/final.align').readlines()][:len(sent_pairs)]
        elif self.aligner_type == 'giza++':
            assert train_file is not None
            giza_path = '/private/home/fhs/codebase/lexind/fairseq/2-word-align-final/giza-pp/GIZA++-v2/GIZA++'
            temp_dir = tempfile.TemporaryDirectory(prefix='giza++')
            d_src = collections.Counter()
            d_trg = collections.Counter()
            w2id_src = collections.defaultdict()
            w2id_trg = collections.defaultdict()
            for sent_pair in open(train_file):
                ss, ts = regex.split(r'\|\|\|', sent_pair.lower())
                for w in ss.strip().split():
                    d_src[w] += 1
                for w in ts.strip().split():
                    d_trg[w] += 1
            for ss, ts in sent_pairs:
                ss = ss.lower()
                ts = ts.lower()
                for w in ss.strip().split():
                    d_src[w] += 1
                for w in ts.strip().split():
                    d_trg[w] += 1
            with open(os.path.join(temp_dir.name, 's.vcb'), 'w') as fout:
                for i, w in enumerate(sorted(d_src.keys())):
                    print(i + 1, w, d_src[w], file=fout)
                    w2id_src[w] = i + 1
                fout.close()
            with open(os.path.join(temp_dir.name, 't.vcb'), 'w') as fout:
                for i, w in enumerate(sorted(d_trg.keys())):
                    print(i + 1, w, d_trg[w], file=fout)
                    w2id_trg[w] = i + 1
                fout.close()
            with open(os.path.join(temp_dir.name, 'bitext.train'), 'w') as fout:
                for sent_pair in open(train_file):
                    ss, ts = regex.split(r'\|\|\|', sent_pair.lower())
                    print(1, file=fout)
                    print(' '.join([str(w2id_src[x]) for x in ss.strip().split()]), file=fout)
                    print(' '.join([str(w2id_trg[x]) for x in ts.strip().split()]), file=fout)
                fout.close()
            with open(os.path.join(temp_dir.name, 'bitext.test'), 'w') as fout:
                for ss, ts in sent_pairs:
                    ss = ss.lower()
                    ts = ts.lower()
                    print(1, file=fout)
                    print(' '.join([str(w2id_src[x]) for x in ss.strip().split()]), file=fout)
                    print(' '.join([str(w2id_trg[x]) for x in ts.strip().split()]), file=fout)
                fout.close()
            os.chdir(f'{temp_dir.name}')
            os.system(f'{giza_path} -S {temp_dir.name}/s.vcb -T {temp_dir.name}/t.vcb -C {temp_dir.name}/bitext.train -tc {temp_dir.name}/bitext.test')
            # read giza++ results
            for i, line in enumerate(open(glob(f'{temp_dir.name}/*tst.A3*')[0])):
                if i % 3 == 2:
                    align = list()
                    is_trg = False
                    is_null = False
                    src_idx = 0
                    for item in line.strip().split():
                        if item == '({':
                            is_trg = True
                        elif item == '})':
                            is_trg = False
                        elif is_trg:
                            if not is_null:
                                trg_idx = int(item)
                                align.append(f'{src_idx}-{trg_idx}')
                        elif item != 'NULL':
                            src_idx += 1
                            is_null = False
                        else:
                            is_null = True
                    aligns.append(' '.join(align))
            temp_dir.cleanup()
        return aligns


class CRISSWrapper(object):

    def __init__(self, path='criss/criss-3rd.pt', args_path='criss/args.pt',
                 tokenizer='facebook/mbart-large-cc25', device='cpu'):
        from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
        from fairseq.sequence_generator import EnsembleModel
        self.device = device
        args = torch.load(args_path)
        task = tasks.setup_task(args)
        models, _model_args = checkpoint_utils.load_model_ensemble(
            path.split(':'),
            arg_overrides=eval('{}'),
            task=task
        )
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            model = model.to(self.device)
        self.model = EnsembleModel(models).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def embed(self, words, langcode='en_XX'):
        lbs, rbs = list(), list()
        tokens, word_ids = list(), list()
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            lbs.append(len(tokens))
            tokens.extend(word_tokens)
            rbs.append(len(tokens))
        tokens = [tokens + ['</s>', langcode]]
        lengths = [len(x) for x in tokens]
        max_length = max(lengths)
        for i in range(len(tokens)):
            word_ids.append(self.tokenizer.convert_tokens_to_ids(['<pad>'] * (max_length - len(tokens[i])) + tokens[i]))
        encoder_input = {
            'src_tokens': torch.tensor(word_ids).to(self.device),
            'src_lengths': torch.tensor(lengths).to(self.device)
        }
        encoder_outs = self.model.forward_encoder(encoder_input)
        np_encoder_outs = encoder_outs[0].encoder_out.float().detach()
        word_features = list()
        for i, lb in enumerate(lbs):
            rb = rbs[i]
            word_features.append(np_encoder_outs[lb:rb].mean(0))
        word_features = torch.cat(word_features, dim=0)
        return word_features


class WordAligner(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, feature_transform=3):
        super(WordAligner, self).__init__()
        layers = list()
        hidden_dims = [input_dim] + hidden_dims
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.ones(feature_transform))
        self.feature_transform = feature_transform

    def forward(self, x):
        transformed_features = torch.cat([x[:, :-self.feature_transform], torch.log(x[:, -self.feature_transform:] + self.bias.abs())], dim=-1)
        return self.model(transformed_features)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
