# Copyright (c) Facebook, Inc. and its affiliates.

from train import *
from data import AlignDataset
import collections
import copy
import numpy as np
from models import Aligner


def eval_align(gold, silver, adjust=0):
    assert len(gold) == len(silver)
    a_size = s_size = p_size = ap_inter = as_inter = 0
    for i, g in enumerate(gold):
        s = set([
            tuple(map(lambda x: int(x), item.split('-')))
            for item in filter(lambda x: x.find('p') == -1, g.split())
        ])
        p = set([tuple(map(lambda x: int(x), regex.split('-|p', item))) for item in g.split()])
        a = set([tuple(map(lambda x: int(x) + adjust, regex.split('-', item))) for item in silver[i].split()])
        ap_inter += len(a.intersection(p))
        as_inter += len(a.intersection(s))
        a_size += len(a)
        p_size += len(p)
        s_size += len(s)
    prec = ap_inter / a_size if a_size > 0 else 0
    rec = as_inter / s_size if s_size > 0 else 0
    return {
        'prec': prec,
        'rec': rec,
        'f1': 2 * prec * rec / (prec + rec) if s_size > 0 and a_size > 0 else 0,
        'aer': 1 - (as_inter + ap_inter) / (a_size + s_size)
    }


def inference(simalign, probs, threshold):
    n, m = probs.shape
    ids = probs.view(-1).argsort(descending=True)
    f = lambda x, m: (x.item()//m, x.item()%m)
    src2trg = collections.defaultdict(set)
    trg2src = collections.defaultdict(set)
    results = set()
    for pair in simalign.split():
        x, y = pair.split('-')
        x = int(x)
        y = int(y)
        src2trg[x].add(y)
        trg2src[y].add(x)
        results.add((x, y))
    for idx in ids:
        x, y = f(idx, m)
        if probs[x, y] < threshold:  # too low similarity
            break
        if (x not in src2trg) and (y not in trg2src): # perfect company, keep
            src2trg[x].add(y)
            trg2src[y].add(x)
            results.add((x, y))
        elif (x in src2trg) and (y in trg2src):  # both have other companies, skip
            continue
        elif x in src2trg:  # x has company, but y is still addable 
            if y == max(src2trg[x]) + 1 or y == min(src2trg[x]) - 1:
                src2trg[x].add(y)
                trg2src[y].add(x)
                results.add((x, y))
        else:
            if x == max(trg2src[y]) + 1 or x == min(trg2src[y]) - 1:
                src2trg[x].add(y)
                trg2src[y].add(x)
                results.add((x, y))
    results = ' '.join([f'{x}-{y}' for x, y in sorted(results)])
    return results


def test(configs, criss, dataset, simaligns, threshold=0.5):
    setup_configs(configs)
    os.system(f'mkdir -p {configs.save_path}')
    torch.save(configs, configs.save_path + '/configs.pt')
    info = collect_bitext_stats(
        configs.bitext_path, configs.align_path, configs.save_path, 
        configs.src_lang, configs.trg_lang, configs.reversed
    )
    aligner = WordAligner(5 + (2 if configs.use_criss else 0), configs.hiddens, 3, 5).to(configs.device)
    model_path = configs.save_path+f'/model.pt'
    results = list()
    aligner.load_state_dict(torch.load(model_path))
    for idx, batch in enumerate(tqdm(dataset.sent_pairs)):
        ss, ts = batch
        ss = ss.split()
        ts = ts.split()
        if criss is not None:
            semb = criss.embed(ss, langcode=configs.src_lang)
            temb = criss.embed(ts, langcode=configs.trg_lang)
            cos_matrix = cos(semb.unsqueeze(1), temb.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)
            ip_matrix = (semb.unsqueeze(1) * temb.unsqueeze(0)).sum(-1).unsqueeze(-1).unsqueeze(-1)
            feat_matrix = torch.cat((cos_matrix, ip_matrix), dim=-1)
        word_pairs = list()
        criss_features = list()
        for i, sw in enumerate(ss):
            for j, tw in enumerate(ts):
                word_pairs.append((sw, tw))
                criss_features.append(feat_matrix[i, j])
        scores = extract_scores(word_pairs, criss_features, aligner, info, configs).reshape(len(ss), len(ts), -1)
        scores = scores.softmax(-1)
        arrange = torch.arange(3).to(configs.device).view(1, 1, -1)
        scores = (scores * arrange).sum(-1)
        result = inference(simaligns[idx], scores, threshold)
        results.append(result)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--align', type=str, help='path to word alignment')
    parser.add_argument('-b', '--bitext', type=str, help='path to bitext')
    parser.add_argument('-g', '--ground-truth', type=str, default='./data/align/', help='path to ground-truth')
    parser.add_argument('-src', '--source', type=str, help='source language code')
    parser.add_argument('-trg', '--target', type=str, help='target language code')
    parser.add_argument('-m', '--model-path', type=str, default='./model/', help='path to output folder')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device for training [cuda|cpu]')
    args = parser.parse_args()

    configs = dotdict.DotDict(
        {
            'align_path': args.align, 
            'bitext_path': args.bitext,
            'save_path': args.model_path,
            'batch_size': 128,
            'epochs': 100,
            'device': args.device,
            'hiddens': [8],
            'use_criss': True,
            'src_lang': args.source,
            'trg_lang': args.target,
            'threshold': 1.0
        }
    )
    criss = CRISSWrapper(device=configs.device)
    dataset = collections.defaultdict(None)
    simaligner = Aligner(
        'criss-align', distortion=0,
        path='criss/criss-3rd.pt', args_path='criss/args.pt',
        matching_method='a'
    )
    lp = (args.source, args.target)
    dset = AlignDataset(args.ground_truth, f'{args.source.split("_")[0]}-{args.target.split("_")[0]}')
    simaligns = simaligner.align_sents(dset.sent_pairs, langcodes=lp)
    aligns = test(configs, criss, dset, simaligns, configs.threshold)
    results = eval_align(dset.ground_truth, aligns, 1)
    print(results)
    from IPython import embed; embed(using=False)
