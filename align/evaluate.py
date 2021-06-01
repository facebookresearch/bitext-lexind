# Copyright (c) Facebook, Inc. and its affiliates.

import regex


def evaluate(gold, silver, offset=0):
    assert len(gold) == len(silver)
    a_size = s_size = p_size = ap_inter = as_inter = 0
    for i, g in enumerate(gold):
        s = set([
            tuple(map(lambda x: int(x), item.split('-')))
            for item in filter(lambda x: x.find('p') == -1, g.split())
        ])
        p = set([tuple(map(lambda x: int(x), regex.split('-|p', item))) for item in g.split()])
        a = set([tuple(map(lambda x: int(x) + offset, regex.split('-', item))) for item in silver[i].split()])
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
