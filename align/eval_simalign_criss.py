# Copyright (c) Facebook, Inc. and its affiliates.

import regex
from data import AlignDataset
from evaluate import evaluate
from models import Aligner


import collections
resdict = collections.defaultdict(None)

aligner = Aligner(
    'criss-align', distortion=0,
    path='criss/criss-3rd.pt',
    args_path='criss/args.pt',
    matching_method='a'
)

dset = AlignDataset('data/align/', 'de-en')
aligns = aligner.align_sents(dset.sent_pairs, langcodes=('de_DE', 'en_XX'))
res = evaluate(dset.ground_truth, aligns, 1)
print('de-en:', res)
