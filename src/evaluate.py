# Copyright (c) Facebook, Inc. and its affiliates.

def evaluate(pr_pairs, gt_pairs):
    gt_set = set([tuple(x) for x in gt_pairs])
    pr_set = set([tuple(x) for x in pr_pairs])
    prec = sum([1 if x in gt_set else 0 for x in pr_set]) \
         / float(len(pr_set)) if len(pr_set) > 0 else 0
    rec = sum([1 if x in pr_set else 0 for x in gt_set]) \
        / float(len(gt_set)) if len(gt_set) > 0 else 0
    gt_src_words = set([x[0] for x in gt_pairs])
    pr_src_words = set([x[0] for x in pr_pairs])
    oov_number = sum([1 if x not in pr_src_words else 0 for x in gt_src_words])
    oov_rate = oov_number / float(len(gt_src_words))
    eval_result = {
        'oov_number': oov_number,
        'oov_rate': oov_rate,
        'precision': prec,
        'recall': rec,
        'f1': 2.0 * prec * rec / (prec + rec) if prec > 0 or rec > 0 else 0.0
    }
    return eval_result
