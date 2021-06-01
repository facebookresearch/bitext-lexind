# Copyright (c) Facebook, Inc. and its affiliates.

from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn as nn


class CRISSWrapper(object):
    def __init__(self, path='criss/criss-3rd.pt',
                 args_path='criss/args.pt',
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

    def word_embed(self, words, langcode='en_XX'):
        tokens = list()
        word_ids = list()
        for word in words:
            word_tokens = self.tokenizer.tokenize(word) + ['</s>', langcode]
            tokens.append(word_tokens)
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
        encoder_mask = 1 - encoder_outs[0].encoder_padding_mask.float().detach()
        encoder_mask = encoder_mask.transpose(0, 1).unsqueeze(2)
        masked_encoder_outs = encoder_mask * np_encoder_outs
        avg_pool = (masked_encoder_outs / encoder_mask.sum(dim=0)).sum(dim=0)
        return avg_pool


class LexiconInducer(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, feature_transform=3):
        super(LexiconInducer, self).__init__()
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
