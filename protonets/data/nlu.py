import os
import sys
import glob

from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor
import corpusflow as cf
from nlp_sdk.lookups.vocab import Vocab
from nlp_sdk.preprocess.seq import pad_sequences
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

NLU_DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../data/nlu')
NLP_CACHE = { }

vocab = Vocab.load_from_text_file(os.path.join(NLU_DATA_DIR, "vocab.txt"))
corpus = cf.Corpus.read_from_file(os.path.join(NLU_DATA_DIR, "data.conllx"))

def convert_tensor(key, d):
    d[key] = torch.from_numpy(d[key]).long()
    return d

def lookup(vocab, key, d):
    d[key] = vocab.lookup_str_list(d[key])
    return d

def pad_text(key, length, d):
    d[key] = pad_sequences([d[key]], maxlen=length)
    return d

def convert_corpus(k, v):
    return { k: v.text }

def load_class_nlp(corpus, d):
    if d['class'] not in NLP_CACHE:
        class_corpus = list(filter(lambda x: x.domain == d["class"], corpus))

        image_ds = TransformDataset(ListDataset(class_corpus),
                                    compose([partial(convert_corpus, 'data'),
                                             partial(lookup, vocab, 'data'),
                                             partial(pad_text, 'data', 28),
                                             partial(convert_tensor, 'data')]))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            NLP_CACHE[d['class']] = sample['data']
            break # only need one sample because batch size equal to dataset length

    return { 'class': d['class'], 'data': NLP_CACHE[d['class']] }

def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def load(opt, splits):
    split_dir = os.path.join(NLU_DATA_DIR, 'splits', opt['data.split'])

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        transforms = [partial(convert_dict, 'class'),
                      partial(load_class_nlp, corpus),
                      partial(extract_episode, n_support, n_query)]
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = []
        with open(os.path.join(split_dir, "{:s}.txt".format(split)), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))
        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
