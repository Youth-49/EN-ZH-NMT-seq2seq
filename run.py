#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu
from nmt_model import Hypothesis, NMT
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab
import argparse
import torch
import torch.nn.utils


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')

    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_src, source='tgt')

    # contruct parallel data (EN->ZH)
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args.batch_size)
    clip_grad = float(args.clip_grad)
    valid_niter = int(args.valid_niter)
    log_every = int(args.log_every)
    model_save_path = args.save_to

    vocab = Vocab.load(args.vocab)

    model = NMT(embed_size=int(args.embed_size),
                hidden_size=int(args.hidden_size),
                dropout_rate=float(args.dropout),
                vocab=vocab)
    model.train()

    uniform_init = float(args.uniform_init)
    if np.abs(uniform_init) > 0.:
        print(f'uniformly initialize parameters [-{uniform_init}, +{uniform_init}]')
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args.cuda else "cpu")
    print(f'use device: {device}')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print(f'epoch {epoch}, iter {train_iter}, avg. loss {report_loss/report_examples:.2f}, avg. ppl {math.exp(report_loss/report_tgt_words):.2f}, cum. examples {cum_examples}, speed {report_tgt_words/(time.time() - train_time):.2f} words/sec, time elapsed {time.time() - begin_time:.2f} sec')


                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print(f'epoch {epoch}, iter {train_iter}, cum. loss {cum_loss/cum_examples:.2f}, cum. ppl {np.exp(cum_loss/cum_tgt_words):.2f} cum. examples {cum_examples}')

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...')

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print(f'validation: iter {train_iter}, dev. ppl {dev_ppl}')

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print(f'save currently the best model to [{model_save_path}]')
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args.patience):
                    patience += 1
                    print('hit patience %d' % patience)

                    if patience == int(args.patience):
                        num_trial += 1
                        print(f'hit #{num_trial} trial')
                        if num_trial == int(args.max_num_trial):
                            print('early stop!')
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args.lr_decay)
                        print(f'load previously best model and decay learning rate to {lr}')

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers')
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args.max_epoch):
                    print('reached maximum number of epochs!')
                    exit(0)


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print(f"load test source sentences from [{args.test_src}]")
    test_data_src = read_corpus(args.test_src, source='src')
    if args.test_tgt:
        print(f"load test target sentences from [{args.test_tgt}]")
        test_data_tgt = read_corpus(args.test_tgt, source='tgt')

    print(f"load model from {args.model_path}")
    model = NMT.load(args.model_path)

    if args.cuda:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args.beam_size),
                             max_decoding_time_step=int(args.max_decoding_time_step))

    if args.test_tgt:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100))

    with open(args.output_file, 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def translate(src: str) -> str:

    test_data_src = []
    test_data_src.append(src.strip().split(' '))

    model = NMT.load('model.bin')

    hypotheses = beam_search_4trans(model, test_data_src,
                             beam_size=int(5),
                             max_decoding_time_step=int(70))

    hyp_sent = ''
    for src_sent, hyps in zip(test_data_src, hypotheses):
        top_hyp = hyps[0]
        hyp_sent = ''.join(top_hyp.value)
    return hyp_sent
        

def beam_search_4trans(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in test_data_src:
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)
    if was_training: model.train(was_training)

    return hypotheses


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--dev_src', type=str, default='./data/dev.en')
    parser.add_argument('--dev_tgt', type=str, default='./date/dev.zh')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=float, default=256)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--max_decoding_time_step', type=int, default=70)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--max_num_trial', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--save_to', type=str, default='model.bin')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_src', type=str, default='./data/train.en')
    parser.add_argument('--train_tgt', type=str, default='./data/train.zh')
    parser.add_argument('--uniform_init', type=float, default=0.1)
    parser.add_argument('--valid_niter', type=int, default=2000)
    parser.add_argument('--vocab', type=str, default='vocab.json')
    parser.add_argument('--test_src', type=str, default='./data/test.en')
    parser.add_argument('--test_tgt', type=str, default='./data/test.zh')
    parser.add_argument('--model_path', type=str, default='model.bin')
    parser.add_argument('--output_file', type=str, default='./outputs/test-output.txt')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--decode', type=bool, default=False)

    args = parser.parse_args()
    vars(args)['model'] = 'Seq2Seq+Attn+BeamSearch'
    
    # seed the random number generators
    seed = int(args.seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args.train:
        train(args)
    elif args.decode:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
