import torch
from torch import optim

import os
import os.path
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
import pprint; pp = pprint.PrettyPrinter(indent=4)

import utils
from utils import read_vocab, Tokenizer, vocab_pad_idx, timeSince, try_cuda
from utils import filter_param, module_grad, colorize
from utils import NumpyEncoder
from env_simplified import R2RBatch, ImageFeatures
import train
from vocab import SUBTRAIN_VOCAB, TRAINVAL_VOCAB, TRAIN_VOCAB

learning_rate = 0.0001
weight_decay = 0.0005

def get_model_prefix(args, image_feature_list):
    model_prefix = train.get_model_prefix(args, image_feature_list)
    model_prefix.replace('follower','search',1)
    return model_prefix

def run_search(args, agent, train_env, val_envs):
    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
        print("evaluating on {}".format(env_name))
        agent.env = val_env
        if hasattr(agent, 'speaker') and agent.speaker:
            agent.speaker.env = val_env
        agent.results_path = '%s/%s_%s.json' % (
            args.SEARCH_DIR, get_model_prefix(
                args, val_env.image_features_list), env_name)
        agent.test(use_dropout=False, feedback='argmax',distance_aware=args.distance_aware,distance_weight=args.distance_weight)
        if not args.no_save:
            agent.write_test_results()
            agent.write_results(results=agent.clean_results, results_path='%s/%s_clean.json'%(args.RESULT_DIR, env_name))
        score_summary, _ = evaluator.score_results(agent.results)
        with open(os.path.join( args.SEARCH_DIR,f'{env_name}_eval_result.json'),'w') as f:
            json.dump(score_summary,f)
        pp.pprint(score_summary)

def cache(args, agent, train_env, val_envs):
    if train_env is not None:
        cache_env_name = ['train'] + list(val_envs.keys())
        cache_env = [train_env] + [v[0] for v in val_envs.values()]
    else:
        cache_env_name = list(val_envs.keys())
        cache_env = [v[0] for v in val_envs.values()]

    print(cache_env_name)
    for env_name, env in zip(cache_env_name,cache_env):
        #if env_name is not 'val_unseen': continue
        agent.env = env
        if agent.speaker: agent.speaker.env = env
        print("Generating candidates for", env_name)
        agent.cache_search_candidates()
        if not args.no_save:
            with open('cache_{}{}{}{}.json'.format(env_name,'_debug' if args.debug else '', args.max_episode_len, args.early_stop),'w') as outfile:
                json.dump(agent.cache_candidates, outfile, cls=NumpyEncoder)
            with open('search_{}{}{}{}.json'.format(env_name,'_debug' if args.debug else '', args.max_episode_len, args.early_stop),'w') as outfile:
                json.dump(agent.cache_search, outfile, cls=NumpyEncoder)
        score_summary, _ = eval.Evaluation(env.splits).score_results(agent.results)
        pp.pprint(score_summary)

def make_arg_parser():
    parser = train.make_arg_parser()
    parser.add_argument("--max_episode_len", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.21)
    parser.add_argument("--mean", action='store_true')
    parser.add_argument("--logit", action='store_true')
    parser.add_argument("--early_stop", action='store_true')
    parser.add_argument("--revisit", action='store_true')
    parser.add_argument("--inject_stop", action='store_true')
    parser.add_argument("--load_reranker", type=str, default='')
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--beam", action='store_true')
    parser.add_argument("--load_speaker", type=str,
            default='')
    parser.add_argument("--job", choices=['search','sweep','train','cache','test'],default='search')
    parser.add_argument('--distance_aware',action='store_true')
    parser.add_argument('--distance_weight',default=1,type=float)
    return parser

def setup_agent_envs(args):
    train_splits = []
    return train.train_setup(args, train_splits)

def test(args, agent, val_envs):
    test_env = val_envs['test'][0]
    test_env.notTest = False
    agent.is_search()
    agent.env = test_env
    agent.results_path = '%stest.json' % (args.RESULT_DIR)
    agent.test(use_dropout=False, feedback='argmax',distance_aware=args.distance_aware,distance_weight=args.distance_weight)
    agent.write_test_results()
    print("finished testing. recommended to save the trajectory again")

def main(args):
    if args.job == 'test':
        args.use_test_set = True
        args.use_pretraining = False


    agent, train_env, val_envs = setup_agent_envs(args)
    agent.is_search()
    agent.search_logit = args.logit
    agent.search_mean = args.mean
    agent.search_early_stop = args.early_stop
    agent.episode_len = args.max_episode_len
    agent.gamma = args.gamma
    agent.revisit = args.revisit
    agent.inject_stop = args.inject_stop
    agent.K = args.K
    agent.beam = args.beam

    agent.search_episode_len = args.max_episode_len
    agent.gamma = args.gamma
    print('gamma', args.gamma, 'ep_len', args.ep_len)
    run_search(args, agent, train_env, val_envs)

    # Load speaker

if __name__ == "__main__":
    utils.run(make_arg_parser(), main)
