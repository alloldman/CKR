import torch
from torch import optim

import os
import os.path
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

import utils
from utils import read_vocab, Tokenizer, vocab_pad_idx, timeSince, try_cuda
from utils import module_grad, colorize, filter_param
from env_simplified import R2RBatch, ImageFeatures,REVERIEBatch
from model import TransformerEncoder,EmbeddingEncoder, EncoderLSTM
from model import object_roomTransformer
from follower import Seq2SeqAgent,TransformerAgent
from eval_reverie import Evaluation as EvaluationR
from vocab import SUBTRAIN_VOCAB, TRAINVAL_VOCAB, TRAIN_VOCAB,REVERIE_VOCAB
import torch.nn as nn
MAX_INPUT_LENGTH = 80 
from torch.optim.lr_scheduler import LambdaLR
import json
max_episode_len = 10
glove_path = 'data/train_glove.npy'
action_embedding_size = 2048+128
hidden_size = 512
dropout_ratio = 0.5
learning_rate = 0.0001
weight_decay = 0.0005
FEATURE_SIZE = 2048+128
log_every = 100
save_every = 10000


def get_model_prefix(args, image_feature_list):
    image_feature_name = "+".join(
        [featurizer.get_name() for featurizer in image_feature_list])
    model_prefix = 'follower{}_{}_{}heads'.format(
        args.feedback_method, image_feature_name, args.num_head)
    if args.use_train_subset:
        model_prefix = 'trainsub_' + model_prefix
    if args.bidirectional:
        model_prefix = model_prefix + "_bidirectional"
    if args.use_pretraining:
        model_prefix = model_prefix.replace(
            'follower', 'follower_with_pretraining', 1)
    if args.transformerDecoder:
        model_prefix = model_prefix.replace(
            'follower','transformer_decoder',1
        )
    return model_prefix


def eval_model(agent, results_path, use_dropout, feedback, allow_cheat=False):
    agent.results_path = results_path
    agent.test(
        use_dropout=use_dropout, feedback=feedback, allow_cheat=allow_cheat)


def train(args, train_env, agent, optimizers,schedulers, n_iters, log_every=log_every, val_envs=None):
    ''' Train on training set, validating on both seen and unseen. '''

    if val_envs is None:
        val_envs = {}

    agent.search_logit = True
    agent.search_mean = False
    agent.search_early_stop = True
    agent.search_episode_len = 40
    agent.gamma = 0.21
    agent.revisit = False
    agent.inject_stop = False
    agent.K = 20
    agent.beam = 1  


    print('Training with %s feedback' % args.feedback_method)

    data_log = defaultdict(list)
    start = time.time()

    split_string = "-".join(train_env.splits)

    def make_path(n_iter):
        return os.path.join(
            args.SNAPSHOT_DIR, '%s_%s_iter_%d' % (
                get_model_prefix(args, train_env.image_features_list),
                split_string, n_iter))

    for idx in range(0, n_iters, log_every):
        agent.env = train_env

        interval = min(log_every, n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)
        loss_str = ''

        env_name = 'train'
        import ipdb;ipdb.set_trace()
        agent.train(optimizers,schedulers, interval, feedback=args.feedback_method,
            use_angle_distance_loss=args.use_angle_distance_loss,use_angle_distance_reward=args.use_angle_distance_reward)
        _loss_str, losses = agent.get_loss_info()
        loss_str += env_name + ' ' + _loss_str
        for k,v in losses.items():
            data_log['%s %s' % (env_name,k)].append(v)

        save_log = []
        to_save = False
        score_summary_list = []
        for env_name, (val_env, evaluator) in sorted(val_envs.items()):
            agent.env = val_env
            agent.not_search()
            agent.test(use_dropout=True, feedback=args.feedback_method,
                       allow_cheat=True)
            _loss_str, losses = agent.get_loss_info()
            loss_str += ', ' + env_name + ' ' + _loss_str
            for k,v in losses.items():
                data_log['%s %s' % (env_name,k)].append(v)

            agent.results_path = '%s/%s_%s_iter_%d.json' % (
                args.RESULT_DIR, get_model_prefix(
                    args, train_env.image_features_list),
                env_name, iter)
            agent.not_search()
            agent.test(use_dropout=False, feedback='argmax',distance_aware=args.distance_aware,distance_weight=args.distance_weight)

            print("evaluating on {}".format(env_name))
            score_summary, _ = evaluator.score_results(agent.results)
            score_summary_list.append((env_name,score_summary))
            for metric, val in sorted(score_summary.items()):
                data_log['%s %s' % (env_name, metric)].append(val)
                loss_str += ', %s: %.3f' % (metric, val)
            if env_name =='val_seen':
                if score_summary['success_rate']>0.55:
                    to_save = True
            if env_name == 'val_unseen':
                if score_summary['success_rate']>0.0:
                    to_save = True
        if not args.no_save and to_save:
            save_path = ''
            metric='success_rate'
            for (env_name,score_summary) in score_summary_list:
                save_path +='%s_sr_%.3f_'%(env_name,score_summary[metric])
            model_path = make_path(iter) + save_path
            save_log.append(
                "new best, saved model to %s" % model_path)
            agent.save(model_path)
            agent.write_results()
        print(('%s (%d %d%%) %s' % (
            timeSince(start, float(iter)/n_iters),
            iter, float(iter)/n_iters*100, loss_str)))
        for s in save_log:
            print(colorize(s))
        if not args.no_save:
            if save_every and iter % save_every == 0:
                agent.save(make_path(iter))

            df = pd.DataFrame(data_log)
            df.set_index('iteration')
            df_path = '%s/%s_%s_log.csv' % (
                args.PLOT_DIR, get_model_prefix(
                    args, train_env.image_features_list), split_string)
            df.to_csv(df_path)

def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_more_train_env(args, train_vocab_path, train_splits):
    setup(args.seed)
    image_features_list = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    if args.instruction_from =='reverie':
        train_env = REVERIEBatch(image_features_list, batch_size=args.batch_size,
                            splits=train_splits, tokenizer=tok)
    else:
        train_env = R2RBatch(image_features_list, batch_size=args.batch_size,
                            splits=train_splits, tokenizer=tok)

    return train_env



def try_parallel(args,module):
    if module is None:
        return module
    if torch.cuda.device_count() > 1 and len(args.gpus) > 1:
        module = nn.DataParallel(module, device_ids=args.gpus)
    return module
    

def make_follower(args, vocab):
    enc_hidden_size = hidden_size//2 if args.bidirectional else hidden_size
    glove = np.load(glove_path) if args.use_glove else None
    feature_size = FEATURE_SIZE
    word_embedding_size = 256 if args.coground else 300
    encoder= TransformerEncoder(
    len(vocab), word_embedding_size, enc_hidden_size, vocab_pad_idx,
    dropout_ratio,num_layers=args.en_nlayer,nhead=args.en_nhead,
    bidirectional=args.bidirectional, glove=glove,ff=args.ff)

    Decoder = object_roomTransformer
    decoder = Decoder(
    action_embedding_size, hidden_size, dropout_ratio,
    feature_size=feature_size, num_head=args.num_multihead,num_layer=args.num_layer,
        transformer_dropout_rate=args.transformer_dropout_rate,ff=args.ff,
        use_room=args.not_room,use_object=args.not_object,concate_room=args.concate_room,
        load_room_relation_weight=args.not_load_room_relation_weight ,load_object_relation_weight=args.not_load_object_relation_weight,
        num_gcn=args.num_gcn,max_degree=args.max_degree,short_cut=args.short_cut,soft_room_label=args.soft_room_label,
        room_relation_vec=args.room_relation_vec)
    decoder = try_cuda(decoder)
    
    if len(args.gpus) > 1:
        decoder = try_parallel(args,decoder)  
    
    encoder = try_cuda(encoder)
    if len(args.gpus) > 1:
        encoder = try_parallel(args,encoder)
    agent = TransformerAgent(
        None, "", encoder, decoder, max_episode_len,
        max_instruction_length=MAX_INPUT_LENGTH,
        attn_only_verb=args.attn_only_verb,loss_weight=args.loss_weight)

    if args.load_follower is not '':
        scorer_exists = os.path.isfile(args.load_follower + '_scorer_enc')
        agent.load(args.load_follower, load_scorer=(args.load_scorer is '' and scorer_exists))
        print(colorize('load follower '+ args.load_follower))

    return agent

def make_env_and_models(args, train_vocab_path, train_splits, test_splits) :
    setup(args.seed)
    image_features_list = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    train_env = REVERIEBatch(image_features_list, batch_size=args.batch_size,
                        splits=train_splits, tokenizer=tok) if len(train_splits) > 0 else None
    test_envs = {
        split: (REVERIEBatch(image_features_list, batch_size=args.batch_size,
                        splits=[split], tokenizer=tok),
                EvaluationR(split,instrType='instructions'))
        for split in test_splits}

    agent = make_follower(args, vocab)
    agent.env = train_env

    return train_env, test_envs, agent

def update_leanring_rate(i):
    if i<6000:
        return 1
    elif i<10000: 
        return 0.5
    else:
        return 0.1
def train_setup(args, train_splits=['train']):
    val_splits = ['val_seen', 'val_unseen']
    if args.use_test_set:
        val_splits = ['test']
    if args.debug:
        log_every = 5
        args.n_iters = 10
        train_splits = val_splits = ['val_unseen']
    vocab = TRAIN_VOCAB

    if args.use_train_subset:
        train_splits = ['sub_' + split for split in train_splits]
        val_splits = ['sub_' + split for split in val_splits]
        vocab = SUBTRAIN_VOCAB
    train_env, val_envs, agent = make_env_and_models(
        args, vocab, train_splits, val_splits)

    if args.use_pretraining:
        pretrain_splits = args.pretrain_splits
        assert len(pretrain_splits) > 0, \
            'must specify at least one pretrain split'
        pretrain_env = make_more_train_env(
            args, vocab, pretrain_splits)

    if args.use_pretraining:
        return agent, train_env, val_envs, pretrain_env
    else:
        return agent, train_env, val_envs


def train_val(args):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    if args.use_pretraining:
        agent, train_env, val_envs, pretrain_env = train_setup(args)
    else:
        agent, train_env, val_envs = train_setup(args)
    if not args.class_new_lr and args.new_decoder:
        agent.decoder.activate_classifier()

    m_dict = {
            'follower': [agent.encoder,agent.decoder],
            'all': agent.modules()
        }
    
    optimizers = [optim.Adam(filter_param(m), lr=args.lr,
        weight_decay=weight_decay) for m in m_dict[args.grad] if len(filter_param(m))]

    schedulers = [LambdaLR(optimizer,lr_lambda=update_leanring_rate) for optimizer in optimizers]

    if args.class_new_lr:
        classifier = [agent.decoder.view_room_classifier,agent.decoder.text_room_classifier]
        agent.decoder.activate_classifier()
        cl_optimizers = [optim.Adam(filter_param(m), lr=args.lr*0.1,
        weight_decay=weight_decay) for m in classifier if len(filter_param(m))]
        cl_schedulers = [LambdaLR(optimizer,lr_lambda=update_leanring_rate) for optimizer in cl_optimizers]
        optimizers = optimizers+cl_optimizers
        schedulers = schedulers+cl_schedulers
    
    for optimizer in optimizers:
        optimizer  = try_parallel(args,optimizer)
    if args.use_pretraining:
        train(args, pretrain_env, agent, optimizers,schedulers,
              args.n_pretrain_iters, val_envs=val_envs)

    train(args, train_env, agent, optimizers,schedulers,
          args.n_iters, val_envs=val_envs)




def make_arg_parser():
    parser = argparse.ArgumentParser()
    ImageFeatures.add_args(parser)
    parser.add_argument("--instruction_from",type=str,default='R2R')
    parser.add_argument("--load_follower", type=str, default='')
    parser.add_argument( "--feedback_method",
            choices=["sample", "teacher", "sample1step","sample2step","sample3step","teacher+sample","recover"], default="sample")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--bidirectional", action='store_true')
    parser.add_argument("--scorer", action='store_true')
    parser.add_argument("--n_iters", type=int, default=20000)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--use_pretraining", action='store_true')
    parser.add_argument("--grad", type=str, default='all')
    parser.add_argument("--pretrain_splits", nargs="+", default=[])
    parser.add_argument("--n_pretrain_iters", type=int, default=50000)
    parser.add_argument("--no_save", action='store_true')
    parser.add_argument("--use_glove", action='store_true')
    parser.add_argument("--attn_only_verb", action='store_true')
    parser.add_argument("--use_train_subset", action='store_true',
        help="use a subset of the original train data for validation")
    parser.add_argument("--use_test_set", action='store_true')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--num_layer',type=int,default=6)
    parser.add_argument('--num_multihead',type=int,default=8)
    parser.add_argument('--transformer_dropout_rate',default=0.1,type=float)
    parser.add_argument('--gpus',nargs="+",default=[0],type=int)
    parser.add_argument('--config',type=str,default='pointer/vilbert_config.json')
    parser.add_argument('--use_angle_distance_loss',action='store_true')
    parser.add_argument('--use_angle_distance_reward',action='store_true' )
    parser.add_argument('--en_nlayer',type=int,default=2)
    parser.add_argument('--en_nhead',type=int,default=6)
    parser.add_argument('--loss_weight',nargs='+',type=float,default=[0,1,0,0],
        help='add three loss weight for the different loss component:angle reward, cross_entropy ,self_monitor ')
    parser.add_argument('--not_room',action='store_false')
    parser.add_argument('--not_object',action='store_false')
    parser.add_argument('--num_gcn',type=int,default=3)
    parser.add_argument('--max_degree',type=int,default=10)
    parser.add_argument('--short_cut',action='store_true')
    parser.add_argument('--ff',type=int,default=2048)
    parser.add_argument('--new_decoder',action='store_true')
    parser.add_argument('--not_load_room_relation_weight',action='store_false')
    parser.add_argument('--lr',default=learning_rate,type=float)
    parser.add_argument('--not_load_object_relation_weight',action='store_false')
    parser.add_argument('--class_new_lr',action='store_true')
    parser.add_argument('--soft_room_label',action='store_true')
    parser.add_argument('--object_top_n',type=int,default=5)
    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), train_val)
