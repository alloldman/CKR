''' Batched Room-to-Room navigation environment '''

import os
import sys
file_path = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(file_path))
sys.path.append(module_path)
module_path = os.path.abspath(os.path.join(file_path,'..','..','build'))
sys.path.append(module_path)
import csv
import numpy as np

import math
import json
import random
import networkx as nx
import functools
import os.path
import time
import paths
import pickle
import os.path
import sys
import itertools
import ipdb
from collections import namedtuple, defaultdict

from utils import load_datasets, load_nav_graphs, structured_map, vocab_pad_idx, decode_base64, k_best_indices, try_cuda, spatial_feature_from_bbox

import torch
from torch.autograd import Variable
from merge_data import load_datas
csv.field_size_limit(sys.maxsize)

# Not needed for panorama action space
# FOLLOWER_MODEL_ACTIONS = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
#
# LEFT_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("left")
# RIGHT_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("right")
# UP_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("up")
# DOWN_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("down")
# FORWARD_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("forward")
# END_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<end>")
# START_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<start>")
# IGNORE_ACTION_INDEX = FOLLOWER_MODEL_ACTIONS.index("<ignore>")


# FOLLOWER_ENV_ACTIONS = [
#     (0,-1, 0), # left
#     (0, 1, 0), # right
#     (0, 0, 1), # up
#     (0, 0,-1), # down
#     (1, 0, 0), # forward
#     (0, 0, 0), # <end>
#     (0, 0, 0), # <start>
#     (0, 0, 0)  # <ignore>
# ]

# assert len(FOLLOWER_MODEL_ACTIONS) == len(FOLLOWER_ENV_ACTIONS)

angle_inc = np.pi / 6.
ANGLE_INC = np.pi / 6.

def _build_action_embedding(adj_loc_list, features):
    feature_dim = features.shape[-1]
    embedding = np.zeros((len(adj_loc_list), feature_dim + 128), np.float32)
    for a, adj_dict in enumerate(adj_loc_list):
        if a == 0:
            # the embedding for the first action ('stop') is left as zero
            continue
        embedding[a, :feature_dim] = features[adj_dict['absViewIndex']]
        loc_embedding = embedding[a, feature_dim:]
        rel_heading = adj_dict['rel_heading']
        rel_elevation = adj_dict['rel_elevation']
        loc_embedding[0:32] = np.sin(rel_heading)
        loc_embedding[32:64] = np.cos(rel_heading)
        loc_embedding[64:96] = np.sin(rel_elevation)
        loc_embedding[96:] = np.cos(rel_elevation)
    return embedding


def build_viewpoint_loc_embedding(viewIndex):
    """
    Position embedding:
    heading 64D + elevation 64D
    1) heading: [sin(heading) for _ in range(1, 33)] +
                [cos(heading) for _ in range(1, 33)]
    2) elevation: [sin(elevation) for _ in range(1, 33)] +
                  [cos(elevation) for _ in range(1, 33)]
    """
    embedding = np.zeros((36, 128), np.float32)
    for absViewIndex in range(36):
        relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
        rel_heading = (relViewIndex % 12) * angle_inc
        rel_elevation = (relViewIndex // 12 - 1) * angle_inc
        embedding[absViewIndex,  0:32] = np.sin(rel_heading)
        embedding[absViewIndex, 32:64] = np.cos(rel_heading)
        embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
        embedding[absViewIndex,   96:] = np.cos(rel_elevation)
    return embedding


# pre-compute all the 36 possible paranoram location embeddings
_static_loc_embeddings = [
    build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)]


def _loc_distance(loc):
    return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)


def _canonical_angle(x):
    ''' Make angle in (-pi, +pi) '''
    return x - 2 * np.pi * round(x / (2 * np.pi))




WorldState = namedtuple(
    "WorldState",
    ["scanId", "viewpointId", "viewIndex", "heading", "elevation"]
)
BottomUpViewpoint = namedtuple("BottomUpViewpoint", ["cls_prob", "image_features", "attribute_indices", "object_indices", "spatial_features", "no_object_mask"])


class ImageFeatures(object):
    NUM_VIEWS = 36
    MEAN_POOLED_DIM = 2048
    feature_dim = MEAN_POOLED_DIM

    IMAGE_W = 640
    IMAGE_H = 480
    VFOV = 60

    @staticmethod
    def from_args(args):
        feats = []
        for image_feature_type in sorted(args.image_feature_type):
            if image_feature_type == "none":
                feats.append(NoImageFeatures())
            elif image_feature_type == "bottom_up_attention":
                # feats.append(BottomUpImageFeatures(
                #     args.bottom_up_detections,
                #     #precomputed_cache_path=paths.bottom_up_feature_cache_path,
                #     precomputed_cache_dir=paths.bottom_up_feature_cache_dir,
                # ))
                raise NotImplementedError('bottom_up_attention has not been implemented for panorama environment')
            elif image_feature_type == "convolutional_attention":
                feats.append(ConvolutionalImageFeatures(
                    args.image_feature_datasets,
                    split_convolutional_features=True,
                    downscale_convolutional_features=args.downscale_convolutional_features
                ))
                raise NotImplementedError('convolutional_attention has not been implemented for panorama environment')
            else:
                assert image_feature_type == "mean_pooled"
                feats.append(MeanPooledImageFeatures(args.image_feature_datasets))
        return feats

    @staticmethod
    def add_args(argument_parser):
        argument_parser.add_argument("--image_feature_type", nargs="+", choices=["none", "mean_pooled", "convolutional_attention", "bottom_up_attention"], default=["mean_pooled"])
        argument_parser.add_argument("--image_attention_size", type=int)
        argument_parser.add_argument("--image_feature_datasets", nargs="+", choices=["imagenet", "places365"], default=["imagenet"], help="only applicable to mean_pooled or convolutional_attention options for --image_feature_type")
        argument_parser.add_argument("--bottom_up_detections", type=int, default=20)
        argument_parser.add_argument("--bottom_up_detection_embedding_size", type=int, default=20)
        argument_parser.add_argument("--downscale_convolutional_features", action='store_true')

    def get_name(self):
        raise NotImplementedError("get_name")

    def batch_features(self, feature_list):
        features = np.stack(feature_list)
        return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

    def get_features(self, state):
        raise NotImplementedError("get_features")

class NoImageFeatures(ImageFeatures):
    feature_dim = ImageFeatures.MEAN_POOLED_DIM

    def __init__(self):
        print('Image features not provided')
        self.features = np.zeros((ImageFeatures.NUM_VIEWS, self.feature_dim), dtype=np.float32)

    def get_features(self, state):
        return self.features

    def get_name(self):
        return "none"

class MeanPooledImageFeatures(ImageFeatures):
    def __init__(self, image_feature_datasets):
        image_feature_datasets = sorted(image_feature_datasets)
        self.image_feature_datasets = image_feature_datasets

        self.mean_pooled_feature_stores = [paths.mean_pooled_feature_store_paths[dataset]
                                           for dataset in image_feature_datasets]
        #self.mean_pooled_feature_stores = [os.path.join('../../img_features/ResNet-152-imagenet.tsv')]
        self.feature_dim = MeanPooledImageFeatures.MEAN_POOLED_DIM * len(image_feature_datasets)
        print('Loading image features from %s' % ', '.join(self.mean_pooled_feature_stores))
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
        self.features = defaultdict(list)
        for mpfs in self.mean_pooled_feature_stores:
            with open(mpfs, "rt") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
                for item in reader:
                    assert int(item['image_h']) == ImageFeatures.IMAGE_H
                    assert int(item['image_w']) == ImageFeatures.IMAGE_W
                    assert int(item['vfov']) == ImageFeatures.VFOV
                    long_id = self._make_id(item['scanId'], item['viewpointId'])
                    features = np.frombuffer(decode_base64(item['features']), dtype=np.float32).reshape((ImageFeatures.NUM_VIEWS, ImageFeatures.MEAN_POOLED_DIM))  # TODO by wlt
                    self.features[long_id].append(features)
        assert all(len(feats) == len(self.mean_pooled_feature_stores) for feats in self.features.values())
        self.features = {
            long_id: np.concatenate(feats, axis=1)
            for long_id, feats in self.features.items()
        }

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def get_features(self, state):
        long_id = self._make_id(state.scanId, state.viewpointId)
        # Return feature of all the 36 views
        return self.features[long_id]

    def get_name(self):
        name = '+'.join(sorted(self.image_feature_datasets))
        name = "{}_mean_pooled".format(name)
        return name

class ConvolutionalImageFeatures(ImageFeatures):
    feature_dim = ImageFeatures.MEAN_POOLED_DIM

    def __init__(self, image_feature_datasets, split_convolutional_features=True, downscale_convolutional_features=True):
        self.image_feature_datasets = image_feature_datasets
        self.split_convolutional_features = split_convolutional_features
        self.downscale_convolutional_features = downscale_convolutional_features

        self.convolutional_feature_stores = [paths.convolutional_feature_store_paths[dataset]
                                             for dataset in image_feature_datasets]

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    @functools.lru_cache(maxsize=3000)
    def _get_convolutional_features(self, scanId, viewpointId, viewIndex):
        feats = []
        for cfs in self.convolutional_feature_stores:
            if self.split_convolutional_features:
                path = os.path.join(cfs, scanId, "{}_{}{}.npy".format(viewpointId, viewIndex, "_downscaled" if self.downscale_convolutional_features else ""))
                this_feats = np.load(path)
            else:
                # memmap for loading subfeatures
                path = os.path.join(cfs, scanId, "%s.npy" % viewpointId)
                mmapped = np.load(path, mmap_mode='r')
                this_feats = mmapped[viewIndex,:,:,:]
            feats.append(this_feats)
        if len(feats) > 1:
            return np.concatenate(feats, axis=1)
        return feats[0]

    def get_features(self, state):
        return self._get_convolutional_features(state.scanId, state.location.viewpointId, state.viewIndex)

    def get_name(self):
        name = '+'.join(sorted(self.image_feature_datasets))
        name = "{}_convolutional_attention".format(name)
        if self.downscale_convolutional_features:
            name = name + "_downscale"
        return name

class BottomUpImageFeatures(ImageFeatures):
    PAD_ITEM = ("<pad>",)
    feature_dim = ImageFeatures.MEAN_POOLED_DIM

    def __init__(self, number_of_detections, precomputed_cache_path=None, precomputed_cache_dir=None, image_width=640, image_height=480):
        self.number_of_detections = number_of_detections
        self.index_to_attributes, self.attribute_to_index = BottomUpImageFeatures.read_visual_genome_vocab(paths.bottom_up_attribute_path, BottomUpImageFeatures.PAD_ITEM, add_null=True)
        self.index_to_objects, self.object_to_index = BottomUpImageFeatures.read_visual_genome_vocab(paths.bottom_up_object_path, BottomUpImageFeatures.PAD_ITEM, add_null=False)

        self.num_attributes = len(self.index_to_attributes)
        self.num_objects = len(self.index_to_objects)

        self.attribute_pad_index = self.attribute_to_index[BottomUpImageFeatures.PAD_ITEM]
        self.object_pad_index = self.object_to_index[BottomUpImageFeatures.PAD_ITEM]

        self.image_width = image_width
        self.image_height = image_height

        self.precomputed_cache = {}
        def add_to_cache(key, viewpoints):
            assert len(viewpoints) == ImageFeatures.NUM_VIEWS
            viewpoint_feats = []
            for viewpoint in viewpoints:
                params = {}
                for param_key, param_value in viewpoint.items():
                    if param_key == 'cls_prob':
                        # make sure it's in descending order
                        assert np.all(param_value[:-1] >= param_value[1:])
                    if param_key == 'boxes':
                        # TODO: this is for backward compatibility, remove it
                        param_key = 'spatial_features'
                        param_value = spatial_feature_from_bbox(param_value, self.image_height, self.image_width)
                    assert len(param_value) >= self.number_of_detections
                    params[param_key] = param_value[:self.number_of_detections]
                viewpoint_feats.append(BottomUpViewpoint(**params))
            self.precomputed_cache[key] = viewpoint_feats

        if precomputed_cache_dir:
            self.precomputed_cache = {}
            import glob
            for scene_dir in glob.glob(os.path.join(precomputed_cache_dir, "*")):
                scene_id = os.path.basename(scene_dir)
                pickle_file = os.path.join(scene_dir, "d={}.pkl".format(number_of_detections))
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    for (viewpoint_id, viewpoints) in data.items():
                        key = (scene_id, viewpoint_id)
                        add_to_cache(key, viewpoints)
        elif precomputed_cache_path:
            self.precomputed_cache = {}
            with open(precomputed_cache_path, 'rb') as f:
                data = pickle.load(f)
                for (key, viewpoints) in data.items():
                    add_to_cache(key, viewpoints)

    @staticmethod
    def read_visual_genome_vocab(fname, pad_name, add_null=False):
        # one-to-many mapping from indices to names (synonyms)
        index_to_items = []
        item_to_index = {}
        start_ix = 0
        items_to_add = [pad_name]
        if add_null:
            null_tp = ()
            items_to_add.append(null_tp)
        for item in items_to_add:
            index_to_items.append(item)
            item_to_index[item] = start_ix
            start_ix += 1

        with open(fname) as f:
            for index, line in enumerate(f):
                this_items = []
                for synonym in line.split(','):
                    item = tuple(synonym.split())
                    this_items.append(item)
                    item_to_index[item] = index + start_ix
                index_to_items.append(this_items)
        assert len(index_to_items) == max(item_to_index.values()) + 1
        return index_to_items, item_to_index

    def batch_features(self, feature_list):
        def transform(lst, wrap_with_var=True):
            features = np.stack(lst)
            x = torch.from_numpy(features)
            if wrap_with_var:
                x = Variable(x, requires_grad=False)
            return try_cuda(x)

        return BottomUpViewpoint(
            cls_prob=transform([f.cls_prob for f in feature_list]),
            image_features=transform([f.image_features for f in feature_list]),
            attribute_indices=transform([f.attribute_indices for f in feature_list]),
            object_indices=transform([f.object_indices for f in feature_list]),
            spatial_features=transform([f.spatial_features for f in feature_list]),
            no_object_mask=transform([f.no_object_mask for f in feature_list], wrap_with_var=False),
        )

    def parse_attribute_objects(self, tokens):
        parse_options = []
        # allow blank attribute, but not blank object
        for split_point in range(0, len(tokens)):
            attr_tokens = tuple(tokens[:split_point])
            obj_tokens = tuple(tokens[split_point:])
            if attr_tokens in self.attribute_to_index and obj_tokens in self.object_to_index:
                parse_options.append((self.attribute_to_index[attr_tokens], self.object_to_index[obj_tokens]))
        assert parse_options, "didn't find any parses for {}".format(tokens)
        # prefer longer objects, e.g. "electrical outlet" over "electrical" "outlet"
        return parse_options[0]

    @functools.lru_cache(maxsize=20000)
    def _get_viewpoint_features(self, scan_id, viewpoint_id):
        if self.precomputed_cache:
            return self.precomputed_cache[(scan_id, viewpoint_id)]

        fname = os.path.join(paths.bottom_up_feature_store_path, scan_id, "{}.p".format(viewpoint_id))
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        viewpoint_features = []
        for viewpoint in data:
            top_indices = k_best_indices(viewpoint['cls_prob'], self.number_of_detections, sorted=True)[::-1]

            no_object = np.full(self.number_of_detections, True, dtype=np.uint8) # will become torch Byte tensor
            no_object[0:len(top_indices)] = False

            cls_prob = np.zeros(self.number_of_detections, dtype=np.float32)
            cls_prob[0:len(top_indices)] = viewpoint['cls_prob'][top_indices]
            assert cls_prob[0] == np.max(cls_prob)

            image_features = np.zeros((self.number_of_detections, ImageFeatures.MEAN_POOLED_DIM), dtype=np.float32)
            image_features[0:len(top_indices)] = viewpoint['features'][top_indices]

            spatial_feats = np.zeros((self.number_of_detections, 5), dtype=np.float32)
            spatial_feats[0:len(top_indices)] = spatial_feature_from_bbox(viewpoint['boxes'][top_indices], self.image_height, self.image_width)

            object_indices = np.full(self.number_of_detections, self.object_pad_index)
            attribute_indices = np.full(self.number_of_detections, self.attribute_pad_index)

            for i, ix in enumerate(top_indices):
                attribute_ix, object_ix = self.parse_attribute_objects(list(viewpoint['captions'][ix].split()))
                object_indices[i] = object_ix
                attribute_indices[i] = attribute_ix

            viewpoint_features.append(BottomUpViewpoint(cls_prob, image_features, attribute_indices, object_indices, spatial_feats, no_object))
        return viewpoint_features

    def get_features(self, state):
        viewpoint_features = self._get_viewpoint_features(state.scanId, state.location.viewpointId)
        return viewpoint_features[state.viewIndex]

    def get_name(self):
        return "bottom_up_attention_d={}".format(self.number_of_detections)

class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features,
        using an adjacency dictionary to replace the MatterSim simulator
    '''

    def __init__(self, adj_dict=None):
        self.adj_dict = adj_dict
        self.label_set = json.load(open('data/labels/all_labels.json'))
        #self.label_set = load_datas()
        self.room_label = json.load(open('data/labels/reverie_room/house_pano_info.json'))
        assert adj_dict is not None, "Error! No adjacency dictionary!"

    def get_start_state(self, scan_ids, viewpoint_ids, headings):
        def f(scan_id, viewpoint_id, heading):
            elevation = 0
            view_index = (12 * round(elevation / ANGLE_INC + 1)
                          + round(heading / ANGLE_INC) % 12)
            return WorldState(scanId=scan_id,
                              viewpointId=viewpoint_id,
                              viewIndex=view_index,
                              heading=heading,
                              elevation=elevation)

        return structured_map(f, scan_ids, viewpoint_ids, headings)

    def get_view_labels(self,world_state):
        query = '_'.join([world_state.scanId,
                              world_state.viewpointId])
        result = []
        for i in range(0,36):
            query1 = query+f'_{i}'
            try:
                labels = self.label_set[query1]
            except:
                labels = [0]
            result.append(labels)
        return result


    def get_adjs(self, world_states):
        def f(world_state):
            query = '_'.join([world_state.scanId,
                              world_state.viewpointId,
                              str(world_state.viewIndex)])
            return self.adj_dict[query]

        return structured_map(f, world_states)

    def make_actions(self, world_states, actions, attrs):
        def f(world_state, action, loc_attrs):
            if action == 0:
                return world_state
            else:
                loc_attr = loc_attrs[action]
                return WorldState(scanId=world_state.scanId,
                                  viewpointId=loc_attr['nextViewpointId'],
                                  viewIndex=loc_attr['absViewIndex'],
                                  heading=(
                                      loc_attr['absViewIndex'] % 12) * ANGLE_INC,
                                  elevation=(
                                      loc_attr['absViewIndex'] // 12 - 1)
                                  * ANGLE_INC)

        return structured_map(f, world_states, actions, attrs)

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, image_features_list, batch_size=100, seed=10, splits=['train'], tokenizer=None, beam_size=1, instruction_limit=None):
        self.image_features_list = image_features_list
        self.data = []
        self.scans = []
        self.gt = {}
        self.tokenizer = tokenizer
        for item in load_datasets(splits):
            # Split multiple instructions into separate entries
            assert item['path_id'] not in self.gt
            self.gt[item['path_id']] = item
            instructions = item['instructions']
            if instruction_limit:
                instructions = instructions[:instruction_limit]
            for j,instr in enumerate(instructions):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instructions'] = instr
                if tokenizer:
                    self.tokenizer = tokenizer
                    new_item['instr_encoding'], new_item['instr_length'] = tokenizer.encode_sentence(instr)
                else:
                    self.tokenizer = None
                self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.instr_id_to_idx = {}
        for i,item in enumerate(self.data):
            self.instr_id_to_idx[item['instr_id']] = i
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        self.beam_size=beam_size
#        self.set_beam_size(beam_size)
        self._init_env()
        self.print_progress = False
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))
        self.batch = self.data
        self.notTest = ('test' not in splits)
    def _init_env(self):
        adj_list_file = os.path.join(file_path,'data/total_adj_list.json')
        with open(adj_list_file, 'r') as f:
            adj_dict = json.load(f)
        self.env = EnvBatch(adj_dict)

    # def set_beam_size(self, beam_size, force_reload=False):
    #     # warning: this will invalidate the environment, self.reset() should be called afterward!
    #     try:
    #         invalid = (beam_size != self.beam_size)
    #     except:
    #         invalid = True
    #     if force_reload or invalid:
    #         self.beam_size = beam_size
    #         self.env = EnvBatch(self.batch_size, beam_size)

    def _load_nav_graphs(self):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def get_path_len(self, scanId, path):
        prev = path[0]
        path_len = 0
        for curr in path[1:]:
            path_len += self.distances[scanId][prev][curr]
        return path_len

    def _next_minibatch(self, sort_instr_length):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if self.print_progress:
            sys.stderr.write("\rix {} / {}".format(self.ix, len(self.data)))
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            for i,item in enumerate(self.data):
                self.instr_id_to_idx[item['instr_id']] = i
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        if sort_instr_length:
            batch = sorted(batch, key=lambda item: item['instr_length'], reverse=True)
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        self.ix = 0

    def _shortest_path_action(self, state, adj_loc_list, goalViewpointId):
        '''
        Determine next action on the shortest path to goal,
        for supervised training.
        '''
        if state.viewpointId == goalViewpointId:
            return 0  # do nothing
        path = self.paths[state.scanId][state.viewpointId][
            goalViewpointId]
        nextViewpointId = path[1]
        for n_a, loc_attr in enumerate(adj_loc_list):
            if loc_attr['nextViewpointId'] == nextViewpointId:
                return n_a

        # Next nextViewpointId not found! This should not happen!
        print('adj_loc_list:', adj_loc_list)
        print('nextViewpointId:', nextViewpointId)
        long_id = '{}_{}'.format(state.scanId, state.viewpointId)
        print('longId:', long_id)
        raise Exception('Bug: nextViewpointId not in adj_loc_list')

    def _deviation(self, state, path):
        all_paths = self.paths[state.scanId][state.viewpointId]
        near_id = path[0]
        near_d = len(all_paths[near_id])
        for item in path:
            d = len(all_paths[item])
            if d < near_d:
                near_id = item
                near_d = d
        return near_d - 1 # MUST - 1

    def _distance(self, state, shortest_path):
        goalViewpointId = shortest_path[-1]
        return self.distances[state.scanId][state.viewpointId][goalViewpointId]

    def _progress(self, state, shortest_path):
        goalViewpointId = shortest_path[-1]
        if state.viewpointId == goalViewpointId:
            return 1.0
        shortest_path_len = len(shortest_path) - 1
        path = self.paths[state.scanId][state.viewpointId][
            goalViewpointId]
        path_len = len(path) - 1
        return 1.0 - float(path_len) / shortest_path_len

    def observe(self, world_states, beamed=False, include_teacher=True, instr_id=None):
        #start_time = time.time()
        obs = []
        adj_loc_lists = self.env.get_adjs(world_states)
        for i,(state,adj_loc_list) in enumerate(zip(world_states,adj_loc_lists)):
            item = self.batch[i]
            obs_batch = []
            if item['scan'] != state.scanId:
                item = self.data[self.instr_id_to_idx[instr_id]]
                assert item['scan'] == state.scanId
            feature = [featurizer.get_features(state) for featurizer in self.image_features_list]
            assert len(feature) == 1, 'for now, only work with MeanPooled feature'
            feature_with_loc = np.concatenate((feature[0], _static_loc_embeddings[state.viewIndex]), axis=-1)
            action_embedding = _build_action_embedding(adj_loc_list, feature[0])
            ob = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : [feature_with_loc],
                'adj_loc_list' : adj_loc_list,
                'action_embedding': action_embedding,
                'instructions' : item['instructions'],
                'labels':self.env.get_view_labels(state),
                'room_type':self.env.room_label[state.scanId][state.viewpointId]
            }
            if include_teacher and self.notTest:
                ob['teacher'] = self._shortest_path_action(state, adj_loc_list, item['path'][-1])
                ob['deviation'] = self._deviation(state, item['path'])
                ob['progress']= self._progress(state, item['path']),
                ob['distance']= self._distance(state, item['path']),
                ob['end_room_type'] =self.env.room_label[state.scanId][item['path'][-1]]
                ob = self.compute_angle_differ(ob)
            if 'instr_encoding' in item:
                ob['instr_encoding'] = item['instr_encoding']
            if 'instr_length' in item:
                ob['instr_length'] = item['instr_length']
            obs_batch.append(ob)
            if beamed:
                obs.append(obs_batch)
            else:
                assert len(obs_batch) == 1
                obs.append(obs_batch[0])

        #end_time = time.time()
        #print("get obs in {} seconds".format(end_time - start_time))
        return obs

    def compute_angle_differ(self,ob):
        teacher = ob['teacher']
        target_index = ob['adj_loc_list'][teacher]['absViewIndex']
        if target_index == -1:
            for i,adj in enumerate(ob['adj_loc_list']):
                if i == teacher:
                    adj['angle_weight'] = 0
                else:
                    adj['angle_weight'] = 0.5
        else:
            for i, adj in enumerate(ob['adj_loc_list']):
                if i == teacher:
                    adj['angle_weight'] = 0
                    continue
                if adj['absViewIndex'] == -1:
                    adj['angle_weight'] = 0.5
                else:
                    index_differ =  abs(target_index - adj['absViewIndex']) %12
                    if index_differ > 6:
                        index_differ = 12 - index_differ
                    angle_differ = index_differ *ANGLE_INC
                    score = 1/2 - math.cos(angle_differ) / 2
                    adj['angle_weight'] = score
                    adj['index_differ'] = index_differ
        return ob

    def get_starting_world_states(self, instance_list, beamed=False):
        scanIds = [item['scan'] for item in instance_list]
        viewpointIds = [item['path'][0] for item in instance_list]
        headings = [item['heading'] for item in instance_list]
        return self.env.get_start_state(scanIds, viewpointIds, headings)

    def reset(self, sort=False, beamed=False, load_next_minibatch=True):
        ''' Load a new minibatch / episodes. '''
        if load_next_minibatch:
            self._next_minibatch(sort)
        assert len(self.batch) == self.batch_size
        return self.get_starting_world_states(self.batch, beamed=beamed)

    def step(self, world_states, actions, last_obs, beamed=False):
        ''' Take action (same interface as makeActions) '''
        attrs = [ob['adj_loc_list'] for ob in last_obs]
        return self.env.make_actions(world_states, actions, attrs)

    def shortest_paths_to_goals(self, starting_world_states, max_steps):
        world_states = starting_world_states
        obs = self.observe(world_states)

        all_obs = []
        all_actions = []
        for ob in obs:
            all_obs.append([ob])
            all_actions.append([])

        ended = np.array([False] * len(obs))
        for t in range(max_steps):
            actions = [ob['teacher'] for ob in obs]
            world_states = self.step(world_states, actions, obs)
            obs = self.observe(world_states)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    all_obs[i].append(ob)
            for i,a in enumerate(actions):
                if not ended[i]:
                    all_actions[i].append(a)
                    if a == 0:
                        ended[i] = True
            if ended.all():
                break
        return all_obs, all_actions

    def gold_obs_actions_and_instructions(self, max_steps, load_next_minibatch=True):
        starting_world_states = self.reset(load_next_minibatch=load_next_minibatch)
        path_obs, path_actions = self.shortest_paths_to_goals(starting_world_states, max_steps)
        encoded_instructions = [obs[0]['instr_encoding'] for obs in path_obs]
        return path_obs, path_actions, encoded_instructions


class REVERIEBatch(R2RBatch):
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, image_features_list, batch_size=100, seed=10, splits=['train'], tokenizer=None, beam_size=1, instruction_limit=None):
        self.image_features_list = image_features_list
        self.data = []
        self.scans = []
        self.gt = {}
        self.tokenizer = tokenizer
        for item in load_datasets(splits,type_='REVERIE'):
            # Split multiple instructions into separate entries
            assert item['id'] not in self.gt
            self.gt[item['id']] = item
            instructions = item['instructions']
            if instruction_limit:
                instructions = instructions[:instruction_limit]
            for j,instr in enumerate(instructions):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['id'], j)
                new_item['instructions'] = instr
                if tokenizer:
                    self.tokenizer = tokenizer
                    new_item['instr_encoding'], new_item['instr_length'] = tokenizer.encode_sentence(instr)
                else:
                    self.tokenizer = None
                self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.instr_id_to_idx = {}
        for i,item in enumerate(self.data):
            self.instr_id_to_idx[item['instr_id']] = i
        self.ix = 0
        self.batch_size = batch_size
        self.beam_size=beam_size
        self._load_nav_graphs()
        self._init_env()
        self.print_progress = False
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))
        self.batch = self.data
        self.notTest = ('test' not in splits)

