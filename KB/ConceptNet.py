import json
import os
import time
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
file_path = os.path.dirname(__file__)
base_path = os.path.join(file_path,'..')
sys.path.append(file_path)
sys.path.append(base_path)
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import requests
import torch
from torch import nn
from utils import try_cuda
from exceptions import EntityUnavailable, EmbeddingNotFound
from logger import logging
import math
import numpy as np
_Relation = Dict[str, Union[str, float]]  # {'@id': str, 'weight': float}
Relation = Dict[str, Union[str, List[_Relation]]]  # {'@id': str, 'related': List[_Relation]}

EMB_SIZE = 300


def _cache_path(filename: str) -> str:
    return os.path.join(file_path,'cache', filename)


def _has_cache(filepath: str) -> bool:
    return os.path.exists(filepath)


def _data_path(filepath: str) -> str:
    return os.path.join(file_path, 'data', filepath)


class Manager:
    BASE_URL = 'http://api.conceptnet.io/related{0}'
    LANG = '/c/en'
    PARAMS = {'filter': LANG, 'limit': 50}

    @classmethod
    def get_uri(cls, entity: str) -> str:
        entity = entity.replace(' ', '_')
        return f'{cls.LANG}/{entity}'

    @classmethod
    def parse_uri(cls, uri: str) -> str:
        assert uri.startswith(cls.LANG + '/')
        uri = uri[len(cls.LANG) + 1:]
        return uri.replace('_', ' ')

    def add_cache(self, obj: Relation):
        self.cache[obj['@id']] = obj

    def __init__(self, cache_filename: str = 'conceptnet.cache'):
        self.cache_path = _cache_path(cache_filename)
        self.cache: Dict[str, Relation] = {}  # @id -> obj
        if not _has_cache(self.cache_path):
            logging.info(f"Cache file {self.cache_path} found, creating.")
            with open(self.cache_path, 'w'): pass

        logging.info(f"Loading cache file {self.cache_path}.")
        with open(self.cache_path, 'r') as cache:
            for line in cache:
                obj = json.loads(line)
                self.add_cache(obj)
        logging.info(f"Loaded {len(self.cache)} entities.")

    def get(self, entity: str) -> Dict[str, Any]:
        uri = self.get_uri(entity)
        if uri in self.cache:
            return self.cache[uri]
        url = self.BASE_URL.format(uri)
        logging.info(f"Requesting {url}")
        try:
            obj = requests.get(url, params = self.PARAMS)
        except Exception as e:
            logging.error(f"Failed to get {uri} with Exception {e}.")
            raise EntityUnavailable(uri)
        if obj is None:
            raise EntityUnavailable(uri)
        obj = obj.json()
        with open(self.cache_path, 'a') as cache:
            json.dump(obj, cache)
            cache.write('\n')
        self.add_cache(obj)
        time.sleep(0.5)
        return obj


class KnowledgeGraph:
    _manager = Manager()

    def _load_core_entities(self):
        with open(self.entity_path, 'r') as entity_file:
            self.core_entities = [entity[:-1].split(',')[0] 
                                  for entity in entity_file]

    def _load_graph(self, weight_thresh: Optional[float] = 0, max_degree: Optional[int] = 50):
        logging.info("Loading graph.")
        self.graph = nx.Graph()
        for entity in self.core_entities:
            obj = self._manager.get(entity)
            entity: int = self._maybe_register_entity(entity)
            self.graph.add_edge(entity, entity, weight=1)
            for i, rel in enumerate(obj['related']):
                id_ = self._manager.parse_uri(rel['@id'])
                weight = float(rel['weight'])
                if weight < weight_thresh or i >= max_degree: continue
                id_ = self._maybe_register_entity(id_)
                if not self._is_core(id_):
                    weight *= 0.1
                self.graph.add_edge(entity, id_, weight=weight)
                self.graph.add_edge(id_, entity, weight=weight)

    def _load_edge_and_weights(self):
        edges = self.graph.edges(data='weight')
        logging.info(f"Loaded {len(edges)} edges.")
        edges = list(zip(*edges))
        self.edges = torch.LongTensor(edges[:2]).cuda()
        self.weights = torch.Tensor(edges[2]).cuda()

    def _load_embeddings(self, emb_size: int = EMB_SIZE):
        if _has_cache(self.embedding_cache_path):
            logging.info(f"Loading embedding cache {self.embedding_cache_path}.")
            with open(self.embedding_cache_path, 'rb') as cache:
                self.embeddings = pickle.load(cache).cuda()
            if self.embeddings.shape == (len(self.ent2ind), emb_size):
                return
            logging.warning(f"Cache file is invalid. Saved shape {self.embeddings.shape}, "
                            f"while needed {(len(self.ent2ind), emb_size)}.")

        logging.info(f"Loading embedding file {self.embedding_path}.")
        unknown = '<unk>'
        needed_entities = {unknown}
        for ent in self.ent2ind.keys():
            needed_entities = needed_entities.union(ent.split())

        embeddings: Dict[str, List[float]] = {}
        with open(self.embedding_path, 'r') as emb_file:
            for line_no, line in enumerate(emb_file):
                if line_no % 100000 == 0:
                    logging.info(f"Loaded line {line_no}.")
                line = line.split(' ')
                assert len(line) == emb_size + 1, f"Invalid embedding in file {self.embedding_path} line {line_no}."
                entity = line[0]
                if entity not in needed_entities:
                    continue
                embeddings[entity] = torch.Tensor([float(e) for e in line[1:]]).cuda()
        if unknown not in embeddings:
            raise EmbeddingNotFound(unknown)
        for entity in needed_entities:
            if entity not in embeddings:
                embeddings[entity] = embeddings[unknown]

        self.embeddings = torch.zeros((len(self.ent2ind), emb_size), requires_grad=False).cuda()
        for ent, ind in self.ent2ind.items():
            for word in ent.split(' '):
                self.embeddings[ind] += embeddings[word]

        logging.info(f"Saving embedding cache to {self.embedding_cache_path}.")
        with open(self.embedding_cache_path, 'wb') as cache:
            pickle.dump(self.embeddings, cache)

    def __init__(self, weight_thresh: Optional[float] = 0,
                 max_degree: Optional[int] = 50,
                 entity_filename: str = 'entities.txt', 
                 embedding_filename: str = 'glove.840B.300d.txt', 
                 embedding_cache_filename: str = 'embedding.cache'):
        self.entity_path = _data_path(entity_filename)
        self.embedding_path = _data_path(embedding_filename)
        self.embedding_cache_path = _cache_path(embedding_cache_filename)

        self._load_core_entities()
        self.ent2ind: Dict[str, int] = {ent: ind for ind, ent in enumerate(self.core_entities)}

        self._load_graph(weight_thresh=weight_thresh, 
                         max_degree=max_degree)
        self._load_edge_and_weights()
        self._load_embeddings()

        logging.info(f"Built KG with {len(self.ent2ind)} entities.")

    def _maybe_register_entity(self, entity: str) -> int:
        if entity not in self.ent2ind:
            self.ent2ind[entity] = len(self.ent2ind)
        return self.ent2ind[entity]

    def _is_core(self, ind: int) -> bool:
        return ind < len(self.core_entities)

    def plot(self):
        logging.info("Plotting knowledge graph.")
        graph = nx.Graph()
        edges = [e for e in self.graph.edges if e[0] % 10 == 0 and e[1] % 10 == 0]  # sample 1/100 nodes
        graph.add_edges_from(edges)
        is_core = lambda node: node < len(self.core_entities)
        node_color = ['r' if is_core(node) else 'b' for node in graph]
        edge_color = ['r' if is_core(node1) and is_core(node2) else 'b'
                      for node1, node2 in graph.edges()]
        nx.draw(graph, node_size=1, node_color=node_color, edge_color=edge_color)
        plt.savefig('knowledge_graph.png')
        logging.info("Knowledge graph saved.")



class AttenLayer(nn.Module):
    def __init__(self,shape1=300,shape2=300,shape_common=300):
        super().__init__()
        self.map_weight1 = nn.Linear(shape1,shape_common,bias=False)
        self.map_weight2 = nn.Linear(shape2,shape_common,bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,x1,x2):
        #input1 x1 b*n*emb_shape1
        #input2 x2 b*m*emb_shape2
        #return b*n*n
        x1 = self.map_weight1(x1)
        x2 = self.map_weight2(x2)
        attn_weigtht = torch.bmm(x1,x2.transpose(1,2))
        return self.softmax(attn_weigtht)



class GCNConv(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim,out_dim,bias=False)
    def forward(self,node_emb,edges,edge_weight):
        node_emb =self.weight(node_emb)
        num_nodes = node_emb.shape[0]
        edge_mitrix = torch.sparse.FloatTensor(edges,edge_weight,(num_nodes,num_nodes))
        #edge_mitrix = gcn_norm(edge_mitrix)
        return torch.sparse.mm(edge_mitrix, node_emb)



class GCN_pre_define(nn.Module):
    def __init__(self,emb_size,process_num=3,max_degree=10,short_cut=False,load_adj_weight=True):
        super().__init__()
        self.kg = KnowledgeGraph(max_degree=max_degree)
        if load_adj_weight:
            adj_weights = np.zeros((1601,1601))
            adj_weights[1:,1:]= np.load(os.path.join(file_path,'data/relations.npy'))
        else:
            adj_weights = np.random.rand(1601,1601)
        self.in_adj = torch.nn.Parameter(torch.from_numpy(adj_weights).float(), requires_grad=True)
        self.gcns = []
        self.process_num = process_num
        self.short_cut = short_cut
        self.embeddings = try_cuda(self.kg.embeddings)
        for i in range(process_num):
            gcn = GCNConv(emb_size,emb_size)
            self.__setattr__(f'gcn{i}',gcn)
            self.gcns.append(gcn)
        self.inside_forward_weight=[]
        for i in range(process_num):
            linear = torch.nn.Linear(300,300,bias=False)
            self.__setattr__(f'linear{i}',linear)
            self.inside_forward_weight.append(linear)
        self.zero_tensor = try_cuda(torch.zeros(1,300))
    
    def norm_matrix(self,matrix):
        to_add = try_cuda(torch.diag(torch.ones(matrix.shape[1])))
        matrix = matrix + to_add        
        batch_size = matrix.shape[0]
        D = matrix.sum(axis=1)
        D = D[:,:,None]
        D = D.pow(-0.5)
        matrix = torch.mul(matrix,D.reshape(batch_size,1,-1))
        matrix = torch.mul(matrix,D.reshape(batch_size,-1,1))
        return matrix
    
    def forward(self,label_set):
        batch_size,label_num = label_set.shape
        labels = label_set.reshape(-1)
        gather_matrix = self.in_adj[labels]
        gather_index = label_set[:,None,:].expand(-1,label_num,-1).reshape(-1,label_num)
        neightbor_matrix = torch.gather(gather_matrix,dim=1,index=gather_index)
        neightbor_matrix = neightbor_matrix.reshape(batch_size,label_num,label_num)
        neightbor_matrix = torch.sigmoid(neightbor_matrix)
        norm_matrix = self.norm_matrix(neightbor_matrix)
        next_ = self.embeddings
        embeddings = self.find_embeddings(label_set,self.embeddings)
        start_embeddings = embeddings
        for i in range(self.process_num):
            gcn_embeddings,next_ = self.gcn_forward(label_set,next_,i)
            embeddings = torch.bmm(norm_matrix,embeddings)
            embeddings =self.inside_forward_weight[i](embeddings)
            embeddings = (embeddings +gcn_embeddings)/2
            embeddings = torch.relu(embeddings)
        if self.short_cut:
            embeddings = (embeddings+start_embeddings)/2
        return embeddings

    def gcn_forward(self,label_set,kg_embeddings,count=0):
        gcn = self.gcns[count]
        embeddings = gcn(kg_embeddings, self.kg.edges, self.kg.weights)
        return self.find_embeddings(label_set,embeddings),embeddings

    def find_embeddings(self,labels,embeddings):
        origin_shape = labels.shape
        labels = labels.reshape(-1)
        embeddings = torch.cat([self.zero_tensor,embeddings],axis=0)
        embeding_features = embeddings[labels]
        embeding_features = embeding_features.reshape((*origin_shape,-1)) #init_value
        return embeding_features