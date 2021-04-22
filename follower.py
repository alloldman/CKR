''' Agents: stop/random/shortest/seq2seq  '''

import json
import sys
import numpy as np
import networkx as nx
import random
from collections import namedtuple, defaultdict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.utils.rnn as rnn_utils
from utils import vocab_pad_idx, vocab_eos_idx, flatten, structured_map, try_cuda
from utils import PriorityQueue,PriorityQueueWithDistance
from running_mean_std import RunningMean
import torch.nn.functional as F
InferenceState = namedtuple("InferenceState", "prev_inference_state, world_state, observation, flat_index, last_action, last_action_embedding, action_count, score, h_t, c_t, last_alpha")
SearchState = namedtuple("SearchState", "flogit,flogp, world_state, observation, action, action_embedding, action_count, h_t,c_t,father") # flat_index,
TransformerSearchState = namedtuple("TransformerSearchState", "flogit,flogp, world_state, observation, action, action_embedding, action_count, h_t,c_t,father, history_input, history_length") # flat_index,

CandidateState = namedtuple("CandidateState", "flogit,flogp,world_states,actions") # flat_index,
Cons = namedtuple("Cons", "first, rest")

def cons_to_list(cons):
    l = []
    while True:
        l.append(cons.first)
        cons = cons.rest
        if cons is None:
            break
    return l

def backchain_inference_states(last_inference_state):
    states = []
    observations = []
    actions = []
    inf_state = last_inference_state
    scores = []
    last_score = None
    attentions = []
    while inf_state is not None:
        states.append(inf_state.world_state)
        observations.append(inf_state.observation)
        actions.append(inf_state.last_action)
        attentions.append(inf_state.last_alpha)
        if last_score is not None:
            scores.append(last_score - inf_state.score)
        last_score = inf_state.score
        inf_state = inf_state.prev_inference_state
    scores.append(last_score)
    return list(reversed(states)), list(reversed(observations)), list(reversed(actions))[1:], list(reversed(scores))[1:], list(reversed(attentions))[1:] # exclude start action

def least_common_viewpoint_path(inf_state_a, inf_state_b):
    # return inference states traversing from A to X, then from Y to B,
    # where X and Y are the least common ancestors of A and B respectively that share a viewpointId
    path_to_b_by_viewpoint =  {
    }
    b = inf_state_b
    b_stack = Cons(b, None)
    while b is not None:
        path_to_b_by_viewpoint[b.world_state.viewpointId] = b_stack
        b = b.prev_inference_state
        b_stack = Cons(b, b_stack)
    a = inf_state_a
    path_from_a = [a]
    while a is not None:
        vp = a.world_state.viewpointId
        if vp in path_to_b_by_viewpoint:
            path_to_b = cons_to_list(path_to_b_by_viewpoint[vp])
            assert path_from_a[-1].world_state.viewpointId == path_to_b[0].world_state.viewpointId
            return path_from_a + path_to_b[1:]
        a = a.prev_inference_state
        path_from_a.append(a)
    raise AssertionError("no common ancestor found")

def batch_instructions_from_encoded(encoded_instructions, max_length, reverse=False, sort=False, tok=None, addEos=True):
    # encoded_instructions: list of lists of token indices (should not be padded, or contain BOS or EOS tokens)
    # make sure pad does not start any sentence
    num_instructions = len(encoded_instructions)
    seq_tensor = np.full((num_instructions, max_length), vocab_pad_idx)
    seq_lengths = []
    inst_mask = []
    for i, inst in enumerate(encoded_instructions):
        if len(inst) > 0:
            assert inst[-1] != vocab_eos_idx
        if reverse:
            inst = inst[::-1]
        if addEos:
            inst = np.concatenate((inst, [vocab_eos_idx]))
        inst = inst[:max_length]
        if tok:
            inst_mask.append(tok.filter_verb(inst,sel_verb=False)[1])
        seq_tensor[i,:len(inst)] = inst
        seq_lengths.append(len(inst))

    seq_tensor = torch.from_numpy(seq_tensor)
    if sort:
        seq_lengths, perm_idx = torch.from_numpy(np.array(seq_lengths)).sort(0, True)
        seq_lengths = list(seq_lengths)
        seq_tensor = seq_tensor[perm_idx]
    else:
        perm_idx = np.arange((num_instructions))

    mask = (seq_tensor == vocab_pad_idx)[:, :max(seq_lengths)]

    if tok:
        for i,idx in enumerate(perm_idx):
            mask[i][inst_mask[idx]] = 1

    ret_tp = try_cuda(Variable(seq_tensor, requires_grad=False).long()), \
             try_cuda(mask.byte()), \
             seq_lengths
    if sort:
        ret_tp = ret_tp + (list(perm_idx),)
    return ret_tp

def final_text_enc(encoded_instructions, max_length, encoder):
    seq, seq_mask, seq_lengths = batch_instructions_from_encoded(encoded_instructions, max_length, reverse=False)
    ctx,h_t,c_t = encoder(seq, seq_lengths)
    return h_t, c_t

def stretch_tensor(alphas, lens, target_len):
    ''' Given a batch of sequences of various lengths, stretch to a target_len
        and normalize the sum to 1
    '''
    batch_size, _ = alphas.shape
    r = torch.zeros(batch_size, target_len)
    al = alphas.unsqueeze(1)
    for idx,_len in enumerate(lens):
        r[idx] = F.interpolate(al[idx:idx+1,:,:_len],size=(target_len),mode='linear',align_corners=False)
        r[idx] /= r[idx].sum()
    return r

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_test_results(self):
        results = []
        for key, item in self.results.items():
            results.append({
                'instr_id': item['instr_id'],
                'trajectory': item['trajectory'],
            })
        with open(self.results_path, 'w') as f:
            json.dump(results, f)

    def write_results(self, results=None, results_path=None):
        if results is None:
            results = {}
            for key, item in self.results.items():
                results[key] = {
                    'instr_id': item['instr_id'],
                    'trajectory': item['trajectory'],
                }
        if results_path is None:
            results_path = self.results_path
        with open(results_path, 'w') as f:
            json.dump(results, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        self.clean_results = {}

        # We rely on env showing the entire batch before repeating anything
        #print 'Testing %s' % self.__class__.__name__
        looped = False
        rollout_scores = []
        beam_10_scores = []
        with torch.no_grad():
            while True:
                rollout_results = self.rollout()
                for result in rollout_results:
                    if result['instr_id'] in self.results:
                        looped = True
                    else:
                        self.results[result['instr_id']] = result

                if looped:
                    break
        # if self.feedback == 'argmax':
        #     print("avg rollout score: ", np.mean(rollout_scores))
        #     print("avg beam 10 score: ", np.mean(beam_10_scores))
        return self.results

def path_element_from_observation(ob):
    return (ob['viewpoint'], ob['heading'], ob['elevation'])

def realistic_jumping(graph, start_step, dest_obs):
    if start_step == path_element_from_observation(dest_obs):
        return []
    s = start_step[0]
    t = dest_obs['viewpoint']
    path = nx.shortest_path(graph,s,t)
    traj = [(vp,0,0) for vp in path[:-1]]
    traj.append(path_element_from_observation(dest_obs))
    return traj

class StopAgent(BaseAgent):
    ''' An agent that doesn't move! '''

    def rollout(self):
        world_states = self.env.reset()
        obs = self.env.observe(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob) ]
        } for ob in obs]
        return traj

class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        world_states = self.env.reset()
        obs = self.env.observe(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)]
        } for ob in obs]
        ended = [False] * len(obs)

        self.steps = [0] * len(obs)
        for t in range(6):
            actions = []
            for i, ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append(0)  # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] == 0:
                    a = np.random.randint(len(ob['adj_loc_list']) - 1) + 1
                    actions.append(a)  # choose a random adjacent loc
                    self.steps[i] += 1
                else:
                    assert len(ob['adj_loc_list']) > 1
                    actions.append(1)  # go forward
                    self.steps[i] += 1
            world_states = self.env.step(world_states, actions, obs)
            obs = self.env.observe(world_states)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(path_element_from_observation(ob))
        return traj

class ShortestAgent(BaseAgent):

    def rollout(self):
        world_states = self.env.reset()
        #obs = self.env.observe(world_states)
        all_obs, all_actions = self.env.shortest_paths_to_goals(world_states, 20)
        return [
            {
                'instr_id': obs[0]['instr_id'],
                # end state will appear twice because stop action is a no-op, so exclude it
                'trajectory': [path_element_from_observation(ob) for ob in obs[:-1]]
            }
            for obs in all_obs
        ]

class Seq2SeqAgent(BaseAgent):

    def __init__(self, env, results_path, encoder, decoder, episode_len=10, beam_size=1, reverse_instruction=True, max_instruction_length=80, attn_only_verb=False):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.beam_size = beam_size
        self.reverse_instruction = reverse_instruction
        self.max_instruction_length = max_instruction_length
        self.attn_only_verb = attn_only_verb
        self.gb = None
        self.want_loss = False

    def _feature_variables(self, obs, beamed=False):
        ''' Extract precomputed features into variable. '''
        feature_lists = list(zip(*[ob['feature'] for ob in (flatten(obs) if beamed else obs)]))
        assert len(feature_lists) == len(self.env.image_features_list)
        batched = []
        for featurizer, feature_list in zip(self.env.image_features_list, feature_lists):
            batched.append(featurizer.batch_features(feature_list))
        return batched

    def _action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        max_num_a = -1
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['adj_loc_list']))

        is_valid = np.zeros((len(obs), max_num_a), np.float32)
        action_embedding_dim = obs[0]['action_embedding'].shape[-1]
        action_embeddings = np.zeros(
            (len(obs), max_num_a, action_embedding_dim),
            dtype=np.float32)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']
            num_a = len(adj_loc_list)
            is_valid[i, 0:num_a] = 1.
            #for n_a, adj_dict in enumerate(adj_loc_list):
            #    action_embeddings[i, :num_a, :] = ob['action_embedding']
            action_embeddings[i, :num_a, :] = ob['action_embedding']
        return (
            Variable(torch.from_numpy(action_embeddings), requires_grad=False).cuda(),
            Variable(torch.from_numpy(is_valid), requires_grad=False).cuda(),
            is_valid)

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            a[i] = ob['teacher'] if not ended[i] else -1
        return try_cuda(Variable(a, requires_grad=False))

    def _progress_target(self, obs, ended, monitor_score):
        t = [None] * len(obs)
        num_elem = 0
        for i,ob in enumerate(obs):
            num_elem += int(not ended[i])
            t[i] = ob['progress'][0] if not ended[i] else monitor_score[i].item()
            #t[i] = ob['progress'][0] if not ended[i] else 1.0
        return try_cuda(torch.tensor(t, requires_grad=False)), num_elem

    def _progress_soft_align(self, alpha_t, seq_lengths):
        if not hasattr(self,'dummy'):
            self.dummy = torch.arange(80).float().cuda()
        score = torch.matmul(alpha_t,self.dummy[:alpha_t.size()[-1]])
        score /= torch.tensor(seq_lengths).float().cuda()
        return score

    def _deviation_target(self, obs, ended, computed_score):
        t = [ob['deviation'] if not ended[i] else computed_score[i].item() for i,ob in enumerate(obs)]
        return try_cuda(torch.tensor(t, requires_grad=False).float())

    def _proc_batch(self, obs, beamed=False):
        encoded_instructions = [ob['instr_encoding'] for ob in (flatten(obs) if beamed else obs)]
        tok = self.env.tokenizer if self.attn_only_verb else None
        return batch_instructions_from_encoded(encoded_instructions, self.max_instruction_length, reverse=self.reverse_instruction, tok=tok)

    def rollout(self):
        if hasattr(self,'search') and self.search:
            self.records = defaultdict(list)
            return self._rollout_with_search()
        if self.beam_size == 1:
            return self._rollout_with_loss()
        else:
            assert self.beam_size >= 1
            beams, _, _ = self.beam_search(self.beam_size)
            return [beam[0] for beam in beams]

    def _score_obs_actions_and_instructions(self, path_obs, path_actions, encoded_instructions):
        batch_size = len(path_obs)
        assert len(path_actions) == batch_size
        assert len(encoded_instructions) == batch_size
        for path_o, path_a in zip(path_obs, path_actions):
            assert len(path_o) == len(path_a) + 1

        seq, seq_mask, seq_lengths, perm_indices = \
            batch_instructions_from_encoded(
                encoded_instructions, self.max_instruction_length,
                reverse=self.reverse_instruction, sort=True)
        loss = 0

        ctx,h_t,c_t = self.encoder(seq, seq_lengths)
        u_t_prev = self.decoder.u_begin.expand(batch_size, -1)  # init action
        ended = np.array([False] * batch_size)
        sequence_scores = try_cuda(torch.zeros(batch_size))

        traj = [{
            'instr_id': path_o[0]['instr_id'],
            'trajectory': [path_element_from_observation(path_o[0])],
            'actions': [],
            'scores': [],
            'observations': [path_o[0]],
            'instr_encoding': path_o[0]['instr_encoding']
        } for path_o in path_obs]

        obs = None
        for t in range(self.episode_len):
            next_obs = []
            next_target_list = []
            for perm_index, src_index in enumerate(perm_indices):
                path_o = path_obs[src_index]
                path_a = path_actions[src_index]
                if t < len(path_a):
                    next_target_list.append(path_a[t])
                    next_obs.append(path_o[t])
                else:
                    next_target_list.append(-1)
                    next_obs.append(obs[perm_index])

            obs = next_obs

            target = try_cuda(Variable(torch.LongTensor(next_target_list), requires_grad=False))

            f_t_list = self._feature_variables(obs) # Image features from obs
            all_u_t, is_valid, _ = self._action_variable(obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx, seq_mask)

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')

            # Supervised training
            loss += self.criterion(logit, target)

            # Determine next model inputs
            a_t = torch.clamp(target, min=0)  # teacher forcing
            # update the previous action
            u_t_prev = all_u_t[np.arange(batch_size), a_t, :].detach()

            action_scores = -F.cross_entropy(logit, target, ignore_index=-1, reduction='none').data
            sequence_scores += action_scores

            # Save trajectory output
            for perm_index, src_index in enumerate(perm_indices):
                ob = obs[perm_index]
                if not ended[perm_index]:
                    traj[src_index]['trajectory'].append(path_element_from_observation(ob))
                    traj[src_index]['score'] = float(sequence_scores[perm_index])
                    traj[src_index]['scores'].append(action_scores[perm_index])
                    traj[src_index]['actions'].append(a_t.data[perm_index])
                    # traj[src_index]['observations'].append(ob)

            # Update ended list
            for i in range(batch_size):
                action_idx = a_t[i].item()
                if action_idx == 0:
                    ended[i] = True

            # Early exit if all ended
            if ended.all():
                break

        return traj, loss



    def _search_collect(self, batch_queue, wss, current_idx, ended):
        cand_wss = []
        cand_acs = []
        for idx,_q in enumerate(batch_queue):
            _wss = [wss[idx]]
            _acs = [0]
            _step = current_idx[idx]
            while not ended[idx] and _step > 0:
                _wss.append(_q.queue[_step].world_state)
                _acs.append(_q.queue[_step].action)
                _step = _q.queue[_step].father
            cand_wss.append(list(reversed(_wss)))
            cand_acs.append(list(reversed(_acs)))
        return cand_wss, cand_acs

    def _wss_to_obs(self, cand_wss, instr_ids):
        cand_obs = []
        for _wss,_instr_id in zip(cand_wss, instr_ids):
            ac_len = len(_wss)
            cand_obs.append(self.env.observe(_wss, instr_id=_instr_id))
        return cand_obs


    def _init_loss(self):
        self.loss = 0
        self.ce_loss = 0
        self.pm_loss = 0
        self.bt_loss = 0
        self.dv_loss = 0
        self.angle_reward = 0
        self.room_loss = 0

    def train(self, optimizers,schedulers, n_iters, feedback='teacher',use_angle_distance_loss=False,use_angle_distance_reward=False):
        ''' Train for a given number of iterations '''
        self.not_search()
        self.feedback = feedback
        self.use_angle_distance_loss = use_angle_distance_loss
        self.use_angle_distance_reward = use_angle_distance_reward 
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        self.ce_losses = []
        self.dv_losses = []
        self.pm_losses = []
        self.bt_losses = []
        self.room_losses = []
        self.angle_rewards = []

        it = range(1, n_iters + 1)
        self.search=False
        try:
            import tqdm
            it = tqdm.tqdm(it)
        except:
            pass
        self.attn_t = np.zeros((n_iters,self.episode_len,self.env.batch_size,80))
        for self.it_idx in it:
            for opt in optimizers:
                opt.zero_grad()
            self.rollout()
            #self._rollout_with_loss()
            if type(self.loss) is torch.Tensor:
                self.loss.backward()
            for opt in optimizers:
                opt.step()
            for scheduler in schedulers:
                scheduler.step()

    def is_search(self):
        self.search=True
        self.decoder.is_search()

    def not_search(self):
        self.search=False
        self.decoder.not_search()

    def _encoder_and_decoder_paths(self, base_path):
        return base_path + "_enc", base_path + "_dec"

    def save(self, path,parallel = False):
        ''' Snapshot models '''
        if parallel:
            encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
            torch.save(self.encoder.module.state_dict(), encoder_path)
            torch.save(self.decoder.module.state_dict(), decoder_path)
        else:
            encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
            torch.save(self.encoder.state_dict(), encoder_path)
            torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, path, load_scorer=False, **kwargs):
        ''' Loads parameters (but not training state) '''
        encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
        self.encoder.load_state_dict(torch.load(encoder_path, **kwargs))
        self.decoder.load_state_dict(torch.load(decoder_path, **kwargs))
    def modules(self):
        _m = [self.encoder, self.decoder]
        return _m

    def modules_paths(self, base_path):
        _mp = list(self._encoder_and_decoder_paths(base_path))
        if self.prog_monitor: _mp.append(base_path + "_pm")
        if self.bt_button: _mp += _mp.append(base_path + "_bt")
        return _mp



class TransformerAgent(Seq2SeqAgent):

    def __init__(self, env, results_path, encoder, decoder, episode_len=10, beam_size=1, reverse_instruction=True, 
                max_instruction_length=80, attn_only_verb=False,loss_weight=[0,1,0]):
        super(TransformerAgent,self).__init__(env, results_path, encoder, decoder, episode_len=10, beam_size=1, reverse_instruction=True, max_instruction_length=80, attn_only_verb=False)
        self.loss_weight = loss_weight
        self.use_angle_distance_loss = False
        self.use_angle_distance_reward = False
        self.search = False
        self.search_episode_len = 40

    def compute_label_for_point(self,label_set):
        new_set = []
        for i,list_ in enumerate(label_set):
            new_list = set()
            for x in list_:
                new_list.update(x)
            new_list = list(new_list)
            new_list = torch.LongTensor(new_list)
            new_set.append(new_list)
        new_set = nn.utils.rnn.pad_sequence(new_set,batch_first=True,padding_value=0)
        return try_cuda(new_set)
    def is_search(self):
        self.search=True
        self.decoder.is_search()

    def not_search(self):
        self.search=False
        self.decoder.not_search()
    def compute_label_for_action(self,obs):
        all_list = []
        max_num_a = 0
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['adj_loc_list']))
        for ob in obs:
            list_view = []
            for adj_loc in ob['adj_loc_list']:
                index= adj_loc['absViewIndex']
                label_list = ob['labels'][index]
                list_view.append(try_cuda(torch.LongTensor(label_list)))
            while len(list_view)< max_num_a:
                list_view.append(torch.LongTensor([]))
            all_list.extend(list_view)
        all_list = torch.nn.utils.rnn.pad_sequence(all_list,batch_first=True,padding_value=0)
        return try_cuda(all_list)                        


    def _rollout_with_loss(self):
        initial_world_states = self.env.reset(sort=True)
        initial_obs = self.env.observe(initial_world_states)
        initial_obs = np.array(initial_obs)
        batch_size = len(initial_obs)
        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(initial_obs)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        # TODO consider not feeding this into the decoder, and just using attention

        self._init_loss()
        last_dev = try_cuda(torch.zeros(batch_size).float())
        last_logit = try_cuda(torch.zeros(batch_size).float())
        ce_criterion = self.criterion
        total_num_elem = 0

        # per step loss
        #ce_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        #pm_criterion = nn.MSELoss(reduction='sum')

        feedback = self.feedback

        ctx,h_t,c_t = self.encoder(seq, seq_lengths,seq_mask)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [],
           'actions': [],
            'action_list':[],
            'room_class_view':[],
            'object_labels':[],
            'language_room':[],
            'object_search_value':[]
        } for ob in initial_obs]
        text_atten_weight = [{'instr_id':ob['instr_id'],
                            'instructions':ob['instructions'],
                            'room_attn_weight':0,
                            'object_attn_weight':0
                            } for ob in initial_obs]
        obs = initial_obs
        world_states = initial_world_states

        # Initial action
        try:
            u_t_prev = self.decoder.u_begin.expand(batch_size, -1)  
        except:
            u_t_prev = self.decoder.module.u_begin.expand(batch_size, -1)
        ended = np.array([False] * batch_size) 

        
        env_action = [None] * batch_size
        sequence_scores = try_cuda(torch.zeros(batch_size))

        history_input = None
        for t in range(self.episode_len+1):
            f_t_list = self._feature_variables(obs) 
            all_u_t, is_valid, _ = self._action_variable(obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            
            prev_h_t = h_t
            action_label_set = self.compute_label_for_action(obs)
            object_label_set = self.compute_label_for_point([ob['labels'] for ob in obs])
            h_t,c_t, history_input, t_ground, v_ground, alpha_t, logit, alpha_v,text_room_class,view_room_class,text_obj_atten,text_room_atten,object_index = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], history_input, h_t, ctx, seq_mask,object_label_set,action_label_set)
            if text_obj_atten is not None:
                for i,weight in enumerate(text_atten_weight):
                    weight['room_attn_weight'] += text_room_atten[i].detach().cpu()
                    weight['object_attn_weight'] += text_obj_atten[i].detach().cpu()
            history_input = history_input.detach()
            if text_room_class is not None:
                text_room_target =try_cuda(torch.LongTensor([ob['end_room_type'] for ob in obs]))
                text_room_loss=torch.nn.functional.cross_entropy(text_room_class,text_room_target)
                self.room_loss+=text_room_loss
            if view_room_class is not None:
                room_target = [ob['room_type'] for ob in obs]
                room_target = try_cuda(torch.LongTensor(room_target))
                room_target = room_target[:,None]
                action_number = all_u_t.shape[1]
                room_target = room_target.expand((-1,action_number)) 
                room_target = room_target.reshape(-1)
                loss=torch.nn.functional.cross_entropy(view_room_class,room_target,reduction='none')
                is_valid_ = is_valid.reshape(-1)
                useful = loss[is_valid_.bool()]
                self.room_loss += useful.mean()

            _logit = logit.detach()
            _logit[is_valid == 0] = -float('inf')

            target = self._teacher_action(obs, ended)
            logit_score = F.softmax(logit,dim=1)
            angle_weight = try_cuda(torch.zeros(logit_score.shape))
            for oi,ob in enumerate(obs):
                for aji,adj in enumerate(ob['adj_loc_list']):
                    angle_weight[oi,aji] = adj['angle_weight']
            new_score = logit_score*angle_weight
            new_score = new_score.sum(axis=1).mean()
            self.angle_reward += new_score 
            if self.use_angle_distance_loss:
                loss = F.cross_entropy(logit,target,reduction='none',ignore_index=-1)
                max_logit,index = logit.max(axis=1)
                angle_weight = try_cuda(torch.zeros(loss.shape))
                for io,ob in enumerate(obs):
                    angle_weight[io] = ob['adj_loc_list'][index[io]]['angle_weight']
                loss = loss*angle_weight
                self.ce_loss += loss.mean()
            else:
                self.ce_loss += ce_criterion(logit, target)            
            total_num_elem += np.size(ended) - np.count_nonzero(ended)

            if feedback == 'teacher':
                a_t = torch.clamp(target, min=0)
            elif feedback == 'sample':
                m = D.Categorical(logits=_logit)
                a_t = m.sample()
            elif feedback == 'argmax':
                _,a_t = _logit.max(1)       
                a_t = a_t.detach()
            elif feedback == 'recover':
                m = D.Categorical(logits=_logit)
                a_t = m.sample()
                for i,ob in enumerate(obs):
                    if ob['deviation'] > 0:
                        a_t[i] = -1
            else:
                if 'sample' in feedback:
                    m = D.Categorical(logits=_logit)
                    a_t = m.sample()
                elif 'argmax' in feedback:
                    _,a_t = _logit.max(1)
                else:
                    import pdb;pdb.set_trace()
                deviation = int(''.join([n for n in feedback if n.isdigit()]))
                for i,ob in enumerate(obs):
                    if ob['deviation'] >= deviation:
                        a_t[i] = target[i]
                a_t = torch.clamp(a_t, min=0)
            u_t_prev = all_u_t[np.arange(batch_size), a_t, :].detach()
            last_logit = _logit[np.arange(batch_size), a_t]

            action_scores = -F.cross_entropy(_logit, a_t, ignore_index=-1, reduction='none').data
            sequence_scores += action_scores

            for i in range(batch_size):
                action_idx = a_t[i].item()
                env_action[i] = action_idx
            action_num = all_u_t.shape[1]
            view_room_class = view_room_class.reshape(batch_size,action_num,-1)
            view_room_class = view_room_class.argmax(axis=2)
            text_room_class = text_room_class.argmax(axis=1)
            action_count = is_valid.sum(axis=1)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(path_element_from_observation(ob))
                    traj[i]['actions'].append(env_action[i])
                    traj[i]['action_list'].append(ob['adj_loc_list'])
                    traj[i]['room_class_view'].append(view_room_class[i].detach().cpu().tolist()[:int(action_count[i].item())])
                    traj[i]['object_labels'].append(object_label_set[i].detach().cpu().tolist())
                    traj[i]['language_room'].append(text_room_class[i].detach().cpu().tolist())
                    traj[i]['object_search_value'].append(object_index[i].cpu().tolist())
            world_states = self.env.step(world_states, env_action, obs)
            obs = self.env.observe(world_states)            

            for i in range(batch_size):
                action_idx = a_t[i].item()
                if action_idx == 0:
                    ended[i] = True
            if ended.all():
                break
        self.loss +=self.angle_reward*self.loss_weight[0]
        self.angle_rewards.append(self.angle_reward.item())
        self.loss += self.ce_loss*self.loss_weight[1]+self.room_loss*self.loss_weight[3]
        self.losses.append(self.loss.item())
        self.ce_losses.append(self.ce_loss.item())
        self.room_losses.append(self.room_loss.item())
        
        for weight in text_atten_weight:
            weight['room_attn_weight'] =weight['room_attn_weight'].cpu().tolist()
            weight['object_attn_weight'] = weight['object_attn_weight'].cpu().tolist()
        return traj,text_atten_weight

    def _rollout_with_search(self):
        if self.env.notTest:
            self._init_loss()
            # ce_criterion = self.criterion
            # pm_criterion = self.pm_criterion
            # bt_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        world_states = self.env.reset(sort=True)
        obs = self.env.observe(world_states)
        batch_size = len(obs)

        seq, seq_mask, seq_lengths = self._proc_batch(obs)
        ctx,h_t,c_t = self.encoder(seq, seq_lengths,seq_mask)

        traj = [{
            'instr_id': ob['instr_id'],
            'instr_encoding': ob['instr_encoding'],
            'trajectory': [path_element_from_observation(ob)],
        } for ob in obs]

        clean_traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)],
        } for ob in obs]
        if self.distance_aware:
            batch_queue = [PriorityQueueWithDistance(distanceMap=self.env.distances,weight=self.distance_weight) for _ in range(batch_size)]
        else:
            batch_queue = [PriorityQueue() for _ in range(batch_size)]
        ending_queue = [PriorityQueue() for _ in range(batch_size)]

        visit_graphs = [nx.Graph() for _ in range(batch_size)]
        for ob, g in zip(obs, visit_graphs): g.add_node(ob['viewpoint'])

        ended = np.array([False] * batch_size)

        for i, (ws, o) in enumerate(zip(world_states, obs)):
            batch_queue[i].push(
                TransformerSearchState(
                    flogit=RunningMean(),
                    flogp=RunningMean(),
                    world_state=ws,
                    observation=o,
                    action=0,
                    action_embedding=self.decoder.u_begin.view(-1).detach(),
                    history_input = self.decoder.history_begin.detach(),
                    history_length = 0,
                    action_count=0,
                    h_t=h_t[i].detach(),c_t=c_t[i].detach(),
                    father=-1),
                0)

        #state_factored = [{} for _ in range(batch_size)]

        for t in range(self.search_episode_len):
            if self.distance_aware:
                current_idx, priority, current_batch = \
                        zip(*[_q.pop(traj[i]['trajectory'][-1][0]) for i, _q in enumerate(batch_queue)])
            else:
                current_idx, priority, current_batch = \
                        zip(*[_q.pop() for i, _q in enumerate(batch_queue)])
            (last_logit,last_logp,last_world_states,last_obs,acs,acs_embedding,
                    ac_counts,prev_h_t,prev_c_t,prev_father,history_input,history_length) = zip(*current_batch)
            
            if t > 0:
                for i,ob in enumerate(last_obs):
                    if not ended[i]:
                        last_vp = traj[i]['trajectory'][-1]
                        traj[i]['trajectory'] += realistic_jumping(
                            visit_graphs[i], last_vp, ob)

                world_states = self.env.step(last_world_states,acs,last_obs)
                obs = self.env.observe(world_states)
                for i in range(batch_size):
                    if (not ended[i] and
                        not visit_graphs[i].has_edge(last_obs[i]['viewpoint'], obs[i]['viewpoint'])):
                        traj[i]['trajectory'].append(path_element_from_observation(obs[i]))
                        visit_graphs[i].add_edge(last_obs[i]['viewpoint'],
                                             obs[i]['viewpoint'])
                for idx, ac in enumerate(acs):
                    if ac == 0:
                        ended[idx] = True
                        batch_queue[idx].lock()
            (last_logit,last_logp,last_world_states,last_obs,acs,acs_embedding,
                    ac_counts,prev_h_t,prev_c_t,prev_father,history_input,history_length) = zip(*current_batch)
            if ended.all(): break
            u_t_prev = torch.stack(acs_embedding, dim=0)
            prev_h_t = torch.stack(prev_h_t,dim=0)
            prev_c_t = torch.stack(prev_c_t,dim=0)
            history_input = rnn_utils.pad_sequence(
            history_input, batch_first=True, padding_value=0)
            if history_input.dim()<3:
                history_input = history_input[:,None,:]
            f_t_list = self._feature_variables(obs)
            all_u_t, is_valid, _ = self._action_variable(obs)
            # 1. local scorer
            action_label_set = self.compute_label_for_action(obs)
            object_label_set = self.compute_label_for_point([ob['labels'] for ob in obs])
            room_target = [ob['room_type'] for ob in obs]
            room_target = try_cuda(torch.LongTensor(room_target))
            room_target = room_target[:,None]
            action_number = all_u_t.shape[1]
            room_target = room_target.expand((-1,action_number)) 
            room_target = room_target.reshape(-1)
            text_room_target =try_cuda(torch.LongTensor([ob['end_room_type'] for ob in obs]))
            gt_labels = (text_room_target, room_target)
            h_t, c_t,history_input,history_length, t_ground, v_ground, alpha_t, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0],history_input, history_length, prev_h_t, ctx, seq_mask,object_label_set,action_label_set,gt_labels)
            #h_1,c_1,to_input,new_length, text_atten,attn_vision,text_alpha,alpha_action,alpha_vision
            # 2. prog monitor
            progress_score = [0] * batch_size

            # Mask outputs of invalid actions
            _logit = logit.detach()
            _logit[is_valid == 0] = -float('inf')

            # Expand nodes
            ac_lens = np.argmax(is_valid.cpu() == 0, axis=1).detach()
            ac_lens[ac_lens == 0] = is_valid.shape[1]

            h_t_data, c_t_data = h_t.detach(),c_t.detach()
            u_t_data = all_u_t.detach()
            log_prob = F.log_softmax(_logit, dim=1).detach()

            # 4. prepare ending evaluation
            cand_instr= [_traj['instr_encoding'] for _traj in traj]
            cand_wss, cand_acs = self._search_collect(batch_queue, world_states, current_idx, ended)
            instr_ids = [_traj['instr_id'] for _traj in traj]
            cand_obs = self._wss_to_obs(cand_wss, instr_ids)
            speaker_scores = [0] * batch_size
            goal_scores = [0] * batch_size
            '''
            if self.goal_button is not None:
                # encode text
                instr_enc_h, instr_enc_c = final_text_enc(cand_instr, self.max_instruction_length, self.goal_button.text_encoder)
                # encode traj / the last image
                traj_enc = self.goal_button.encode_traj(cand_obs,cand_acs)
                goal_scores = self.goal_button.score(instr_enc, traj_enc)
            '''

            if self.gb:
                text_enc = [ob['instr_encoding'][-40:] for ob in obs]
                text_len = [len(_enc) for _enc in text_enc]
                text_tensor = np.zeros((batch_size, max(text_len)))
                for _idx, _enc in enumerate(text_enc):
                    text_tensor[_idx][:len(_enc)] = _enc
                text_tensor = torch.from_numpy(text_tensor).long().cuda()
                stop_button = self.gb(text_tensor, text_len, f_t_list[0])

            for idx in range(batch_size):
                if ended[idx]: continue

                _len = ac_lens[idx]
                new_logit = last_logit[idx].fork(_logit[idx][:_len].cpu().tolist())
                new_logp = last_logp[idx].fork(log_prob[idx][:_len].cpu().tolist())

                # entropy
                entropy = torch.sum(-log_prob[idx][:_len] * torch.exp(log_prob[idx][:_len]))

                # record
                if self.env.notTest:
                    _dev = obs[idx]['deviation']
                    self.records[_dev].append(
                        (ac_counts[idx],
                         obs[idx]['progress'],
                         _logit[idx][:_len].cpu().tolist(),
                         log_prob[idx][:_len].cpu().tolist(),
                         entropy.item(),
                         speaker_scores[idx]))                                        

                # selectively expand nodes
                K = self.K
                select_k = _len if _len < K else K
                top_ac = list(torch.topk(_logit[idx][:_len],select_k)[1])
                if self.inject_stop and 0 not in top_ac:
                    top_ac.append(0)

                # compute heuristics
                new_heur = new_logit if self.search_logit else new_logp

                if self.search_mean:
                    _new_heur = [_h.mean for _h in new_heur]
                else:
                    _new_heur = [_h.sum for _h in new_heur]

                visitedVps = [ws[1] for ws in cand_wss[idx]]
                for ac_idx, ac in enumerate(top_ac):
                    nextViewpointId = obs[idx]['adj_loc_list'][ac]['nextViewpointId']
                    if not self.revisit and (ac > 0 and nextViewpointId in visitedVps):
                        # avoid re-visiting
                        continue

                    if not self.beam and ac_idx == 0:
                        _new_heur[ac] = float('inf')

                    if ac == 0:
                        if not self.gb or (ac_idx == 0 and stop_button[idx][1] > stop_button[idx][0]):
                            # Don't stop unless the stop button says so
                            ending_heur = _new_heur[ac]
                            #ending_heur = stop_button[idx][1]
                            #ending_heur = stop_button[idx][1] / (stop_button[idx][0]+ stop_button[idx][1])
                            #CandidateState = namedtuple("CandidateState", "flogit,flogp,world_states,actions") # flat_index,
                            
                            new_ending = CandidateState(
                                    flogit=new_logit[ac],
                                    flogp=new_logp[ac],
                                    world_states=cand_wss[idx],
                                    actions=cand_acs[idx],
                                   # pm=progress_score[idx],
                                   # speaker=speaker_scores[idx],
                                   # scorer=_logit[idx][ac],
                                )
                            ending_queue[idx].push(new_ending, ending_heur)

                    if ac > 0 or self.search_early_stop:# or \
                            #(self.gb and stop_button[idx][1] > stop_button[idx][0]):

                        new_node = TransformerSearchState(
                            flogit=new_logit[ac],
                            flogp=new_logp[ac],
                            world_state=world_states[idx],
                            observation=obs[idx],
                            action=ac,
                            action_embedding=u_t_data[idx,ac],
                            action_count=ac_counts[idx]+1,
                            h_t=h_t_data[idx],c_t=c_t_data[idx],
                            father=current_idx[idx],
                            history_input=history_input[idx,:history_length[idx]+1],
                            history_length=history_length[idx].item()
                        )
                        batch_queue[idx].push(new_node, _new_heur[ac])

                if batch_queue[idx].size() == 0:
                    batch_queue[idx].lock()
                    ended[idx] = True

                #if ending_queue[idx].size() > 0:
                #    batch_queue[idx].lock()
                #    ended[idx] = True

        # For those not ended, forcefully goto the best ending point
        #print(np.size(ended) - np.count_nonzero(ended))

        # Force everyone to go to the best in ending queue
        # When ending queue has the same heursitic as expansion queue, this is
        # equivalent to go to the best ending point in expansion;

        # cache the candidates
        if hasattr(self, 'cache_candidates'):
            for idx in range(batch_size):
                instr_id = traj[idx]['instr_id']
                if instr_id not in self.cache_candidates:
                    cand = []
                    for item in ending_queue[idx].queue:
                        cand.append((instr_id, item.world_states, item.actions, item.flogit.sum, item.flogit.mean, item.flogp.sum, item.flogp.mean, item.pm, item.speaker, item.scorer))
                    self.cache_candidates[instr_id] = cand

        # cache the search progress
        if hasattr(self, 'cache_search'):
            for idx in range(batch_size):
                instr_id = traj[idx]['instr_id']
                if instr_id not in self.cache_search:
                    cand = []
                    for item in batch_queue[idx].queue:
                        cand.append((item.world_state, item.action, item.father, item.flogit.sum, item.flogp.sum))
                    self.cache_search[instr_id] = cand

        # actually move the cursor
        for idx in range(batch_size):
            instr_id = traj[idx]['instr_id']
            if ending_queue[idx].size() == 0:
                #print("Warning: some instr does not have ending, ",
                #        "this can be a desired behavior though")
                self.clean_results[instr_id] = {
                        'instr_id': traj[idx]['instr_id'],
                        'trajectory': traj[idx]['trajectory'],
                        }
                continue

            last_vp = traj[idx]['trajectory'][-1]
            if hasattr(self, 'reranker') and ending_queue[idx].size() > 1:
                inputs = []
                inputs_idx = []
                num_candidates = 100
                while num_candidates > 0 and ending_queue[idx].size() > 0:
                    _idx, _pri, item = ending_queue[idx].pop()
                    inputs_idx.append(_idx)
                    inputs.append([len(item.world_states), item.flogit.sum, item.flogit.mean, item.flogp.sum, item.flogp.mean, item.pm, item.speaker] * 4)
                    num_candidates -= 1
                inputs = try_cuda(torch.Tensor(inputs))
                reranker_scores = self.reranker(inputs)
                sel_cand = inputs_idx[torch.argmax(reranker_scores)]
                cur = ending_queue[idx].queue[sel_cand]
            else:
                cur = ending_queue[idx].peak()[-1]
            # keep switching if cur is not the shortest path?

            ob = self.env.observe([cur.world_states[-1]], instr_id=instr_id)
            traj[idx]['trajectory'] += realistic_jumping(
                visit_graphs[idx], last_vp, ob[0])
            ended[idx] = 1

            for _ws in cur.world_states: # we don't collect ws0, this is fine.
                clean_traj[idx]['trajectory'].append((_ws.viewpointId, _ws.heading, _ws.elevation))
                self.clean_results[instr_id] = clean_traj[idx]

        return traj,{}

    def get_loss_info(self):
        val_loss = np.average(self.losses)
        ce_loss = np.average(self.ce_losses)
        room_loss = np.average(self.room_losses)
        angle_reward = np.average(self.angle_rewards)
        loss_str = f'loss {val_loss:.3f}|ce {ce_loss:.3f} | rm_loss {room_loss:.3f} |angle_reward {angle_reward:.3f}\n'
        return loss_str, {'loss': val_loss,
                          'ce' : ce_loss,
                          'room':room_loss,
                          'angle_reward':angle_reward}

    def _test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        self.clean_results = {}
        self.text_attn_weight = []
        self.all_num = 0
        self.losses = []
        looped = False
        rollout_scores = []
        beam_10_scores = []

        with torch.no_grad():
            while True:
                rollout_results,atten_text_weight = self.rollout()
                for result in rollout_results:
                    if result['instr_id'] in self.results:
                        looped = True
                    else:
                        self.results[result['instr_id']] = result
                        self.text_attn_weight+=atten_text_weight
                if looped:
                    break
        return self.results,self.text_attn_weight

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, beam_size=1, distance_aware=True,distance_weight=1):
        ''' Evaluate once on each instruction in the current environment '''
        self.losses = []
        self.ce_losses = []
        self.room_losses = []
        self.angle_rewards = []
        self.dv_losses = []
        self.pm_losses = []
        self.bt_losses = []

        if not allow_cheat: 
            assert feedback in ['argmax', 'sample'] 
        self.feedback = feedback
        self.distance_aware = distance_aware
        self.distance_weight = distance_weight
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        return self._test()