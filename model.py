import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from utils import try_cuda
from attn import MultiHeadAttention

from env_simplified import ConvolutionalImageFeatures, BottomUpImageFeatures
from KB.ConceptNet import GCN_pre_define

def make_image_attention_layers(args, image_features_list, hidden_size):
    image_attention_size = args.image_attention_size or hidden_size
    attention_mechs = []
    for featurizer in image_features_list:
        if isinstance(featurizer, ConvolutionalImageFeatures):
            if args.image_attention_type == 'feedforward':
                attention_mechs.append(MultiplicativeImageAttention(
                    hidden_size, image_attention_size,
                    image_feature_size=featurizer.feature_dim))
            elif args.image_attention_type == 'multiplicative':
                attention_mechs.append(FeedforwardImageAttention(
                    hidden_size, image_attention_size,
                    image_feature_size=featurizer.feature_dim))
        elif isinstance(featurizer, BottomUpImageFeatures):
            attention_mechs.append(BottomUpImageAttention(
                hidden_size,
                args.bottom_up_detection_embedding_size,
                args.bottom_up_detection_embedding_size,
                image_attention_size,
                featurizer.num_objects,
                featurizer.num_attributes,
                featurizer.feature_dim
            ))
        else:
            attention_mechs.append(None)
    attention_mechs = [
        try_cuda(mech) if mech else mech for mech in attention_mechs]
    return attention_mechs

# TODO: make all attention module return logit instead of weight

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the PE once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() / d_model * \
                            (-math.log(10000.0)))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1)], requires_grad=False)
        return self.dropout(x)


# TODO: try variational dropout (or zoneout?)
class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=1, glove=None):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=(dropout_ratio if self.num_layers > 1 else 0),
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
                                         hidden_size * self.num_directions)

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def forward(self, inputs, lengths, seq_mask):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        if not self.use_glove:
            embeds = self.drop(embeds)
        h0, c0 = self.init_state(batch_size)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        # (batch, seq_len, hidden_size*num_directions), (batch, hidden_size)
        return ctx, decoder_init, c_t

class SoftDotMultiHead(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim, num_head):
        '''Initialize layer.'''
        super(SoftDotMultiHead, self).__init__()
        self.multi = MultiHeadAttention(num_head, dim, dim, dim)

    def forward(self, h, k, v, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        k,v: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        output, attn = self.multi(h.unsqueeze(1), k, v, mask.unsqueeze(1))
        return output.squeeze(1), attn.squeeze(1)

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.dim = dim
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        # TODO: attn = attn / math.sqrt(self.dim) # prevent extreme softmax

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class ContextOnlySoftDotAttention(nn.Module):


    def __init__(self, dim, context_dim=None):
        '''Initialize layer.'''
        super(ContextOnlySoftDotAttention, self).__init__()
        if context_dim is None:
            context_dim = dim
        self.linear_in = nn.Linear(dim, context_dim, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weighted_context, attn


class FeedforwardImageAttention(nn.Module):
    def __init__(self, context_size, hidden_size, image_feature_size=2048):
        super(FeedforwardImageAttention, self).__init__()
        self.feature_size = image_feature_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.fc1_feature = nn.Conv2d(
            image_feature_size, hidden_size, kernel_size=1, bias=False)
        self.fc1_context = nn.Linear(context_size, hidden_size, bias=True)
        self.fc2 = nn.Conv2d(hidden_size, 1, kernel_size=1, bias=True)

    def forward(self, feature, context):
        batch_size = feature.shape[0]
        feature_hidden = self.fc1_feature(feature)
        context_hidden = self.fc1_context(context)
        context_hidden = context_hidden.unsqueeze(-1).unsqueeze(-1)
        x = feature_hidden + context_hidden
        x = self.fc2(F.relu(x))
        # batch_size x (width * height) x 1
        attention = F.softmax(x.view(batch_size, -1), 1).unsqueeze(-1)
        # batch_size x feature_size x (width * height)
        reshaped_features = feature.view(batch_size, self.feature_size, -1)
        x = torch.bmm(reshaped_features, attention)  # batch_size x
        return x.squeeze(-1), attention.squeeze(-1)


class MultiplicativeImageAttention(nn.Module):
    def __init__(self, context_size, hidden_size, image_feature_size=2048):
        super(MultiplicativeImageAttention, self).__init__()
        self.feature_size = image_feature_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.fc1_feature = nn.Conv2d(
            image_feature_size, hidden_size, kernel_size=1, bias=True)
        self.fc1_context = nn.Linear(context_size, hidden_size, bias=True)
        self.fc2 = nn.Conv2d(hidden_size, 1, kernel_size=1, bias=True)

    def forward(self, feature, context):
        batch_size = feature.shape[0]
        # batch_size x hidden_size x width x height
        feature_hidden = self.fc1_feature(feature)
        # batch_size x hidden_size
        context_hidden = self.fc1_context(context)
        # batch_size x 1 x hidden_size
        context_hidden = context_hidden.unsqueeze(-2)
        # batch_size x hidden_size x (width * height)
        feature_hidden = feature_hidden.view(batch_size, self.hidden_size, -1)
        # batch_size x 1 x (width x height)
        x = torch.bmm(context_hidden, feature_hidden)
        # batch_size x (width * height) x 1
        attention = F.softmax(x.view(batch_size, -1), 1).unsqueeze(-1)
        # batch_size x feature_size x (width * height)
        reshaped_features = feature.view(batch_size, self.feature_size, -1)
        x = torch.bmm(reshaped_features, attention)  # batch_size x
        return x.squeeze(-1), attention.squeeze(-1)


class BottomUpImageAttention(nn.Module):
    def __init__(self, context_size, object_embedding_size,
                 attribute_embedding_size, hidden_size, num_objects,
                 num_attributes, image_feature_size=2048):
        super(BottomUpImageAttention, self).__init__()
        self.context_size = context_size
        self.object_embedding_size = object_embedding_size
        self.attribute_embedding_size = attribute_embedding_size
        self.hidden_size = hidden_size
        self.num_objects = num_objects
        self.num_attributes = num_attributes
        self.feature_size = (image_feature_size + object_embedding_size +
                             attribute_embedding_size + 1 + 5)

        self.object_embedding = nn.Embedding(
            num_objects, object_embedding_size)
        self.attribute_embedding = nn.Embedding(
            num_attributes, attribute_embedding_size)

        self.fc1_context = nn.Linear(context_size, hidden_size)
        self.fc1_feature = nn.Linear(self.feature_size, hidden_size)
        # self.fc1 = nn.Linear(context_size + self.feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, bottom_up_features, context):
        # image_features: batch_size x max_num_detections x feature_size
        # object_ids: batch_size x max_num_detections
        # attribute_ids: batch_size x max_num_detections
        # no_object_mask: batch_size x max_num_detections
        # context: batch_size x context_size

        # batch_size x max_num_detections x embedding_size
        attribute_embedding = self.attribute_embedding(
            bottom_up_features.attribute_indices)
        # batch_size x max_num_detections x embedding_size
        object_embedding = self.object_embedding(
            bottom_up_features.object_indices)
        # batch_size x max_num_detections x (feat size)
        feats = torch.cat((
            bottom_up_features.cls_prob.unsqueeze(2),
            bottom_up_features.image_features,
            attribute_embedding, object_embedding,
            bottom_up_features.spatial_features), dim=2)

        # attended_feats = feats.mean(dim=1)
        # attention = None

        # batch_size x 1 x hidden_size
        x_context = self.fc1_context(context).unsqueeze(1)
        # batch_size x max_num_detections x hidden_size
        x_feature = self.fc1_feature(feats)
        # batch_size x max_num_detections x hidden_size
        x = x_context * x_feature
        x = x / torch.norm(x, p=2, dim=2, keepdim=True)
        x = self.fc2(x).squeeze(-1)  # batch_size x max_num_detections
        x.data.masked_fill_(bottom_up_features.no_object_mask, -float("inf"))
        # batch_size x 1 x max_num_detections
        attention = F.softmax(x, 1).unsqueeze(1)
        # batch_size x feat_size
        attended_feats = torch.bmm(attention, feats).squeeze(1)
        return attended_feats, attention

class WhSoftDotAttentionCompact(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, dim, context_dim):
        '''Initialize layer.'''
        super(WhSoftDotAttentionCompact, self).__init__()
        if dim != context_dim:
            dot_dim = min(dim, context_dim)
            self.linear_in = nn.Linear(dim, dot_dim//2, bias=True)
            self.linear_in_2 = nn.Linear(context_dim, dot_dim//2, bias=True)
        self.dim = dim
        self.context_dim = context_dim
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, ctx, mask=None, v=None):
        if self.dim != self.context_dim:
            target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
            context = self.linear_in_2(ctx)
        else:
            target = h.unsqueeze(2)
            context = ctx
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn_sm = self.sm(attn)
        attn3 = attn_sm.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len
        context = v if v is not None else ctx
        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weighted_context, attn

class VisualSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim, dot_dim=256):
        '''Initialize layer.'''
        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, k, mask=None, v=None):
        '''Propagate h through the network.

        h: batch x h_dim
        k: batch x v_num x v_dim
        '''
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        context = self.linear_in_v(k)  # batch x v_num x dot_dim
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
        attn_sm = self.sm(attn)
        attn3 = attn_sm.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num
        ctx = v if v is not None else k
        weighted_context = torch.bmm(
            attn3, ctx).squeeze(1)  # batch x v_dim
        return weighted_context, attn

class WhSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim=None):
        '''Initialize layer.'''
        super(WhSoftDotAttention, self).__init__()
        if v_dim is None:
            v_dim = h_dim
        self.h_dim = h_dim
        self.v_dim = v_dim
        self.linear_in_h = nn.Linear(h_dim, v_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, k, mask=None, v=None):
        '''Propagate h through the network.
        h: batch x h_dim
        k: batch x v_num x v_dim
        '''
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        attn = torch.bmm(k, target).squeeze(2)  # batch x v_num
        #attn /= math.sqrt(self.v_dim) # scaled dot product attention
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn_sm = self.sm(attn)
        attn3 = attn_sm.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num
        ctx = v if v is not None else k
        weighted_context = torch.bmm(
            attn3, ctx).squeeze(1)  # batch x v_dim
        return weighted_context, attn






###############################################################################
# transformer models
###############################################################################

class EmbeddingEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=1, glove=None):
        super(EmbeddingEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.position_encoding = PositionalEncoding(hidden_size, dropout_ratio)
        self.use_glove = glove is not None
        self.fc = nn.Linear(embedding_size, hidden_size)
        nn.init.xavier_normal_(self.fc.weight)
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = torch.zeros(batch_size,
                         self.hidden_size*self.num_layers*self.num_directions,
                         requires_grad=False)
        c0 = torch.zeros(batch_size,
                         self.hidden_size*self.num_layers*self.num_directions,
                         requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def forward(self, inputs, lengths,seq_mask=None,nomap=False):
        batch_size = inputs.size(0)
        embeds = self.embedding(inputs)
        if nomap:
            return (embeds, *self.init_state(batch_size))
        embeds = self.fc(embeds)
        embeds = self.position_encoding(embeds)
        max_len = max(lengths)
        embeds = embeds[:,:max_len,:]
        return (embeds, *self.init_state(batch_size))

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=2,nhead=6, glove=None,ff=2048):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_size,nhead=nhead,dropout=dropout_ratio,dim_feedforward=ff)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.position_embedding = PositionalEncoding(embedding_size,dropout=0)
        self.fc = nn.Linear(embedding_size,hidden_size)

    def forward(self,inputs,lengths,seq_mask):
        batch_size = inputs.size(0)
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        if not self.use_glove:
            embeds = self.drop(embeds)
        seq_mask = seq_mask.bool()
        embeds = self.position_embedding(embeds)
        max_len = max(lengths)
        embeds = embeds[:,:max_len,:]
        embeds = embeds.transpose(0,1)
        output =  self.transformer_encoder(embeds,src_key_padding_mask=seq_mask)
        output = output.transpose(0,1)
        output = self.fc(output)
        lengths = try_cuda(torch.tensor(lengths))
        h_0 = output[torch.arange(batch_size),lengths-1,:]
        return(output,h_0,h_0)

def hard_softmax(y,dim=1):
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class object_roomTransformer(nn.Module):
    #fake code:
    #1.language_input->attn_with h_0 get text_object_attn(1*300),text_room_attn(1*512)
    #2.text_object_attn(1*300) -> attn_weight with knowledge base get top K(5*300)
    #   ->gather with top K 1*512()->  refine_object_attn
    #3.text_room_attn -> text_room_label ->room_label_text
    #    ->text_room_loss
    #4.label_set -> viewpoint_label -> GCN_with_predefine edges
    #   ->label_feature_list->atten with refine_obejct_attn
    #5.label_set ->object_top_5 ->room_label-> view_room_loss
    #find label for room

    def __init__(self, embedding_size, hidden_size, dropout_ratio,
                feature_size=2048+128, image_attention_layers=None,
                visual_hidden_size=1024,num_head=8,num_layer=6,concate_room=False,
                wo_instr_input=False,transformer_dropout_rate=0.1,action_prediction_mode='single',label_size=300,
                use_room=True,use_object=True,num_gcn=3,gcn_type='in',
                max_degree=10,short_cut=False,ff=2048,label_length = 5,soft_room_label=True,room_relation_vec=True,object_top_n=5,load_room_relation_weight=True,load_object_relation_weight=True):
        super(object_roomTransformer,self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.action_prediction_mode = action_prediction_mode
        self.wo_instr_input = wo_instr_input
        self.hidden_size = hidden_size
        self.search = False
        self.topk = object_top_n
        self.u_begin = try_cuda(Variable(
            torch.zeros(embedding_size), requires_grad=False))
        self.history_begin = try_cuda(Variable(
            torch.zeros(1,hidden_size), requires_grad=False)
        )
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=0)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.use_room = use_room
        self.use_object = use_object
        self.soft_room_label = soft_room_label
        num_rooms = 31
        repeat = 128
        decoder_layer =nn.TransformerDecoderLayer(hidden_size,num_head,dropout=transformer_dropout_rate,dim_feedforward=ff)
        self.decoder = nn.TransformerDecoder(decoder_layer,num_layer)
        self.object_text_attention_layer = WhSoftDotAttention(hidden_size, hidden_size)
        self.room_text_attention_layer = WhSoftDotAttention(hidden_size,hidden_size)
        self.object_map = nn.Linear(hidden_size,300)
        self.text_room_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_size,31),
            nn.Softmax(dim=1)
        )
        for p in self.text_room_classifier.parameters():
            p.requires_grad=False
        self.view_room_classifier = nn.Sequential(
            nn.Linear(label_length*300, hidden_size),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_size,31),
            nn.Softmax(dim=1)
        )
        for p in self.view_room_classifier.parameters():
            p.requires_grad=False
        self.sm = nn.Softmax(dim=1)
        self.object_attention_layer = WhSoftDotAttention(300,300)
        self.object_refine_layer = WhSoftDotAttention(300,300)
        self.room_attention_layer = WhSoftDotAttention(hidden_size,300)
        self.room_relation_vec = room_relation_vec
        if room_relation_vec:
            self.room_class_relation = torch.nn.Parameter(torch.randn(num_rooms,num_rooms,repeat),requires_grad=True)
        else:
            if load_room_relation_weight:
                x = np.load('KB/data/relations_room.npy')
                self.room_class_relation = torch.nn.Parameter(torch.relu(torch.from_numpy(x).float()),requires_grad=True)
            else:
                self.room_class_relation = torch.nn.Parameter(torch.rand(num_rooms,num_rooms), requires_grad=True)
        input_dim = feature_size + embedding_size
        if not wo_instr_input:
            input_dim = input_dim + hidden_size
        if use_object:
            input_dim = input_dim+300
        self.input_mapping = nn.Linear(input_dim,hidden_size)
        self.object_gcn = GCN_pre_define(300,process_num=num_gcn,max_degree=max_degree,short_cut=short_cut,load_adj_weight=load_object_relation_weight)
        self.concate_room = concate_room
        if use_room:
            if concate_room:
                self.action_selector = WhSoftDotAttention(hidden_size+hidden_size,visual_hidden_size+62)
            else:
                self.action_selector = WhSoftDotAttention(hidden_size+hidden_size,visual_hidden_size+repeat)
        else:
            self.action_selector = WhSoftDotAttention(hidden_size+hidden_size,visual_hidden_size)
        self.visual_mlp = nn.Sequential(
                nn.BatchNorm1d(feature_size),
                nn.Linear(feature_size, visual_hidden_size),
                nn.BatchNorm1d(visual_hidden_size),
                nn.Dropout(dropout_ratio),
                nn.ReLU())
        self.visual_attention_layer = WhSoftDotAttention(hidden_size, visual_hidden_size)

    def activate_classifier(self):
        for p in self.view_room_classifier.parameters():
            p.requires_grad=True
        for p in self.text_room_classifier.parameters():
            p.requires_grad=True

    def forward(self,*args):
        if self.search:
            return self._forward_search(*args)
        else:
            return self._forward_train(*args)

    def encode_label(self,labels):
        embeddings = self.object_gcn.kg.embeddings
        zero_tensor = try_cuda(torch.zeros(1,300))
        embeddings = torch.cat([zero_tensor,embeddings],axis=0)
        batch_size,label_num = labels.shape
        if label_num < 5:
            print('not enough label')
            return try_cuda(torch.zeros(batch_size,5,300))
        else:
            labels = labels[:,:5]
            origin_shape = labels.shape
            labels = labels.reshape(-1)
            embeding_features = embeddings[labels]
            embeding_features = embeding_features.reshape((*origin_shape,-1))
        return embeding_features

    def get_label_sim(self,text_label_class,view_label_class,repeat=128):
        if self.concate_room:
            return torch.cat([text_label_class,view_label_class],axis=1)
        if not self.room_relation_vec:
            mid = torch.matmul(self.room_class_relation,text_label_class[:,:,None])
            similarity = torch.bmm(view_label_class[:,None,:],mid)
            similarity = similarity.squeeze()[:,None]
            return similarity.expand(-1,128)
        else:
            mid =torch.matmul(text_label_class,self.room_class_relation)
            return torch.bmm(view_label_class[:,None,:],mid.transpose(0,1)).squeeze()
    def find_nearby(self,text_object_query):
        if self.topk == 0:
            return text_object_query
        batch_size = text_object_query.shape[0]
        embeddings = self.object_gcn.kg.embeddings
        similarity = torch.mm(text_object_query,embeddings.transpose(0,1))
        values,index = torch.topk(similarity,self.topk,dim=1)
        values= torch.softmax(values,dim=1)
        find_object = embeddings[index.reshape(-1)].reshape(batch_size,5,-1)
        attn_object = find_object * values[:,:,None].expand(-1,-1,300)
        attn_object = attn_object.sum(axis=1)
        return attn_object,index

    def is_search(self):
        self.search = True

    def not_search(self):
        self.search = False

    def _build_current_input(self,u_t_prev, all_u_t, visual_context,
                h_0, ctx,object_label_set=None,view_label_set=None):
        batch_size,action_count,_ = all_u_t.shape
        ctx_pos = self.positional_encoding(ctx)#b,seq,512
        text_obj_atten,alpha_atten_obj = self.object_text_attention_layer(h_0,ctx_pos,v=ctx)#(2)
        text_room_atten,alpha_atten_room = self.room_text_attention_layer(h_0,ctx_pos,v=ctx)#(1)
        text_atten = (text_obj_atten+text_room_atten)/2
        text_alpha = (alpha_atten_obj+alpha_atten_room)/2
        #xxx model
        text_obj_query = self.object_map(text_obj_atten)
        text_refined_object,object_index = self.find_nearby(text_obj_query)
        text_room_class = self.text_room_classifier(text_room_atten)
        text_hard_room_label = hard_softmax(text_room_class)
        #yyy model
        #common_sense part
        object_label_features = self.object_gcn(object_label_set)
        object_features,_ = self.object_attention_layer(text_refined_object,object_label_features)
        encoded_room_feature = self.encode_label(view_label_set)
        encoded_room_feature = encoded_room_feature.reshape(batch_size*action_count,-1)
        view_room_class = self.view_room_classifier(encoded_room_feature)#room type classification
        view_hard_room_label = hard_softmax(view_room_class)
        text_hard_room_label_ = text_hard_room_label[:,None,:].expand(-1,action_count,-1).\
                                reshape(action_count*batch_size,-1)
        if self.soft_room_label:
            sim = self.get_label_sim(view_room_class,text_room_class[:,None,:].expand(-1,action_count,-1).\
                                reshape(action_count*batch_size,-1))
        else:
            sim = self.get_label_sim(text_hard_room_label_,view_hard_room_label)
            sim = sim.reshape(batch_size,action_count,-1)
        #ragular part
        g_v = all_u_t.view(-1, self.feature_size)#b*ac, 2048+128z
        g_v = self.visual_mlp(g_v).view(batch_size, action_count, -1)#b,ac,1024
        attn_vision, _alpha_vision = self.visual_attention_layer(h_0, g_v, v=all_u_t)#b,2048+128 #b,seq,1
        alpha_vision = self.sm(_alpha_vision)
        concat_input = torch.cat([attn_vision,u_t_prev],axis=1)
        if not self.wo_instr_input:
            concat_input = torch.cat([text_atten,concat_input],axis=1)#b,2048+128+2048+128
        if self.use_object:
            concat_input =  torch.cat([object_features,concat_input],axis=1)
        _input_curent = self.input_mapping(concat_input)
        return _input_curent,ctx_pos , text_atten, sim.reshape(batch_size,action_count,-1), \
                g_v, attn_vision, text_alpha,alpha_vision,text_room_class,view_room_class, \
                alpha_atten_obj, alpha_atten_room,object_index

    def _forward_train(self,u_t_prev, all_u_t, visual_context, input_history, h_0, ctx,
                ctx_mask,object_label_set=None,view_label_set=None):
        import ipdb;ipdb.set_trace()
        _input_curent,ctx_pos,text_atten,\
        room_features,g_v,attn_vision,text_alpha,\
        alpha_vision,text_room_class ,view_room_loss, alpha_atten_obj, alpha_atten_room, object_index = self._build_current_input(u_t_prev, all_u_t,
                                visual_context, h_0, ctx,
                                object_label_set,view_label_set)
        input_curent = _input_curent[:,None,:]#b,512
        ctx_mask = ctx_mask.bool()
        if input_history is not None:
            to_input = torch.cat([input_history,input_curent],axis=1)#b,seq+1,512
        else:
            to_input = input_curent
        _to_input = self.positional_encoding(to_input)
        ctx_input = ctx_pos.transpose(0,1)
        _to_input = _to_input.transpose(0,1)
        outputs = self.decoder(_to_input,ctx_input, memory_key_padding_mask=ctx_mask.bool())
        outputs = outputs.transpose(0,1)
        h_1 = outputs[:,-1,:]
        c_1 = outputs[:,:-1,:]
        c_1 = c_1.mean(axis=1)
        #  action_selector = torch.cat((attn_text, h_1),axis=1)#512,512
        #  _,alpha_action = self.action_attention_layer(action_selector,g_v)
        if self.use_room:
            g_v = torch.cat([g_v,room_features],axis=2)
        alpha_action = self.action_prediction(text_atten,h_1,all_u_t,g_v)
        return h_1,c_1,to_input,text_atten,attn_vision,text_alpha,alpha_action,alpha_vision,text_room_class, \
        view_room_loss,self.sm(alpha_atten_obj),self.sm(alpha_atten_room),object_index

    def _forward_search(self,u_t_prev, all_u_t, visual_context, input_history, history_length,h_0, ctx,
                ctx_mask=None,object_label_set=None,view_label_set=None,gt_labels=None):
        input_curent,ctx_pos,text_atten,room_features,\
        g_v,attn_vision,text_alpha,alpha_vision,text_room_class ,view_room_loss,\
        alpha_atten_obj,alpha_atten_room,object_index = self._build_current_input(u_t_prev, all_u_t,
                        visual_context, h_0, ctx,
                        object_label_set,view_label_set)
        batch_size,action_count,_ = all_u_t.shape
        batch_size = all_u_t.shape[0]
        _,seq_len,_ = input_history.size()
        to_input = input_history
        to_input[np.arange(batch_size),history_length,:] = input_curent
        tgt_mask = to_input != 0
        tgt_mask = tgt_mask.sum(axis=2) != 0
        c_mask = tgt_mask.clone()
        tgt_mask = ~tgt_mask.bool()
        _to_input = self.positional_encoding(to_input)
        _to_input = _to_input.transpose(0,1)
        ctx_input = ctx_pos.transpose(0,1)
        ctx_mask = ctx_mask.bool()
        outputs = self.decoder(_to_input,ctx_input,tgt_key_padding_mask=tgt_mask,memory_key_padding_mask=ctx_mask)
        outputs = outputs.transpose(0,1)
        h_1 = outputs[np.arange(batch_size),history_length,:]
        c_mask[np.arange(batch_size),history_length] = 0
        c_1 = outputs*c_mask[:,:,None].expand(batch_size,seq_len,512).float()
        history_length = torch.tensor(history_length)
        c_1 = c_1.sum(axis=1)/try_cuda(history_length[:,None].expand(batch_size,self.hidden_size)).clamp_min(1).float()
        #action_selector = torch.cat((attn_text, h_1),axis=1)#512,512
        #_,alpha_action = self.action_attention_layer(action_selector,g_v)
        if self.use_room:
            if gt_labels is not None:
                text_labels = gt_labels[0]
                view_labels = gt_labels[1]
                text_labels_oh = try_cuda(torch.zeros(text_room_class.shape))
                view_labels_oh = try_cuda(torch.zeros(view_room_loss.shape))
                text_labels_oh.scatter_(1, text_labels[:,None], 1)
                view_labels_oh.scatter(1,view_labels[:,None],1)
                sim = self.get_label_sim(view_labels_oh,text_labels_oh[:,None,:].expand(-1,action_count,-1).\
                        reshape(action_count*batch_size,-1))
                sim = sim.reshape(batch_size,action_count,-1)
                room_features=sim
            g_v = torch.cat([g_v,room_features],axis=2)
        alpha_action = self.action_prediction(text_atten,h_1,all_u_t,g_v)
        new_length = history_length+1
        to_input = torch.cat([to_input,try_cuda(torch.zeros(batch_size ,1, self.hidden_size))],dim=1)
        return h_1,c_1,to_input,new_length, text_atten,attn_vision,text_alpha,alpha_action,alpha_vision

    def action_prediction(self,text_atten,h_1,all_u_t,g_v):
        selector_hi = torch.cat([text_atten,h_1],axis=1)
        _,alpha_action = self.action_selector(selector_hi,g_v)
        return alpha_action


###############################################################################
# scorer models
###############################################################################

