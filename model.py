import torch.nn as nn
from preprocess import CustomDataset
import pandas as pd
import torch.nn.functional as F
import pickle
from torch.utils.data import DataLoader
from preprocess import SpeakerDS
import torch
import math
import functools
from manifolds import PoincareBall
from nets import MobiusGRU
import itertools
import geoopt.manifolds.stereographic.math as pmath_geo
from datetime import datetime


def one_rnn_transform(W, h, U, x, c):
    W_otimes_h = pmath_geo.mobius_matvec(W, h, k=c)
    U_otimes_x = pmath_geo.mobius_matvec(U, x, k=c)
    Wh_plus_Ux = pmath_geo.mobius_add(W_otimes_h, U_otimes_x, k=c)
    return Wh_plus_Ux


class AttentionHawkes(torch.nn.Module):

    def __init__(self, dimensions, bs, attention_type='general'):
        super(AttentionHawkes, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = torch.nn.Linear(
                dimensions, dimensions, bias=False)

        self.linear_out = torch.nn.Linear(
            dimensions * 2, dimensions, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()
        self.ae = torch.nn.Parameter(torch.FloatTensor(bs, 1, 1))
        self.ab = torch.nn.Parameter(torch.FloatTensor(bs, 1, 1))

    def forward(self, query, context, delta_t, c=1.0):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(
            query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(
            batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(
            batch_size, output_len, query_len)

        mix = attention_weights*(context.permute(0, 2, 1))
        bt = torch.exp(-1*self.ab * delta_t)
        term_2 = F.relu(self.ae * mix * bt)
        mix = torch.sum(term_2+mix, -1).unsqueeze(1)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)
        output = self.linear_out(combined).view(
            batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class AttentionHyp(torch.nn.Module):

    def __init__(self, dimensions, bs, attention_type='general'):
        super(AttentionHyp, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        self.c = torch.tensor([1.0]).to('cuda')
        if self.attention_type == 'general':
            self.linear_in = torch.nn.Linear(
                dimensions, dimensions, bias=False)

        self.linear_out = torch.nn.Linear(
            dimensions * 2, dimensions, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()
        self.ae = torch.nn.Parameter(torch.FloatTensor(bs, 1, 1))
        self.ab = torch.nn.Parameter(torch.FloatTensor(bs, 1, 1))

    def forward(self, query, context, delta_t, c):
        batch_size, output_len, dimensions = query.size()
        # print(query.size(), 'query')
        query_len = context.size(1)
        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(
            query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(
            batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(
            batch_size, output_len, query_len)

        # converting hyp
        attention_weights = pmath_geo.expmap0(attention_weights, k=c)
        attention_weights = pmath_geo.project(attention_weights, k=c)
        mix = pmath_geo.mobius_pointwise_mul(
            attention_weights, context.permute(0, 2, 1), k=c)
        mix = pmath_geo.project(mix, k=c)
        bt = torch.exp(-1*self.ab * delta_t)  # convert the e^-(lambda*time)
        bt = pmath_geo.expmap0(bt, k=c)
        bt = pmath_geo.project(bt, k=c)
        tmp_mul = pmath_geo.mobius_pointwise_mul(self.ae, mix, k=c)
        tmp_mul = pmath_geo.project(tmp_mul, k=c)
        tmp_mul = pmath_geo.mobius_pointwise_mul(tmp_mul, bt, k=c)
        tmp_mul = pmath_geo.project(tmp_mul, k=c)
        term_2 = F.relu(tmp_mul)
        mix = pmath_geo.mobius_add(mix, term_2, dim=-1, k=c)
        mix = pmath_geo.project(mix, k=c)
        bs = mix.shape[0]
        li = []
        for i in range(bs):
            tmp = pmath_geo.weighted_midpoint(
                mix[i],
                k=c,
                dim=0
            )
            li.append(tmp)
        fin_mat = torch.stack(li).unsqueeze(1).to('cuda')
        mix = pmath_geo.logmap0(fin_mat, k=c)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)
        output = self.linear_out(combined).view(
            batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class Attention(nn.Module):

    def __init__(self, dimensions, attention_type='general'):
        super().__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):

        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(
            query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(
            batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(
            batch_size, output_len, query_len)

        mix = torch.bmm(attention_weights, context)

        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(
            batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, reverse=False):
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)
        h = h.cuda()
        c = c.cuda()
        outputs = []
        hidden_state_h = []
        hidden_state_c = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))  # short term mem
            # lookback vectors number of speakers (1,5,5,1)
            # [1,2,3,]
            # discounted short term mem
            c_s2 = c_s1 * timestamps[:, s: s + 1].expand_as(c_s1)
            c_l = c - c_s1  # long term mem
            c_adj = c_l + c_s2  # adjusted = long + disc short term mem
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(o)
            hidden_state_c.append(c)
            hidden_state_h.append(h)
        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()
        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)
        return outputs, (h, c)


class TimeLSTMHyp(nn.Module):
    def __init__(self, input_size, hidden_size, device='cuda', cuda_flag=False, bidirectional=False):
        super(TimeLSTMHyp, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = torch.nn.Parameter(
            torch.Tensor(hidden_size, hidden_size * 4))
        self.U_all = torch.nn.Parameter(
            torch.Tensor(hidden_size * 4, input_size))
        self.W_d = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bidirectional = bidirectional
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.W_all, self.U_all, self.W_d]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, timestamps, hidden_states, c, reverse=False):
        b, seq, embed = inputs.size()
        h = hidden_states[0]
        _c = hidden_states[1]
        if self.cuda_flag:
            h = h.cuda()
            _c = _c.cuda()
        outputs = []
        hidden_state_h = []
        hidden_state_c = []

        for s in range(seq):
            c_s1 = pmath_geo.expmap0(torch.tanh(pmath_geo.logmap0(
                pmath_geo.mobius_matvec(self.W_d, _c, k=c), k=c)), k=c)  # short term mem
            c_s2 = pmath_geo.mobius_pointwise_mul(
                c_s1, timestamps[:, s: s + 1].expand_as(c_s1), k=c)  # discounted short term mem
            c_l = pmath_geo.mobius_add(-c_s1, _c, k=c)  # long term mem
            c_adj = pmath_geo.mobius_add(c_l, c_s2, k=c)

            W_f, W_i, W_o, W_c_tmp = self.W_all.chunk(4, dim=1)
            U_f, U_i, U_o, U_c_tmp = self.U_all.chunk(4, dim=0)

            f = pmath_geo.logmap0(one_rnn_transform(
                W_f, h, U_f, inputs[:, s], c), k=c).sigmoid()
            i = pmath_geo.logmap0(one_rnn_transform(
                W_i, h, U_i, inputs[:, s], c), k=c).sigmoid()
            o = pmath_geo.logmap0(one_rnn_transform(
                W_o, h, U_o, inputs[:, s], c), k=c).sigmoid()
            c_tmp = pmath_geo.logmap0(one_rnn_transform(
                W_c_tmp, h, U_c_tmp, inputs[:, s], c), k=c).sigmoid()

            f_dot_c_adj = pmath_geo.mobius_pointwise_mul(f, c_adj, k=c)
            i_dot_c_tmp = pmath_geo.mobius_pointwise_mul(i, c_tmp, k=c)
            _c = pmath_geo.mobius_add(i_dot_c_tmp, f_dot_c_adj, k=c)

            h = pmath_geo.mobius_pointwise_mul(
                o, pmath_geo.expmap0(torch.tanh(_c), k=c), k=c)
            outputs.append(o)
            hidden_state_c.append(_c)
            hidden_state_h.append(h)

        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()
        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)

        return outputs, (h, _c)


class HYPHEN(nn.Module):
    def __init__(self, input_size, hidden_size, bs, attn_type='vanilla', learnable_curvature=False, init_curvature_val=0., n_class = 2):
        if attn_type not in ['vanilla', 'hawkes', 'hyp_hawkes']:
            raise ValueError(" Attn not of correct type")
        super().__init__()
        self.hyp_lstm = TimeLSTMHyp(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_class)
        self.dropout = nn.Dropout(0.3)
        if learnable_curvature:
            # print("Init")
            self.c = torch.nn.Parameter(
                torch.tensor([init_curvature_val]).to('cuda'))
            # self.c =
        else:
            self.c = torch.FloatTensor([1.0]).to('cuda')
        self.attn_type = attn_type
        self.hidden_size = hidden_size
        if attn_type == 'hawkes':
            self.attention = AttentionHawkes(
                hidden_size, bs)  # Hawkes and temporal attn
        elif attn_type == 'hyp_hawkes':
            self.attention = AttentionHyp(
                hidden_size, bs)  # Hawkes and temporal attn
        else:
            self.attention = Attention(hidden_size)
        self.cell_source = MobiusGRU(
            hidden_size, hidden_size, 1, k=self.c).to('cuda')

    def init_hidden(self, bs):
        h = (torch.zeros(bs, self.hidden_size, requires_grad=True)).to(
            'cuda')
        c = (torch.zeros(bs, self.hidden_size, requires_grad=True)).to(
            'cuda')

        return (h, c)

    def forward(self, inputs, timestamps, timestamps_inv):
        bs = inputs.shape[0]
        h_init, c_init = self.init_hidden(bs)
        inputs = pmath_geo.expmap0(inputs, k=self.c)  # Exp mapping
        output, (_, _) = self.hyp_lstm(
            inputs, timestamps_inv, (h_init, c_init), self.c)
        context, output = self.cell_source(output.permute(1, 0, 2))
        output = output.permute(1, 0, 2)
        context = context.permute(1, 0, 2)
        output = pmath_geo.logmap0(output, k=self.c)
        context = pmath_geo.logmap0(context, k=self.c)
        if self.attn_type == 'vanilla':
            output_fin, _ = self.attention(output, context)
        else:
            output_fin, _ = self.attention(output, context, timestamps, self.c)
        output_fin = output_fin.permute(1, 0, 2)
        output_fin = output_fin.squeeze(0)
        output_fin = self.linear1(output_fin)
        output_fin = F.relu(output_fin)
        output_fin = self.dropout(output_fin)
        output_fin = self.linear2(output_fin)
        return output_fin


if __name__ == "__main__":

    with open('speaker_train_2.pkl', 'rb') as f:
        speakers_train = pickle.load(f)
    print("Train data loaded")
    ds_train = SpeakerDS(speakers_train)
    train_data_loader = DataLoader(
        dataset=ds_train, batch_size=128, shuffle=True)
    dl = next(iter(train_data_loader))
    inp_tensor = dl['speech_data'].squeeze(1)
    print(inp_tensor.shape)
    dates = dl['dates'].squeeze(1)
    dates_inv = dl['dates_inv'].squeeze(1)
    print(dl['dates'].squeeze(1).shape, 'date')

    bs = inp_tensor.shape[0]
    model2 = HYPHEN(768, 128, bs, learnable_curvature=False).to('cuda')
    gg = model2(inp_tensor, dates, dates_inv)
