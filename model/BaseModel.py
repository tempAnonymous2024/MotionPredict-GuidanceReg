#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math

class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_c, out_c, node_n = 22, seq_len = 35, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_c
        self.out_features = out_c
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        self.weight_seq = Parameter(torch.FloatTensor(seq_len, seq_len))

        self.weight_c = Parameter(torch.FloatTensor(in_c, out_c))

        if bias:
            self.bias = Parameter(torch.FloatTensor(seq_len))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.support = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(1))
        self.weight_c.data.uniform_(-stdv, stdv)
        self.weight_seq.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # input [b,c,22,35]

        # 先进行图卷积再进行空域卷积
        # [b,c,22,35（dct系数）] -> [b,35,22,c] -> [b,35,22,c]
        support = torch.matmul(self.att, input.permute(0, 3, 2, 1))

        # [b,35,22,c] -> [b,35,22,64]
        output_gcn  = torch.matmul(support, self.weight_c)

        # 进行空域卷积
        # [b,35,22,64] -> [b,22,64,35]
        output_fc = torch.matmul(output_gcn.permute(0, 2, 3, 1), self.weight_seq).permute(0, 2, 1, 3).contiguous()


        if self.bias is not None:
            return (output_fc + self.bias)
        else:
            return output_fc

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, channal, p_dropout, bias=True, node_n=22, seq_len = 20):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = channal
        self.out_features = channal

        self.gc1 = GraphConvolution(channal, channal, node_n=node_n, seq_len=seq_len, bias=bias)
        self.bn1 = nn.BatchNorm1d(channal*node_n*seq_len)

        self.gc2 = GraphConvolution(channal, channal, node_n=node_n, seq_len=seq_len, bias=bias)
        self.bn2 = nn.BatchNorm1d(channal*node_n*seq_len)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):

        y = self.gc1(x)
        b, c, n, l = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n, l).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, c, n, l = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn2(y).view(b, c, n, l).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, in_channal, out_channal, node_n=22, seq_len=20, p_dropout=0.3, num_stage=1 ):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(in_c=in_channal, out_c=out_channal, node_n=node_n, seq_len=seq_len)
        self.bn1 = nn.BatchNorm1d(out_channal*node_n*seq_len)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(channal=out_channal, p_dropout=p_dropout, node_n=node_n, seq_len=seq_len))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(in_c=out_channal, out_c=in_channal, node_n=node_n, seq_len=seq_len)
        self.bn2 = nn.BatchNorm1d(in_channal*node_n*seq_len)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()


    def forward(self, x):

        y = self.gc1(x)
        b, c, n, l = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n, l).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        # b, n, f = y.shape
        # y = self.bn2(y.view(b, -1)).view(b, n, f)
        # y = self.act_f(y)
        # y = self.do(y)

        return y + x

class GCN_encoder(nn.Module):
    def __init__(self, in_channal, out_channal, node_n=22, seq_len=20, p_dropout=0.3, num_stage=1 ):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_encoder, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(in_c=in_channal, out_c=out_channal, node_n=node_n, seq_len=seq_len)
        self.bn1 = nn.BatchNorm1d(out_channal*node_n*seq_len)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(channal=out_channal, p_dropout=p_dropout, node_n=node_n, seq_len=seq_len))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(in_c=out_channal, out_c=out_channal, node_n=node_n, seq_len=seq_len)
        self.bn2 = nn.BatchNorm1d(out_channal*node_n*seq_len)
        self.reshape_conv = torch.nn.Conv2d(in_channels=in_channal, out_channels=out_channal, kernel_size=(1, 1))
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()


    def forward(self, x):

        y = self.gc1(x)
        b, c, n, l = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n, l).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        # b, c, n, l = y.shape
        # y = self.bn2(y.view(b, -1)).view(b, c, n, l).contiguous()
        # y = self.act_f(y)
        # y = self.do(y)

        return y + self.reshape_conv(x)

class GCN_decoder(nn.Module):
    def __init__(self, in_channal, out_channal, node_n=22, seq_len=20, p_dropout=0.3, num_stage=1):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_decoder, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(in_c=in_channal, out_c=in_channal, node_n=node_n, seq_len=seq_len)
        self.bn1 = nn.BatchNorm1d(in_channal*node_n*seq_len)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(channal=in_channal, p_dropout=p_dropout, node_n=node_n, seq_len=seq_len))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(in_c=in_channal, out_c=out_channal, node_n=node_n, seq_len=seq_len)
        self.bn2 = nn.BatchNorm1d(in_channal*node_n*seq_len)

        self.reshape_conv = torch.nn.Conv2d(in_channels=in_channal, out_channels=out_channal, kernel_size=(1, 1))

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, c, n, l = y.shape
        y = y.view(b, -1).contiguous()
        y = self.bn1(y).view(b, c, n, l).contiguous()
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y) + self.reshape_conv(x)

        return y

# 最原始的做法，att_map(22, 22)，Wq=conv(22, 22)
class SAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SAttentionBlock, self).__init__()
        self.in_channels = in_channels
        # self.out_channels = out_channels
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, joint, dct_n = input.shape
        # input: B, C, J, D -> B, J, C, D
        input = input.permute(0, 2, 1, 3)
        # input: B, J, C, D -> q: B, J, C * D
        q = self.query(input).view(batch_size, -1, channels * dct_n)
        # input: B, J, C, D -> k: B, C * D, J
        k = self.key(input).view(batch_size, -1, channels * dct_n).permute(0, 2, 1)
        # input: B, J, C, D -> v: B, J, C * D
        v = self.value(input).view(batch_size, -1, channels * dct_n)
        # (q: B, J, C * D) x (k: B, C * D, J) -> attn_matrix: B, J, J
        attn_matrix = torch.bmm(q, k) / math.sqrt(channels * dct_n)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        # out: B, J, C * D -> B, J, C, D -> B, C, J, D
        out = torch.bmm(attn_matrix, v)
        out = out.view(batch_size, joint, channels, dct_n).permute(0, 2, 1, 3)

        return self.gamma * out + input.permute(0, 2, 1, 3)

# # b站做法，att_map(22, 22)，Wq=linear(16*35，16*35)
# class SAttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(SAttentionBlock, self).__init__()
#         self.in_channels = in_channels
#         self.query = nn.Linear(in_channels, in_channels)
#         self.key = nn.Linear(in_channels, in_channels)
#         self.value = nn.Linear(in_channels, in_channels)
#         self.gamma = Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, input):
#         batch_size, channels, joint, dct_n = input.shape
#         # input: B, C, J, D -> input_i: B, J, C*D
#         # input_i = input.permute(0, 2, 1, 3).view(batch_size, joint, channels * dct_n)
#         input_i = input.permute(0, 2, 1, 3).reshape(batch_size, joint, channels * dct_n)
#         # input_i = input_i.reshape(batch_size, joint, channels * dct_n)
#         # input: B, J, C*D -> q: B, J, C * D
#         q = self.query(input_i)
#         # input: B, J, C*D -> k: B, C * D, J
#         k = self.key(input_i).permute(0, 2, 1)
#         # input: B, J, C*D -> v: B, J, C * D
#         v = self.value(input_i)
#         # (q: B, J, C * D) x (k: B, C * D, J) -> attn_matrix: B, J, J
#         attn_matrix = torch.bmm(q, k) / math.sqrt(channels * dct_n)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
#         attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
#         # out: B, J, C * D -> B, J, C, D -> B, C, J, D
#         out = torch.bmm(attn_matrix, v)
#         out = out.view(batch_size, joint, channels, dct_n).permute(0, 2, 1, 3)
#
#         return self.gamma * out + input

# 原始做法的改进，将J和D看成是图片的H和W，att_map(22*35, 22*35),Wq=conv(16, 16)
# class SAttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(SAttentionBlock, self).__init__()
#         self.in_channels = in_channels
#         self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.gamma = Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, input):
#         batch_size, channels, joint, dct_n = input.shape
#         # input: B, C, J, D
#         # input: B, C, J, D -> q: B, J * D, C
#         q = self.query(input).view(batch_size, -1, joint * dct_n).permute(0, 2, 1)
#         # input: B, J, C, D -> k: B, C, J * D
#         k = self.key(input).view(batch_size, -1, joint * dct_n)
#         # input: B, J, C, D -> v: B, C, J * D
#         v = self.value(input).view(batch_size, -1, joint * dct_n)
#         # (q: B, C, J * D) x (k: B, J * D, C) -> attn_matrix: B, J * D, J * D
#         attn_matrix = torch.bmm(q, k) / math.sqrt(channels)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
#         attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
#         # out: B, C, J * D -> B, C, J, D
#         out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
#         out = out.view(batch_size, channels, joint, dct_n)
#
#         return self.gamma * out + input

# # 原始做法的改进，将J和D看成是图片的H和W，att_map(22, 22),Wq=conv(16, 16)
# class SAttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(SAttentionBlock, self).__init__()
#         self.in_channels = in_channels
#         self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.gamma = Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, input):
#         batch_size, channels, joint, dct_n = input.shape
#         # input: B, C, J, D
#         # input: B, C, J, D -> q: B, J, C' * D
#         q = self.query(input).permute(0, 2, 1, 3).reshape(batch_size, joint, -1)
#         # input: B, C, J, D -> k: B, C' * D, J
#         k = self.key(input).permute(0, 1, 3, 2).reshape(batch_size, channels * dct_n, -1)
#         # input: B, J, C, D -> v: B, J, C' * D
#         v = self.value(input).permute(0, 2, 1, 3).reshape(batch_size, joint, -1)
#         # (q: B, J, C' * D) x (k: B, C' * D, J) -> attn_matrix: B, J, J
#         attn_matrix = torch.bmm(q, k) / math.sqrt(channels * dct_n)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
#         attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
#         # out: B, J, C' * D -> B, C', J, D
#         out = torch.bmm(attn_matrix, v)
#         out = out.view(batch_size, joint, channels, dct_n).permute(0, 2, 1, 3)
#
#         return self.gamma * out + input

# # DSTA做法
# class SAttentionBlock(nn.Module):
#     def __init__(self, in_channels, inter_channels, num_node=22, attentiondrop=0.3):
#         super(SAttentionBlock, self).__init__()
#         self.inter_channels = inter_channels
#         self.in_channels = in_channels
#
#         self.atts = torch.zeros((1, num_node, num_node)).cuda()
#         self.alphas = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)
#         # 有点问题
#         self.ff_nets = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 1, 1, padding=0, bias=True),
#             nn.BatchNorm2d(in_channels),
#         )
#         self.in_nets = nn.Conv2d(in_channels, 2 * in_channels, 1, bias=True)
#         self.alphas = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)
#         self.tan = nn.Tanh()
#         self.relu = nn.LeakyReLU(0.1)
#         self.drop = nn.Dropout(attentiondrop)
#
#
#     def forward(self, x):
#         N, C, V, T = x.size()
#         x = x.permute(0, 1, 3, 2)
#         attention = self.atts
#         q, k = torch.chunk(self.in_nets(x).view(N, 2 * self.inter_channels, T, V), 2, dim=1)
#         attention = attention + self.tan(torch.einsum('nctu,nctv->nuv', [q, k]) / (self.inter_channels * T)) * self.alphas
#         attention = self.drop(attention)
#         y = torch.einsum('nctu,nuv->nctv', [x, attention]).contiguous().view(N, self.in_channels, T, V)
#         y = self.relu(x + y)
#         y = self.ff_nets(y)
#         y = self.relu(x + y)
#
#         return y.permute(0, 1, 3, 2)


# class TAttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(TAttentionBlock, self).__init__()
#         self.in_channels = in_channels
#         # self.out_channels = out_channels
#         self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.gamma = Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, input):
#         batch_size, channels, joint, dct_n = input.shape
#         # input: B, C, J, D -> B, D, J, C
#         input = input.permute(0, 3, 2, 1)
#         # input: B, D, J, C -> q: B, D, C * J
#         q = self.query(input).view(batch_size, -1, channels * joint)
#         # input: B, D, J, C -> k: B, C * J, D
#         k = self.key(input).view(batch_size, -1, channels * joint).permute(0, 2, 1)
#         # input: B, D, J, C -> v: B, D, C * J
#         v = self.value(input).view(batch_size, -1, channels * joint)
#         # (q: B, D, C * J) x (k: B, C * J, D) -> attn_matrix: B, D, D
#         attn_matrix = torch.bmm(q, k) / math.sqrt(channels * joint)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
#         attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
#         # out: B, D, C * J
#         out = torch.bmm(attn_matrix, v)
#         out = out.view(batch_size, dct_n, joint, channels).permute(0, 3, 2, 1)
#
#         return self.gamma * out + input.permute(0, 3, 2, 1)

# b站做法，att_map(16, 16)，Wq=linear(22*35，22*35)
# class CAttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(CAttentionBlock, self).__init__()
#         self.in_channels = in_channels
#         # self.out_channels = out_channels
#         self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         # self.query = nn.Linear(in_channels, in_channels)
#         # self.key = nn.Linear(in_channels, in_channels)
#         # self.value = nn.Linear(in_channels, in_channels)
#         self.gamma = Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, input):
#         batch_size, channels, joint, dct_n = input.shape
#         # input: B, C, J, D -> B, C, J * D
#         input_i = input.view(batch_size, channels, joint * dct_n)
#         # input: B, C, J * D -> q: B, C, J * D
#         q = self.query(input_i)
#         # input: B, C, J * D -> k: B, C, J * D -> k.T: B, J * D, C
#         k = self.key(input_i).permute(0, 2, 1)
#         # input: B, D, J * D -> v: B, C, J * D
#         v = self.value(input_i)
#         # (q: B, C, J * D) x (k.T: B, J * D, C) -> attn_matrix: B, C, C
#         attn_matrix = torch.bmm(q, k) / math.sqrt(joint * dct_n)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
#         attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
#         # out: B, C, J * D -> B, C, J, D
#         out = torch.bmm(attn_matrix, v)
#         out = out.view(batch_size, channels, joint, dct_n)
#
#         return self.gamma * out + input

# # 最原始的做法，att_map(16, 16)，Wq=conv(16, 16)
# class CAttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(CAttentionBlock, self).__init__()
#         self.in_channels = in_channels
#         # self.out_channels = out_channels
#         self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.gamma = Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, input):
#         batch_size, channels, joint, dct_n = input.shape
#         # input: B, C, J, D
#         # input: B, C, J, D -> q: B, C, J * D
#         q = self.query(input).view(batch_size, -1, joint * dct_n)
#         # input: B, J, C, D -> k: B, J * D, C
#         k = self.key(input).view(batch_size, -1, joint * dct_n).permute(0, 2, 1)
#         # input: B, J, C, D -> v: B, C, J * D
#         v = self.value(input).view(batch_size, -1, joint * dct_n)
#         # (q: B, C, J * D) x (k: B, J * D, C) -> attn_matrix: B, C, C
#         attn_matrix = torch.bmm(q, k) / math.sqrt(joint * dct_n)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
#         attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
#         # out: B, C, J * D -> B, C, J, D
#         out = torch.bmm(attn_matrix, v)
#         out = out.view(batch_size, channels, joint, dct_n)
#
#         return self.gamma * out + input

# # C3 att_map(16, 16)，Wq=conv(22，22)
# class CAttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(CAttentionBlock, self).__init__()
#         self.in_channels = in_channels
#         # self.out_channels = out_channels
#         self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         # self.query = nn.Linear(in_channels, in_channels)
#         # self.key = nn.Linear(in_channels, in_channels)
#         # self.value = nn.Linear(in_channels, in_channels)
#         self.gamma = Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, input):
#         batch_size, channels, joint, dct_n = input.shape
#         # input: B, C, J, D -> B, J, C, D
#         input_i = input.permute(0, 2, 1, 3)
#         # input: B, J, C, D -> q: B, C, J * D
#         q = self.query(input_i).permute(0, 2, 1, 3).reshape(batch_size, channels, -1)
#         # input: B, J, C, D -> k: B, C, J * D -> k.T: B, J * D, C
#         k = self.key(input_i).permute(0, 1, 3, 2).reshape(batch_size, joint * dct_n, -1)
#         # input: B, J, C, D -> v: B, C, J * D
#         v = self.value(input_i).permute(0, 2, 1, 3).reshape(batch_size, channels, -1)
#         # (q: B, C, J * D) x (k.T: B, J * D, C) -> attn_matrix: B, C, C
#         attn_matrix = torch.bmm(q, k) / math.sqrt(joint * dct_n)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
#         attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
#         # out: B, C, J * D -> B, C, J, D
#         out = torch.bmm(attn_matrix, v)
#         out = out.view(batch_size, channels, joint, dct_n)
#
#         return self.gamma * out + input

class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.fc1(x)
        y = nn.ReLU(y)
        y = self.dropout(y)
        y = self.fc2(y)

        return y

# class SAttention(nn.Module):
#     def __init__(self, in_channels, opt):
#         super(SAttention, self).__init__()
#         self.opt = opt
#         self.in_channels = in_channels
#         self.bn1 = nn.BatchNorm1d(int(opt.d_model * opt.dct_n * opt.in_features / 3))
#         self.act_f = nn.Tanh()
#         self.do = nn.Dropout(opt.drop_out)
#         self.SAttention1 = SAttentionBlock(in_channels=int(self.opt.in_features / 3))
#         self.SAttention2 = SAttentionBlock(in_channels=int(self.opt.in_features / 3))
#
#     def forward(self, x):
#         y = self.SAttention1(x)
#         b, c, j, d = y.shape
#         y = y.reshape(b, -1).contiguous()
#         y = self.bn1(y).view(b, c, j, d).contiguous()
#         y = self.act_f(y)
#         y = self.do(y)
#         y = self.SAttention2(y)
#
#         return y