import torch.nn as nn
import torch
import numpy as np

from config import *


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, n_head * d_k)
        self.W_K = nn.Linear(d_model, n_head * d_k)
        self.W_V = nn.Linear(d_model, n_head * d_v)
        self.scale_product = ScaledDotProductAttention()
        self.linear = nn.Linear(n_head * d_v, d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch, tgt_len, d_model]
        :param K: [batch, src_len, d_model]
        :param V: [batch, src_len, d_model]
        :param attn_mask: [batch, tgt_len, src_len]  padding mask + decoder mask
        :returns attn_value: [batch, tgt_len, d_model]
        :returns attn_mask: [batch, n_head, tgt_len, src_len]
        """
        batch_size = Q.shape[0]
        # q_s: [batch, n_head, tgt_len, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)
        # k_s: [batch, n_head, src_len, d_k]
        k_s = self.K_Q(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)
        # v_s: [batch, n_head, src_len, d_v]
        v_s = self.V_Q(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)
        # value: [batch, n_head, tgt_len, d_v] attn: []
        """scale dot product attention"""
        # attn_mask: [batch_size, len_q, len_k] -> [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        attn_value, attn = self.scale_product(q_s, k_s, v_s, attn_mask)
        """concat"""
        # value: [batch, tgt_len, n_head * d_v]
        # view需要变量内存连续，transpose后往往不连续，所以用contiguous把内存连续起来
        attn_value = attn_value.transpose(1, 2).contiguous().view(batch_size, -1, n_head * d_v)
        """linear"""
        # attn_value: [batch, tgt_len, d_model]
        attn_value = self.linear(attn_value)
        return attn_value, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_s, k_s, v_s, attn_mask):
        """
        :param q_s: [batch, n_head, tgt_len, d_k]
        :param k_s: [batch, n_head, src_len, d_k]
        :param v_s: [batch, n_head, src_len, d_v]
        :param attn_mask: [batch, n_head, tgt_len, src_len]
        :return: attn_value: [batch, ]
        """
        # score: [batch, n_head, tgt_len, src_len]
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill(attn_mask, -torch.inf)
        attn = nn.Softmax(dim=-1)(scores)
        # attn_value: [batch, n_head, tgt_len, d_v]
        attn_value = torch.matmul(attn, v_s)
        return attn_value, attn
