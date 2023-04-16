import torch.nn as nn
import torch

from config import *
from transformer_decoder import TransformerDecoder
from transformer_encoder import TransformerEncoder
from utils import AddNorm, PositionalEmbedding


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.pos_emb = PositionalEmbedding()
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, enc_input, dec_input):
        """
        :param enc_input: [batch, src_len, d_model]
        :param dec_input: [batch, tgt_len, d_model]
        :return: output: [batch, tgt_len, tgt_vocab_size]
        """
        """encoder"""
        enc_emb = nn.Embedding(enc_input, embedding_dim=d_model)
        pos_enc_emb = self.pos_emb(enc_emb)
        enc_out = self.encoder(pos_enc_emb)
        """decoder"""
        dec_emb = nn.Embedding(dec_input, embedding_dim=d_model)
        pos_dec_emb = self.pos_emb(dec_emb)
        dec_out = self.decoder(pos_dec_emb, enc_out)
        """linear"""
        # [batch, tgt_len, tgt_vocab_size]
        linear_out = self.linear(dec_out)
        """softmax"""
        # for each token => dim=1
        out = nn.Softmax(dim=1)(linear_out)
        return out

