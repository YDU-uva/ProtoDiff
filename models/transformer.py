"""
This file contains the G.pt model and its building blocks (minGPT without masking, etc.).
"""
import math

import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Modified from: https://github.com/karpathy/minGPT
    A version of minGPT without masking (we use diffusion instead).
    """

    def __init__(
        self, input_prototype_sizes, len_input,
        n_layer=12, n_head=12, n_embd=512, encoder_depth=1, attn_pdrop=0.0,
        resid_pdrop=0.0, embd_pdrop=0.0,
    ):
        super().__init__()
        # Determine how many parameters are placed into each individual Transformer token:
        self.input_splits = input_prototype_sizes
        block_size = len_input


        # input embedding stem
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, attn_pdrop, resid_pdrop) for _ in range(n_layer)]
        )

        self.block_size = block_size
        self.input_prototype_projections = self.build_encoder(n_embd, encoder_depth, self.input_splits)
        self.ln_in = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    @staticmethod
    def build_encoder(n_embd, encoder_depth, input_splits):
        input_prototype_projections = nn.ModuleList()
        for len_input in input_splits:
            in_proj = [nn.Linear(len_input, n_embd, bias=False)]
            for _ in range(encoder_depth - 1):
                in_proj.append(nn.GELU())
                in_proj.append(nn.Linear(n_embd, n_embd, bias=False))
            in_proj = nn.Sequential(*in_proj)
            input_prototype_projections.append(in_proj)
        return input_prototype_projections

    @staticmethod
    def build_decoder(n_embd, decoder_depth, output_splits):
        # Create a unique MLP decoder for each noised token
        output_parameter_projections = nn.ModuleList()
        for output_chunk_size in output_splits:
            out_proj = []
            for _ in range(decoder_depth - 1):
                out_proj.append(nn.Linear(n_embd, n_embd, bias=False))
                out_proj.append(nn.GELU())
            out_proj.append(nn.Linear(n_embd, output_chunk_size, bias=False))
            out_proj = nn.Sequential(*out_proj)
            output_parameter_projections.append(out_proj)
        return output_parameter_projections

    def get_block_size(self):
        return self.block_size

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def encode_prototypes(self, parameters):
        """
        Chunk input parameter vector, apply per-chunk encoding, and
        stack projected chunks along the sequence (token) dimension.
        """
        assert len(parameters) == 3
        representations = []
        for parameter, in_proj in zip(parameters, self.input_prototype_projections):
            representations.append(in_proj(parameter))
        representations = torch.cat(representations, dim=1)  # (b, t, d)
        representations = self.ln_in(representations)
        assert representations.dim() == 3
        return representations

    def forward(self, x):
        embeddings = self.encode_prototypes(x)
        b, t, d = embeddings.size()
        assert t == self.block_size, f"Expected {self.block_size} tokens on dim=1, but got {t}"
        # forward the GPT model
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(embeddings + position_embeddings)
        x = self.blocks(x)

        return x


class FrequencyEmbedder(nn.Module):

    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer('frequencies', frequencies)

    def forward(self, x):
        # x should be of size (N,) or (N, D)
        N = x.size(0)
        if x.dim() == 1:  # (N,)
            x = x.unsqueeze(1)  # (N, D) where D=1
        x_unsqueezed = x.unsqueeze(-1).to('cuda', torch.float)  # (N, D, 1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed  # (N, D, num_frequencies)
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        #embedded = torch.cat([s, c], dim=-1).view(N, -1)  # (N, D * 2 * num_frequencies + D)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(N, -1)
        return embedded


class Gpt(nn.Module):

    """
    The G.pt model.
    """

    def __init__(
        self,
        prototype_sizes=512,                    # A list of integers indicating the total number of parameters in each layer
        num_frequencies=256,                # number of frequencies sampled for embedding scalars
        max_freq_log2=20,                   # max log2 frequency for embedding scalars
        predict_xstart=True,                # if True, G.pt predicts signal (False = predict noise)
        use_global_residual=True,
        **gpt_kwargs                        # Arguments for the Transformer model (depth, heads, etc.)
    ):
        super().__init__()
        self.predict_xstart = predict_xstart
        input_prototype_sizes = self.compute_token_sizes(prototype_sizes, num_frequencies)

        self.decoder = GPT(input_prototype_sizes, **gpt_kwargs)
        self.scalar_embedder = FrequencyEmbedder(num_frequencies, max_freq_log2)

        # Initialize with identity output:
        self.use_global_residual = use_global_residual

    @staticmethod
    def get_scalar_token_size(num_frequencies):
        """
        Computes the size of each metadata token after being projected by the frequency embedder.
        """
        return num_frequencies * 2 + 1

    def compute_token_sizes(self, parameter_sizes, num_frequencies):
        input_parameter_sizes = [deepcopy(parameter_sizes)]
        # account for the second weight vector that will be input:
        input_parameter_sizes.append(input_parameter_sizes[0])
        # Account for the scalar inputs (diffusion timestep and loss/error/return inputs):
        scalar_token_size = [self.get_scalar_token_size(num_frequencies)]
        input_parameter_sizes.extend(scalar_token_size)
        return input_parameter_sizes

    @torch.no_grad()
    def gradient_norm(self):
        """
        Computes the gradient norm for monitoring purposes.
        """
        total_norm = 0.0
        for p in self.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm


    def forward(self, x, x_ori, t):
        x_prototype = torch.mean(x_ori, dim=1)
        x_diff = x - x_prototype
        normalize = x_diff.view(x_diff.shape[0], -1).std(-1).unsqueeze(1)
        x_norm = x_diff / normalize
        t_embedding = self.scalar_embedder(t)
        inp = [torch.unsqueeze(x_norm, dim=1), x_ori, torch.unsqueeze(t_embedding, dim=1)]
        output = self.decoder(inp)

        return output
