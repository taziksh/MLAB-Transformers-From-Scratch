"""
Student implementation of BERT.

Complete this file from top to bottom and pass the tests in ../tests/test_bert.py.
"""

import typing

import numpy as np
import torch as t
from torch import nn
from einops import rearrange, repeat
from torchtyping import TensorType

import pdb


class LayerNorm(nn.Module):
    """
    Layer normalization. See https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input of size
        [*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]
        If a single integer is used, it is treated as a singleton list, and this module will
        normalize over the last dimension which is expected to be of that specific size.

    Attributes:
        LayerNorm.weight: The learnable weights gamma of the module of shape normalized_shape.
            The values are initialized to 1.
        LayerNorm.bias: The learnable bias beta of the module of shape normalized_shape.
            The values are initialized to 0.

    Dependencies:
        None.

    Hints:
        Norm over the last len(normalized_shape) dimensions, not simply all but the first dimension.
    """

    def __init__(self, normalized_shape: typing.Union[int, tuple]):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.weight = t.ones(self.normalized_shape)
        self.bias = t.zeros(self.normalized_shape)

    def forward(self, input: TensorType[...]):
        """Apply Layer Normalization over a mini-batch of inputs."""
        eps = 1e-05
        #TODO: why is this off by ~5%, chatGPT says it's ok
        # dims_to_norm = tuple(range(1, len(input.shape)))
        # mean = t.mean(input, dim=dims_to_norm, keepdim=True)
        # var = t.var(input, dim=dims_to_norm, keepdim=True)

        num_normed_dims = self.weight.dim()
        normalizing_dims = tuple(range(-num_normed_dims, 0))  # (-(n-1), ..., -2, -1)
        var, mean = t.var_mean(input, normalizing_dims, unbiased=False)
        var = rearrange(var, 'x ... -> x ...' + ' ()' * num_normed_dims)
        mean = rearrange(mean, 'x ... -> x ...' + ' ()' * num_normed_dims)

        return self.weight * (input - mean) / t.sqrt(var + eps) + self.bias

class Embedding(nn.Module):
    """
    A simple lookup table storing embeddings of a fixed dictionary and size.
    See https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

    This module is often used to store word embeddings and retrieve them using indices. The input
    to the module is a list of indices, and the summed_embeddings is the corresponding word
    embeddings.

    Args:
        Embedding.weight (Tensor): The learnable weights of the module of shape
        (num_embeddings, embedding_dim) initialized from a normal distribution (mu=0, sigma=1).

    Dependencies:
        None.

    Hints:
        Use torch.randn to create a random tensor sampled by a normal distribution.
    """

    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.weight = t.randn((vocab_size, embed_size))

    def forward(self, input):
        return self.weight[input]
        """Look up the input list of indices in the embedding matrix."""

class BertEmbedding(nn.Module):
    """
    BERT embedding layer.
    See §3.4 of Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf#page=5

    You should create Embedding parameters for tokens, positions, and token types/segments.
    BERT uses learned position embeddings rather than sinusoidal position embeddings.
    The forward pass sums the three embeddings then passes them through LayerNorm and Dropout.

    For an explanation of segments, see Input/Output Representations from the BERT paper
    https://arxiv.org/pdf/1810.04805.pdf#page=4

    Note that BERT does not scale the embeddings by sqrt(d_model).

    Args:
        vocab_size, hidden_size, max_position_embeddings, type_vocab_size (int): Embeddings dims.
        dropout: Dropout rate.

    Attributes:
        BertEmbedding.token_embedding (Embedding): Token embeddings.
        BertEmbedding.position_embedding (Embedding): Position embeddings.
        BertEmbedding.token_type_embedding (Embedding): Token type/segment embeddings.
        BertEmbedding.layer_norm (LayerNorm): Layer normalization.
        BertEmbedding.dropout (nn.Dropout): Dropout module.

    Dependencies:
        Embedding, LayerNorm

    Hints:
        Initialization order matters for our seeded random tests. Initialize token_embedding, then
            position_embedding, then token_type_embedding.
        Use torch.arange to create an ascending integer list to index your position embeddings.
            You'll have to take care to repeat/expand your tensors to the appropriate sizes.
    """

    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int,
                 type_vocab_size: int, dropout: float):
        super().__init__()
        self.token_embedding = Embedding(vocab_size=vocab_size, embed_size=hidden_size)
        self.position_embedding = Embedding(vocab_size=max_position_embeddings, embed_size=hidden_size)
        self.token_type_embedding = Embedding(vocab_size=type_vocab_size, embed_size=hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids):
        """Add embeddings and apply layer norm and dropout."""
        # pdb.set_trace()
        # in NLP, shape is (batch_size, sequence_length
        sequence_length = input_ids.shape[-1]
        '''
        Here's a step-by-step explanation of how the line works:
            1. t.arange(sequence_length) creates a 1-dimensional tensor of shape (sequence_length,) containing a sequence of integers from 0 to sequence_length - 1.
            2. unsqueeze(0) adds a new dimension to the tensor at position 0, resulting in a 2-dimensional tensor of shape (1, sequence_length).
            3. expand_as(input_ids) replicates the tensor along all dimensions to match the shape of input_ids. In this case, it replicates the tensor along the first dimension (the batch dimension) to match the batch size of input_ids, resulting in a tensor of shape (batch_size, sequence_length).
        '''
        position_ids = t.arange(sequence_length).unsqueeze(0).expand_as(input_ids)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(position_ids) + self.token_type_embedding(token_type_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadedSelfAttention(nn.Module):
    """
    Multi-headed scaled dot product self attention.
    See §3.2 in Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf#page=3

    Each head's attention is performed on a 1/num_heads split of hidden_size. I.e. if hidden_size
    is 768 and num_heads is 12, we do attention on vectors of size 64.

    Args:
        num_heads (int): Number of attention heads.
        hidden_size (int): The size of the embedding dimension.

    Attributes:
        MultiHeadedSelfAttention.project_query (nn.Linear): Projects the input into query space.
        MultiHeadedSelfAttention.project_key (nn.Linear): Projects the input into key space.
        MultiHeadedSelfAttention.project_value (nn.Linear): Projects the input into value space.
        MultiHeadedSelfAttention.project_output (nn.Linear): Projects the concatenated and weighted
            values back into hidden representation space.

    Dependencies:
        None.

    Hints:
        Think about how to do single-headed attention with just 1 sequence, then with a batch of
            sequences, then with multiple heads across a batch of sequences.
        An efficient way of handling multiple heads is to first project the input into Q, K, and V
            vectors of size hidden_size then to split each of those along the hidden_size dimension
            so there are num_heads number of smaller vectors. At the end, concatenate them together
            back into a vector of size hidden_size. einops and einsum are particularly useful here.
    """

    def __init__(self, num_heads: int, hidden_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.project_query = t.nn.Linear(hidden_size, hidden_size)
        self.project_key = t.nn.Linear(hidden_size, hidden_size)
        self.project_value = t.nn.Linear(hidden_size, hidden_size)
        self.project_output = t.nn.Linear(hidden_size, hidden_size)

    def forward(self, input: TensorType['batch', 'seq_length', 'hidden_size'],
                attn_mask: typing.Optional[TensorType['batch', 'seq_length']] = None
                ) -> TensorType['batch', 'seq_length', 'hidden_size']:
        """Apply multi-headed scaled dot product self attention with an optional attention mask."""
        split_pattern = 'batch seq_length (heads dim_k) -> batch heads seq_length dim_k'
        queries = rearrange(self.project_query(input), split_pattern, heads = self.num_heads)
        keys = rearrange(self.project_key(input), split_pattern, heads = self.num_heads)
        values = rearrange(self.project_value(input), split_pattern, heads = self.num_heads)
        # seq_length, hidden_split @ hidden_split, seq_length

        attn = t.einsum('bnqh, bnkh -> bnqk', [queries, keys]) / np.sqrt(queries.shape[-1])
        attn = t.softmax(attn, dim=-1)

        if attn_mask is not None:
            attn.masked_fill_(attn_mask, -1e9)
        values = t.einsum('bnqk, bnkh -> bnqh', [attn, values])
        values = rearrange(values, 'batch heads seq_length dim_k -> batch seq_length (heads dim_k)', heads = self.num_heads)
        output = self.project_output(values)
        return output

class GELU(nn.Module):
    """
    Applies the Gaussian Error Linear Units function with no approximation.
    See https://pytorch.org/docs/stable/generated/torch.nn.GELU.html

    GELU(x) = x * Φ(x) where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.

    Dependencies:
        None.

    Hint:
        The CDF of a normal distribution with a mean of 0 and variance of 1 is
            0.5 * (1.0 + erf(value / sqrt(2))) where erf is the error function (torch.erf).
    """
    def cdf(self, input):
        """Apply the CDF function."""
        return 0.5 * (1.0 + t.erf(input / np.sqrt(2.0)))

    def forward(self, input):
        """Apply the GELU function."""
        cdf = self.cdf(input)
        return input * cdf


class BertMLP(nn.Module):
    """
    BERT MLP/fully-connected feed forward network layers.
    See §3.3 in Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf#page=5

    Implement 2 linear layers that go from hidden_size to intermediate_size and back
    with a GELU activation in the middle.

    Args:
        hidden_size (int): Token embedding size.
        intermediate_size (int): Intermediate hidden layer size.

    Attributes:
        BertMLP.lin1 (nn.Linear): First linear layer.
        BertMLP.lin2 (nn.Linear): Second linear layer.

    Dependencies:
        GELU
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.lin1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = GELU()
        self.lin2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, input: TensorType['batch_size', 'seq_length', 'hidden_size']
                ) -> TensorType['batch_size', 'seq_length', 'hidden_size']:
        """Apply linear projection, GELU, and another linear projection."""
        res = self.lin1(input)
        res = self.gelu(res)
        res = self.lin2(res)
        return res


class BertBlock(nn.Module):
    """
    One of the BERT transformer encoder blocks/layers.
    See §3.1/Figure 1 in Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf#page=3

    Mainly consists of self attention, adding a residual connection, layer norm, feed-forward/MLP,
    adding another residual connection, and layer normalization.

    We apply dropout to the output of each sub-layer, before it is added to the sub-layer input
    and normalized (i.e. before each residual connection right before layer normalization).

    Args:
        hidden_size (int): Token embedding size.
        intermediate_size (int): Intermediate hidden layer size within the MLP layer.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.

    Attributes:
        BertBlock.attention (MultiHeadedSelfAttention): Multi-headed self attention layer.
        BertBlock.layernorm1 (LayerNorm): First layer normalization layer.
        BertBlock.mlp (BertMLP): Fully connected MLP layer.
        BertBlock.layernorm2 (LayerNorm): Second layer normalization layer.
        BertBlock.dropout (nn.Dropout): Dropout module.

    Dependencies:
        MultiHeadedSelfAttention
        LayerNorm
        BertMLP

    Hints:
        If you are failing only test_bert_block_with_dropout, it may be a pointless random number
            discrepancy that's not your fault. See the comment on that test in ../tests/test_bert.py
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadedSelfAttention(num_heads, hidden_size)
        self.layernorm1 = LayerNorm(hidden_size)
        self.mlp = BertMLP(hidden_size, intermediate_size)
        self.layernorm2 = LayerNorm(hidden_size)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, input, attn_mask=None):
        """Apply each of the layers in the block."""
        res = input + self.attention(input, attn_mask=attn_mask)
        res = self.layernorm1(res)
        res = res + self.mlp(res)
        res = self.layernorm2(res)
        res = self.dropout(res)
        return res


class Bert(nn.Module):
    """
    The full BERT transformer encoder model which goes from token IDs to logits.

    Consists of the embedding, blocks, and a token output head. The token output head layer has a
    linear layer to the hidden size, GELU, layer norm, the an unembedding linear layer to the
    vocabulary size to produce ouput logits over all tokens.

    Args:
        hidden_size (int): Token embedding size.
        vocab_size (int): Vocabulary size.
        max_position_embeddings (int): Maximum number of tokens per sequence.
        type_vocab_size (int): Number of different token type/sequence tokens.
        dropout (float): Dropout probability.
        intermediate_size (int): Intermediate hidden layer size within the MLP layers.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer blocks.

    Attributes:
        Bert.embed (BertEmbedding): Embedding layer.
        Bert.blocks (nn.Sequential): Sequence of BertBlocks.
        Bert.lin (nn.Linear): Output head linear layer.
        Bert.gelu (GELU): Output head GELU layer.
        Bert.layer_norm (LayerNorm): Output head layer normalization layer.
        Bert.unembed (nn.Linear): Output layer unembedding layer.

    Dependencies:
        BertEmbedding
        BertBlock
        GELU
        LayerNorm

    Hints:
        Assume all tokens are in the same segment/have the same token type with
            token_type_ids = t.zeros_like(input_ids, dtype=t.int64)
    """

    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int,
                 type_vocab_size: int, dropout: float, intermediate_size: int,
                 num_heads: int, num_layers: int):
        super().__init__()
        self.embed = BertEmbedding(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout)
        self.blocks = t.nn.Sequential(*[BertBlock(hidden_size, intermediate_size, num_heads, dropout) for i in range(num_layers)])
        self.lin = t.nn.Linear(hidden_size, hidden_size)
        self.gelu = GELU()
        self.layer_norm = LayerNorm(hidden_size)
        self.unembed = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        """Apply embedding, blocks, and token output head."""
        token_type_ids = t.zeros_like(input_ids, dtype=t.int64)
        res = self.embed(input_ids, token_type_ids)
        res = self.blocks(res)
        res = self.lin(res)
        res = self.gelu(res)
        res = self.layer_norm(res)
        enc = self.unembed(res)
        return enc

class BertWithClassify(nn.Module):
    """
    The full BERT transformer encoder model which goes from token IDs to logits.

    Just like the above Bert class, but in addition to the token output head, apply a classification
    head to the CLS encoding output of the transformer blocks. The classification head consists just
    of dropout and a linear layer from the hidden size to the number of classes.

    Args:
        Same args as BERT.
        num_classes: Number of output classes.

    Attributes:
        Same attributes as BERT.
        BertWithClassify.classification_dropout (nn.Dropout): Classification dropout layer.
        BertWithClassify.classification_head (nn.Linear): Classification linear layer.

    Dependencies:
        BertEmbedding
        BertBlock
        GELU
        LayerNorm

    Hints:
        The encoded representations output from the transformer blocks are used to get both the
            token output logits and the classification logits. Don't apply the classification
            head to the token output logits from the token output head.
        The classification output uses just the CLS token encoding, which you can get from the
            output with encodings[:, 0] (the 0th token for all sequences in the batch).
    """

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                 dropout, intermediate_size, num_heads, num_layers, num_classes):
        super().__init__()
        self.embed = BertEmbedding(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout)
        self.blocks = t.nn.Sequential(*[BertBlock(hidden_size, intermediate_size, num_heads, dropout) for i in range(num_layers)])
        self.lin = t.nn.Linear(hidden_size, hidden_size)
        self.gelu = GELU()
        self.layer_norm = LayerNorm(hidden_size)
        self.unembed = nn.Linear(hidden_size, vocab_size)
        self.classification_dropout = t.nn.Dropout(dropout)
        self.classification_head = nn.Linear(num_classes, num_classes)

    def forward(self, input_ids):
        """Returns a tuple of logits, classifications."""
        token_type_ids = t.zeros_like(input_ids, dtype=t.int64)
        res = self.embed(input_ids, token_type_ids)
        res = self.blocks(res)
        res = self.lin(res)
        res = self.gelu(res)
        res = self.layer_norm(res)
        enc = self.unembed(res)       
        enc = self.classification_dropout(enc) 
        res = self.classification_head(enc[:, 0])
        return res
