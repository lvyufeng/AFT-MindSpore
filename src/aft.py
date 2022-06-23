import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Parameter
from mindspore.common.initializer import initializer

class AFTLocal(nn.Cell):
    def __init__(self, d_model: int, seq_len: int, local_window_size: int, has_bias: bool = True):
        super().__init__()
        self.local_window_size = local_window_size

        self.query = nn.Dense(d_model, d_model, has_bias=has_bias)
        self.key = nn.Dense(d_model, d_model, has_bias=has_bias)
        self.value = nn.Dense(d_model, d_model, has_bias=has_bias)

        self.pos_bias = Parameter(initializer('zeros', (seq_len, seq_len)), name='pos_bias')
        self.local_mask = Parameter(self.create_local_mask(seq_len, local_window_size), name='local_mask')

        self.activation = nn.Sigmoid()
        self.output = nn.Dense(d_model, d_model)
        self.einsum = ops.Einsum('ijb,jbd->ibd')
    
    @staticmethod
    def create_local_mask(seq_len, local_window_size):
        local_mask = mnp.ones((seq_len, seq_len), dtype=mindspore.bool_)
        local_mask = mnp.tril(local_mask, local_window_size - 1)
        local_mask = mnp.triu(local_mask, -(local_window_size - 1))
    
        return local_mask

    def construct(self, query, key, value, mask=None):
        seq_len, _, _ = query.shape

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        pos_bias = self.pos_bias[:seq_len, :seq_len] * self.local_mask[:seq_len, :seq_len]
        pos_bias = pos_bias.expand_dims(-1)
        if mask is not None:
            pos_bias.masked_fill(~mask, float('-inf'))
        max_key = key.max(axis=0, keepdims=True)
        max_pos_bias = pos_bias.max(axis=1,  keepdims=True)

        exp_key = ops.exp(key - max_key)
        exp_pos_bias = ops.exp(pos_bias - max_pos_bias)

        num = self.einsum((exp_pos_bias, exp_key * value))
        den = self.einsum((exp_pos_bias, exp_key))

        y = self.activation(query) * num / den

        return self.output(y)

class AFTFull(nn.Cell):
    def __init__(self, d_model: int, seq_len: int, has_bias: bool = True):
        super().__init__()
        self.query = nn.Dense(d_model, d_model, has_bias=has_bias)
        self.key = nn.Dense(d_model, d_model, has_bias=has_bias)
        self.value = nn.Dense(d_model, d_model, has_bias=has_bias)

        self.pos_bias = Parameter(initializer('zeros', (seq_len, seq_len)), name='pos_bias')

        self.activation = nn.Sigmoid()
        self.output = nn.Dense(d_model, d_model)
        self.einsum = ops.Einsum('ijb,jbd->ibd')

    def construct(self, query, key, value, mask=None):
        seq_len, _, _ = query.shape

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        pos_bias = self.pos_bias[:seq_len, :seq_len]
        pos_bias = pos_bias.expand_dims(-1)
        if mask is not None:
            pos_bias.masked_fill(~mask, float('-inf'))
        max_key = key.max(axis=0, keepdims=True)
        max_pos_bias = pos_bias.max(axis=1,  keepdims=True)

        exp_key = ops.exp(key - max_key)
        exp_pos_bias = ops.exp(pos_bias - max_pos_bias)

        num = self.einsum((exp_pos_bias, exp_key * value))
        den = self.einsum((exp_pos_bias, exp_key))

        y = self.activation(query) * num / den

        return self.output(y)

class AFTSample(nn.Cell):
    def __init__(self, d_model: int, has_bias: bool = True):
        super().__init__()
        self.query = nn.Dense(d_model, d_model, has_bias=has_bias)
        self.key = nn.Dense(d_model, d_model, has_bias=has_bias)
        self.value = nn.Dense(d_model, d_model, has_bias=has_bias)

        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.output = nn.Dense(d_model, d_model)
        self.einsum = ops.Einsum('ijb,jbd->ibd')

    def construct(self, query, key, value):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        weights = self.softmax(key) * value
        weights = weights.sum(0, keepdims=True)

        y = self.activation(query) * weights

        return self.output(y)
