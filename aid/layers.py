# Copyright 2020 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout


def _scaled_dot_product_attention(q, k, v, mask=None):
  """ https://www.tensorflow.org/tutorials/text/transformer """
  
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(Layer):
  """ https://www.tensorflow.org/tutorials/text/transformer """
  
  def __init__(self, d_model=256, num_heads=4, **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads
    self.wq = Dense(d_model)
    self.wk = Dense(d_model)
    self.wv = Dense(d_model)
    self.dense = Dense(d_model)

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, inputs, training=False, mask=None):
    del training
    v, k, q = inputs
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = _scaled_dot_product_attention(
      q, k, v, mask)
    del attention_weights

    # (batch_size, seq_len_v, num_heads, depth)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # (batch_size, seq_len_v, d_model)
    concat_attention = tf.reshape(
      scaled_attention, (batch_size, -1, self.d_model))

    return self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

  def get_config(self):
    config = super(MultiHeadAttention, self).get_config()
    config.update({
      'num_heads': self.num_heads,
      'd_model': self.d_model})
    return config


class LayerNormalizedProjection(Layer):
  def __init__(self, dim=512, activation='elu', **kwargs):
    super(LayerNormalizedProjection, self).__init__(**kwargs)
    self.dim = dim
    self.activation = activation
    self.ln1 = LayerNormalization()
    self.fc1 = Dense(dim, activation=None, use_bias=False, name='fc1')
    self.act = Activation(self.activation)
    self.fc2 = Dense(dim, activation=None, use_bias=False, name='fc2')
    self.ln2 = LayerNormalization()

  def call(self, inputs, training=False):
    net = self.ln1(inputs)
    net = self.fc1(net)
    net = self.act(net)
    net = self.fc2(net)
    return self.ln2(net)

  def get_config(self):
    config = super(LayerNormalizedProjection, self).get_config()
    config.update({'dim': self.dim, 'activation': self.activation})
    return config


class BatchNormalizedProjection(Layer):
  def __init__(self, dim=512, activation='elu', **kwargs):
    super(BatchNormalizedProjection, self).__init__(**kwargs)
    self.dim = dim
    self.activation = activation
    self.bn1 = BatchNormalization(name='bn1')
    self.fc1 = Dense(dim, activation=None, use_bias=False, name='fc1')
    self.fc2 = Dense(dim, activation=None, use_bias=False, name='fc2')
    self.bn2 = BatchNormalization(name='bn2')

  def call(self, inputs, training=False):
    net = self.bn1(inputs, training=training)
    net = self.fc1(net)
    net = getattr(tf.nn, self.activation)(net)
    net = self.fc2(net)
    return self.bn2(net, training=training)

  def get_config(self):
    config = super(BatchNormalizedProjection, self).get_config()
    config.update({'dim': self.dim, 'activation': self.activation})
    return config


def _point_wise_feed_forward_network(d_model, dff, activation='relu'):
  return Sequential([
    Dense(dff, activation=activation),
    Dense(d_model)
  ])


class EncoderLayer(Layer):
  """ https://www.tensorflow.org/tutorials/text/transformer """
  
  def __init__(self, d_model=256, num_heads=4, dff=256, rate=0.1,
               eps=1e-6, **kwargs):
    super(EncoderLayer, self).__init__(**kwargs)

    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.rate = rate
    self.eps = eps

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = _point_wise_feed_forward_network(d_model, dff)
    self.layernorm1 = LayerNormalization(epsilon=eps)
    self.layernorm2 = LayerNormalization(epsilon=eps)
    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)

  def get_config(self):
    config = super(EncoderLayer, self).get_config()
    config.update({
      'd_model': self.d_model,
      'num_heads': self.num_heads,
      'dff': self.dff,
      'rate': self.rate,
      'eps': self.eps})
    return config
    
  def call(self, x, training=False):
    attn_output = self.mha([x, x, x], training=training)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)
    return out2, attn_output


class Encoder(Layer):
  def __init__(self, num_layers=2, d_model=256, num_heads=4, dff=256,
               rate=0.1, **kwargs):
    super(Encoder, self).__init__(**kwargs)
    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dff = dff
    self.rate = rate

    self.dropout = Dropout(rate)
    for i in range(num_layers - 1):
      setattr(self, f'enc_layer_{i}',
              EncoderLayer(d_model, num_heads, dff, rate))
    self.final = MultiHeadAttention(d_model, num_heads)

  def get_config(self):
    config = super(Encoder, self).get_config()
    config.update({
      'num_layers': self.num_layers,
      'd_model': self.d_model,
      'num_heads': self.num_heads,
      'dff': self.dff,
      'rate': self.rate})
    return config

  def call(self, x, training=False):
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = self.dropout(x, training=training)
    features = []
    for i in range(self.num_layers - 1):
      layer = getattr(self, f'enc_layer_{i}')
      x, attn_output = layer(x, training=training)
      feat = tf.reduce_mean(attn_output, axis=1)
      features.append(feat)
    attn_output = self.final([x, x, x], training=training)
    feat = tf.reduce_mean(attn_output, axis=1)
    features.append(feat)
    return tf.concat(features, axis=-1)
