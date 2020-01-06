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


from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D

from aid.features import F
from aid.layers import Encoder
from aid.layers import BatchNormalizedProjection
from aid.layers import LayerNormalizedProjection


class IurNet(tf.keras.Model):
  def __init__(self, num_symbols=None, num_action_types=None,
               padded_length=None, episode_len=16, embedding_dim=512,
               features='syms+action_type+hour', num_layers=2,
               d_model=256, num_heads=4, dff=256, dropout_rate=0.1,
               subword_embed_dim=512, action_embed_dim=512,
               filter_activation='relu', num_filters=256,
               min_filter_width=2, max_filter_width=5,
               final_activation='relu', **kwargs):
    super(IurNet, self).__init__(**kwargs)
    self.embedding_dim = embedding_dim
    self.num_symbols = num_symbols
    self.num_action_types = num_action_types
    self.padded_length = padded_length
    self.episode_len = episode_len
    self.features = features
    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.dropout_rate = dropout_rate
    self.subword_embed_dim = subword_embed_dim
    self.action_embed_dim = action_embed_dim
    self.min_filter_width = min_filter_width
    self.max_filter_width = max_filter_width
    self.num_filters = num_filters
    self.filter_activation = filter_activation
    self.final_activation = final_activation
    self.subword_embedding = Embedding(self.num_symbols, self.subword_embed_dim,
                                       name='subword_embedding')
    self.action_embedding = Embedding(self.num_action_types,
                                      self.action_embed_dim,
                                      name='action_embedding')
    for width in range(self.min_filter_width, self.max_filter_width + 1):
      setattr(self, f'conv_{width}',
              Conv1D(self.num_filters, width, activation=self.filter_activation))
    self.dense_1 = Dense(self.d_model)
    self.encoder = Encoder(self.num_layers, self.d_model,
                           self.num_heads, self.dff, rate=self.dropout_rate)
    self.mlp = LayerNormalizedProjection(self.embedding_dim,
                                         activation=self.final_activation)

  @tf.function
  def call(self, inputs, training=False):
    features = []
    
    # Extract text features
    net = inputs[F.SYMBOLS.value]
    batch_size = tf.shape(net)[0]
    net = tf.reshape(net, [-1, self.padded_length])
    swe = self.subword_embedding(net)
    fs = []
    for width in range(self.min_filter_width, self.max_filter_width + 1):
      layer = getattr(self, f'conv_{width}')
      net = layer(swe)
      net = tf.reduce_max(net, axis=1, keepdims=False)
      fs.append(net)
    net = tf.concat(fs, axis=-1)
    feature_dim = net.get_shape()[-1]
    net = tf.reshape(net, [batch_size, self.episode_len, feature_dim])
    features.append(net)

    # Action embedding
    embedded_actions = self.action_embedding(inputs[F.ACTION_TYPE.value])
    features.append(embedded_actions)

    # Hour embedding
    hour = inputs[F.HOUR.value]
    features.append(tf.one_hot(hour, 24, dtype=tf.float32, name='hour_onehot'))

    net = tf.concat(features, axis=-1)
    net = self.dense_1(net)
    net = self.encoder(net, training=training)
    net = self.mlp(net, training=training)
    return net
