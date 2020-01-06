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
# =============================================================================

import tensorflow as tf
import math


class AngularMargin(tf.keras.layers.Layer):
  def __init__(self, output_dim, scale=64., margin=0.5):
    super(AngularMargin, self).__init__()
    self.output_dim = output_dim
    self.scale = scale
    self.margin = margin
    self.cos_m = math.cos(margin)
    self.sin_m = math.sin(margin)
    self.threshold = math.cos(math.pi - margin)
    self.mm = self.sin_m * margin

  def build(self, input_shape):
    embedding_shape, labels_shape = input_shape
    del labels_shape
    self.kernel = self.add_weight(
      name='kernel',
      shape=(embedding_shape[-1], self.output_dim),
      initializer='glorot_uniform',
      trainable=True)
    
  def call(self, inputs):
    embedding, labels = inputs
    embedding = tf.linalg.l2_normalize(embedding, axis=1)
    weights = tf.linalg.l2_normalize(self.kernel, axis=0)
    cos_t = tf.linalg.matmul(embedding, weights)
    cos_t2 = tf.square(cos_t)
    sin_t2 = tf.subtract(1., cos_t2)
    sin_t = tf.sqrt(sin_t2)
    cos_mt = self.scale * tf.subtract(tf.multiply(cos_t, self.cos_m),
                                      tf.multiply(sin_t, self.sin_m))
    cond_v = cos_t - self.threshold
    cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
    keep_val = self.scale * (cos_t - self.mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)
    mask = tf.one_hot(labels, self.output_dim)
    inv_mask = tf.subtract(1., mask)
    s_cos_t = tf.multiply(self.scale, cos_t)
    logits = tf.add(tf.multiply(s_cos_t, inv_mask),
                    tf.multiply(cos_mt_temp, mask))
    return logits

  def get_config(self):
    return {'output_dim': self.output_dim, 'scale': self.scale, 'margin': self.margin}
