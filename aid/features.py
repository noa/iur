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


import json
from pprint import pformat
from glob import glob
from enum import Enum

from absl import logging

import tensorflow as tf
import numpy as np


class F(Enum):
  SYMBOLS = "syms"
  NUM_SYMBOLS_PER_POST = "lens"
  NUM_POSTS = "num_posts"
  ACTION_TYPE = "action_type"
  AUTHOR_ID = "author_id"
  HOUR = "hour"


FEATURE_SHAPE = {
  F.SYMBOLS: [None],
  F.NUM_SYMBOLS_PER_POST: [None],
  F.NUM_POSTS: [None],
  F.AUTHOR_ID: [None],
  F.ACTION_TYPE: [None],
  F.HOUR: [None]
}


FEATURE_FORMAT = {
  F.SYMBOLS: tf.io.FixedLenSequenceFeature(
    [], dtype=tf.int64),
  F.NUM_SYMBOLS_PER_POST: tf.io.FixedLenSequenceFeature(
    [], dtype=tf.int64),
  F.NUM_POSTS: tf.io.FixedLenFeature([], dtype=tf.int64),
  F.AUTHOR_ID: tf.io.FixedLenFeature([], dtype=tf.int64),
  F.ACTION_TYPE: tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
  F.HOUR: tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
}


FEATURE_TYPE = {
  F.SYMBOLS: tf.int64,
  F.NUM_SYMBOLS_PER_POST: tf.int64,
  F.NUM_POSTS: tf.int64,
  F.AUTHOR_ID: tf.int64,
  F.ACTION_TYPE: tf.int64,
  F.HOUR: tf.int64
}


SEQUENCE_FEATURES = set([
  F.SYMBOLS,
  F.NUM_SYMBOLS_PER_POST,
  F.ACTION_TYPE,
  F.HOUR
])


def get_feature_shape(feature):
  if feature in FEATURE_SHAPE:
    return FEATURE_SHAPE[feature]
  raise ValueError(
    (f"{feature} shape information not defined. "
     f"Add feature shape to `FEATURE_SHAPE` dictionary."))


def get_feature_format(feature):
  if feature in FEATURE_FORMAT:
    return FEATURE_FORMAT[feature]
  raise ValueError(
    (f"{feature} format information not defined. "
     f"Add feature format to `FEATURE_FORMAT` dictionary."))


def get_feature_type(feature):
  if feature in FEATURE_TYPE:
    return FEATURE_TYPE[feature]
  raise ValueError(
    (f"{feature} type information not defined. "
     f"Add feature type to `FEATURE_TYPE` dictionary."))


def is_sequence_feature(feature):
  if isinstance(feature, str):
    feature = F(feature)
  return feature in SEQUENCE_FEATURES


def _parse(features, sep=','):
  if isinstance(features, str):
    feature_set = {F[f] for f in features.split(sep)}
  elif isinstance(features, F):
    feature_set = set([features])
  elif isinstance(features, set) or isinstance(features, list):
    feature_set = set()
    for feature in features:
      if isinstance(feature, F):
        feature_set.add(feature)
      if isinstance(feature, str):
        try:
          feature_set.add(F(feature))
        except KeyError:
          print(f"{feature} not in F")
          raise
  else:
    raise ValueError(features)

  if not feature_set:
    raise ValueError(features)

  return feature_set


def _feature_type(features):
  """ Return feature types as a dict with string keys """
  return {f.value: get_feature_type(f) for f in features}


def _feature_format(features):
  """ Return feature formats as a dict with string keys """
  return {f.value: get_feature_format(f) for f in features}


def _feature_shape(features):
  """ Return feature shapes as a dict with string keys """
  return {f.value: get_feature_shape(f) for f in features}


class FeatureConfig:
  """ A `FeatureConfig` describes the `Features` of a problem
    and should capture sufficient information to serialize
    and deserialize data from protobufs.

  """

  def __init__(self, *, context_features, sequence_features,
               label_feature, padded_length,
               num_symbols, num_action_types):
    """
    Arguments:
      context_features: `str`, `set`, or `list` of features.
      sequence_features: `str`, `set`, or `list` of features.
      label_feature: Feature associated with the label.
      padded_length: Length to which symbols are padded.
      num_symbols: Number of subwords in the vocabulary.
      num_action_types: Number of action types (e.g. subreddits).

    """
    self._context_features = _parse(context_features)
    self._sequence_features = _parse(sequence_features)
    if isinstance(label_feature, str):
      label_feature = F(label_feature)
    self._label_feature = F(label_feature)
    self._padded_length = padded_length
    self._num_symbols = num_symbols
    self._num_action_types = num_action_types

  def to_dict(self):
    return {
      "context_features": [x.value for x in self.context_features],
      "sequence_features": [x.value for x in self.sequence_features],
      "label_feature": self.label_feature.value,
      "padded_length": self.padded_length,
      "num_symbols": self.num_symbols,
      "num_action_types": self.num_action_types
    }

  def save_as_json(self, path):
    with open(path, "w") as write_file:
      setting_as_dict = self.to_dict()
      json.dump(setting_as_dict, write_file, indent=4)

  @classmethod
  def from_json(cls, path):
    with open(path) as fp:
      json_contents = json.load(fp)
      return cls(**json_contents)

  def __str__(self):
    return pformat(self.to_dict())

  @property
  def context_features(self):
    return self._context_features

  @property
  def sequence_features(self):
    return self._sequence_features

  @property
  def label_feature(self):
    return self._label_feature

  @property
  def padded_length(self):
    return self._padded_length

  @property
  def num_symbols(self):
    return self._num_symbols

  @property
  def num_action_types(self):
    return self._num_action_types

  @property
  def features(self):
    return self.context_features | self.sequence_features

  @property
  def length_feature(self):
    return F.NUM_POSTS.value

  @property
  def shape(self):
    label_shape = get_feature_shape(self.label_feature)
    feature_shape_list = []
    for f in self.features:
      if f is F.SYMBOLS:
        feature_shape_list.append((f.value, (None, self.padded_length)))
      else:
        feature_shape_list.append((f.value, get_feature_shape(f)))
    feature_shape_dict = dict(feature_shape_list)
    del feature_shape_dict[self.label_feature.value]
    return feature_shape_dict, label_shape

  @property
  def parse_single_example_fn(self):
    context_format = _feature_format(self.context_features)
    sequence_format = _feature_format(self.sequence_features)

    def data_map_fn(serialized_example):
      features = tf.io.parse_single_sequence_example(
        serialized_example, context_features=context_format,
        sequence_features=sequence_format)

      # Flatten the context and sequence features
      feature_dict = {}
      for f, v in features[0].items():
        feature_dict[f] = v
      for f, v in features[1].items():
        assert f not in feature_dict
        feature_dict[f] = v

      # Unpack serialized symbols, padding documents to fixed length.
      # When `RaggedTensor` become first-class citizens, this may
      # be simplified to just passing along the ragged tensor object
      # directly.
      assert F.SYMBOLS.value in feature_dict
      assert F.NUM_SYMBOLS_PER_POST.value in feature_dict
      values = feature_dict[F.SYMBOLS.value]
      lengths = feature_dict[F.NUM_SYMBOLS_PER_POST.value]
      rt = tf.RaggedTensor.from_row_lengths(values, lengths)
      dt = rt.to_tensor()
      paddings = [[0, 0], [0, self.padded_length-tf.shape(dt)[1]]]
      padded_t = tf.pad(dt, paddings, 'CONSTANT', constant_values=0)
      feature_dict[F.SYMBOLS.value] = padded_t

      labels = feature_dict[self.label_feature.value]
      return feature_dict, labels

    return data_map_fn


def make_sequence_example(context_features, sequence_features):
  def _floats_to_feature_list(values):
    return [
      tf.train.Feature(float_list=tf.train.FloatList(value=value))
      for value in values]

  def _ints_to_feature_list(values):
    ret = []
    for value in values:
      if isinstance(value, int) or isinstance(value, np.int64):
        value = [value]
      ret.append(tf.train.Feature(int64_list=tf.train.Int64List(value=value)))
    return ret

  def _values_to_feature_list(feature, values):
    if get_feature_type(feature) == tf.int64:
      return _ints_to_feature_list(values)
    if get_feature_type(feature) == tf.float32:
      return _floats_to_feature_list(values)
    raise ValueError(feature)

  def _int_to_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  feature = {
    f.value: _int_to_feature(v) for f, v in context_features.items()}

  feature_list = {
    f.value: tf.train.FeatureList(feature=_values_to_feature_list(f, v))
    for f, v in sequence_features.items()}

  for feat in sequence_features.keys():
    assert feat.value in feature_list

  for feat in context_features.keys():
    assert feat.value in feature

  example = tf.train.SequenceExample(
    feature_lists=tf.train.FeatureLists(feature_list=feature_list),
    context=tf.train.Features(feature=feature))

  return example


def tfrecord_dataset(input_file_pattern, config, compression_type=None):
  """
  Arguments
    input_file_pattern: Regex matching input tfrecord files.
    config: `FeatureConfig` instance
    compression_type: (Optional.) A tf.string scalar evaluating to one
      of "" (no compression), "ZLIB", or "GZIP".

  Returns: tf.data.Dataset

  """
  filenames = tf.data.Dataset.list_files(input_file_pattern)
  dataset = tf.data.TFRecordDataset(filenames,
                                    compression_type=compression_type)
  return dataset.map(
    config.parse_single_example_fn,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)


def random_episode_from_tfrecord(dataset, config,
                                 min_episode_length=1,
                                 max_episode_length=16):
  """
  Arguments
    dataset: tf.data.Dataset instance
    config: FeatureConfig class instance
    min_episode_length (1): Minimum episode length.
    max_episode_length (16): Maximum episode length.

  Returns: tf.data.Dataset

  """
  def sample_episode(features, label):
    if min_episode_length == max_episode_length:
      length = min_episode_length
    else:
      raise NotImplementedError
    num_action = features[F.NUM_POSTS.value]
    maxval = num_action - length + 1
    start_index = tf.reshape(
      tf.random.uniform([1], minval=0, maxval=maxval,
                        dtype=tf.dtypes.int64), [])
    end_index = start_index + length
    features[F.NUM_POSTS.value] = length  # length of episode
    for key in config.sequence_features:
      key = key.value
      features[key] = features[key][start_index:end_index]
      #features[key] = tf.reshape(
      #  features[key], [length])

    return features, label

  return dataset.map(
    sample_episode,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)


def print_tfrecords(input_file_pattern, config, compression_type=None,
                    n_to_print=1):
  """ Arguments
    input_file_pattern: Regex matching input tfrecord files.
    config: `FeatureConfig` instance.
    compression_type (None): File compression type
    n_to_print: How many examples to print.

  """
  if not tf.executing_eagerly():
    tf.enable_eager_execution()
  dataset = tfrecord_dataset(
    input_file_pattern, config, compression_type=compression_type)
  for example in dataset.take(n_to_print):
    print(example)


def write_tfrecords_from_generator(tfrecord_path, setting, generator,
                                   shard_size=5000):
  """ Arguments
    tfrecord_path: Prefix for output TFRecord files.
    setting: `FeatureConfig` instance
    generator: Python generator yielding episodes of user actions.
    shard_size (5000): How many examples to store per TFRecord file.

  """
  shard = 0

  # Continue writing examples until we run out of authors, sharding the
  # TFRecords into multiple files with `shard_size` authors per shard.
  while True:
    shard_path = tfrecord_path + '.{:03d}'.format(shard) + '.tf'
    with tf.io.TFRecordWriter(shard_path) as writer:
      logging.info(f"Writing {shard_path}")
      for _ in range(shard_size):
        try:
          episode = next(generator)
          features, label = episode

          context_features = {
            k: features[k.value] for k in setting.context_features}
          sequence_features = {
            k: features[k.value] for k in setting.sequence_features}

          example = make_sequence_example(
            context_features,
            sequence_features)
          writer.write(example.SerializeToString())
        except StopIteration:
          return
      shard += 1
