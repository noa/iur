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


import gzip
import json
import itertools

from glob import glob

from absl import logging
from absl import app
from absl import flags

from aid.features import write_tfrecords_from_generator
from aid.features import is_sequence_feature
from aid.features import FeatureConfig
from aid.features import F

FLAGS = flags.FLAGS

flags.DEFINE_string('json', None, 'File pattern matching input JSON files')
flags.DEFINE_string('tf', None, 'Output prefix for TFRecord files')
flags.DEFINE_string('config', 'config.json',
                    'Path to feature configuration file')
flags.DEFINE_string('label_feature', 'author_id',
                    'Feature to use as label for training.')
flags.DEFINE_integer('shard_size', 5000,
                     'Number of examples per TFRecord file')
flags.DEFINE_integer('max_length', 32, 'Maximum document length')

flags.mark_flags_as_required(['json', 'tf'])


def ragged_features(syms):
  assert syms
  assert isinstance(syms, list)
  assert isinstance(syms[0], list)
  values = []
  lens = []
  for doc in syms:
    doc = doc[:FLAGS.max_length]
    values.extend(doc)
    lens.append(len(doc))
  assert len(values) == sum(lens)
  return {
    F.SYMBOLS.value: values,
    F.NUM_SYMBOLS_PER_POST.value: lens,
    F.NUM_POSTS.value: len(lens)}


def preprocess(features, config):
  updates = []
  for feature, value in features.items():
    if feature == F.SYMBOLS.value:
      updates.append(ragged_features(value))

  for update in updates:
    features.update(update)

  assert F.NUM_POSTS.value in features
  assert F.SYMBOLS.value in features

  return features


def histories_from_json(file_pattern, config):
  for path in glob(file_pattern):
    logging.info(f"Reading {path}")
    if path.endswith('gz'):
      f = gzip.open(path, 'rb')
    else:
      f = open(path, 'r')
    for line in f:
      features = preprocess(json.loads(line), config)
      label = features[FLAGS.label_feature]
      yield (features, label)
    f.close()


def main(argv):
  del argv
  pat = FLAGS.tf + '*.tf'
  if len(glob(pat)):
    logging.info(f"Found existing files matching {pat}; aborting")
    return
  config=FeatureConfig.from_json(FLAGS.config)
  logging.info(f"Reading histories from: {FLAGS.json}")
  generator = histories_from_json(FLAGS.json, config)
  write_tfrecords_from_generator(FLAGS.tf, config, generator,
                                 shard_size=FLAGS.shard_size)


if __name__ == '__main__':
  app.run(main)
