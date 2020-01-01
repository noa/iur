#! /usr/bin/env python

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

"""The input consists of raw data from a Reddit dump and we output
preprocessed data (features) to a JSON file. The JSON file may
then be converted to TFRecords as a separate step.

Note that for simplicity all preprocessing happens in-memory. This
limits the size of datasets that may be preprocessed using this
script.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import pickle
import tempfile
import pandas as pd
import gzip
try:
  import ujson as json
except ImportError:
  print("Install the `ujson` pip package to speed things up.")
  import json
from collections import Counter
from collections import defaultdict

from tqdm import tqdm

import sentencepiece as spm

from absl import logging
from absl import flags
from absl import app

from aid.features import F
from aid.features import FeatureConfig
from aid.features import write_tfrecords_from_generator

args = flags.FLAGS

flags.DEFINE_string('df', None, 'Path to pickled DataFrame')
flags.DEFINE_string('ids', None, 'Path to user history file')
flags.DEFINE_string('output_dir', '.', 'Output directory')
flags.DEFINE_string('model_dir', '.', 'Model directory')
flags.DEFINE_string('json_filename', 'examples.json', 'Output JSON file name.')
flags.DEFINE_string('config', 'reddit.json', 'Experiment configuration')
flags.DEFINE_string('unk_subreddit', '<unk>', 'Name of unknown subreddit')
flags.DEFINE_string('model_prefix', 'model', 'Prefix for subword model files')
flags.DEFINE_string('model_type', 'unigram', 'Model type')
flags.DEFINE_string('subreddit_path', '.', 'Path to subreddit pickle')
flags.DEFINE_float('character_coverage', 1.0, 'Character coverage')
flags.DEFINE_integer('input_sentence_size', 1000000,
                     'Number of sentences used to fit subword model')
flags.DEFINE_integer('pad_id', 0, 'Padding ID')
flags.DEFINE_integer('bos_id', -1, 'BOS ID')
flags.DEFINE_integer('eos_id', 1, 'EOS ID')
flags.DEFINE_integer('unk_id', 2, 'Unk ID')
flags.DEFINE_float('min_ascii_fraction', 0.75,
                   'Filter comments with less than this fraction of ASCII')
flags.DEFINE_integer('min_chars', 1, 'Minimum comment length')
flags.DEFINE_integer('min_subwords', 10, 'Minimum number of subwords')
flags.DEFINE_string('text_key', 'body', 'Column name for text field')
flags.DEFINE_string('subreddit_key', 'subreddit', 'Column name for subreddit')
flags.DEFINE_integer('n_to_print', 1, 'Number of comments to print to console')
flags.DEFINE_string('sample_file_path', None, 'Path to JSON lines file with sample indices')

flags.mark_flags_as_required(['df', 'ids'])


def get_hour_from_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).hour


def keep_comment(text, min_ascii_fraction=0.75, min_length=1):
  """ For purposes of vocabulary creation ignore non-ascii documents """
  len_total = len(text)
  if len_total < min_length:
    return False
  len_ascii = sum(c.isalpha() for c in text)
  frac_ascii = float(len_ascii) / float(len_total)
  if frac_ascii < min_ascii_fraction:
    return False
  return True


def fit_subword_vocabulary(df):
  """ Fit subword vocabulary, see https://arxiv.org/abs/1808.06226"""
  config = FeatureConfig.from_json(args.config)
  
  model_prefix = os.path.join(
    args.model_dir,
    f"{config.num_symbols}_{args.model_type}")

  if os.path.exists(model_prefix + ".model"):
    logging.info(f"Subword model {model_prefix}.model already exists; using that")
    return
  else:
    logging.info(f"Fitting new subword vocabulary of size {config.num_symbols}")
  
  with tempfile.NamedTemporaryFile(mode='w+t') as temp:
    index = 0
    logging.info(f"Writing text content to {temp.name}")
    for key, row in df.iterrows():
      text = row[args.text_key]
      if keep_comment(
          text,
          min_ascii_fraction=args.min_ascii_fraction,
          min_length=args.min_chars):
        text = text.replace('\n', ' ')
        temp.write(text + '\n')
      if index > args.input_sentence_size:
        break
      index += 1

    temp.flush()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Creating Vocabulary with SentencePiece...")

    trainer_args = [
      f'--input={temp.name}',
      f'--model_prefix={model_prefix}',
      f'--vocab_size={config.num_symbols}',
      f'--model_type={args.model_type}',
      f'--character_coverage={args.character_coverage}',
      f'--input_sentence_size={args.input_sentence_size}',
      f'--shuffle_input_sentence=true',
      f'--pad_id={args.pad_id}',
      f'--eos_id={args.eos_id}',
      f'--bos_id={args.bos_id}',
      f'--unk_id={args.unk_id}'
    ]
    logging.info("Fitting SentencePiece model")
    spm.SentencePieceTrainer.Train(' '.join(trainer_args))


def fit_subreddit_vocab(df):
  config = FeatureConfig.from_json(args.config)
  subreddit_map_path = os.path.join(
    args.model_dir,
    f'{config.num_action_types}_subreddits.pickle')
  if os.path.exists(subreddit_map_path):
    logging.info(f"Using existing subreddit map: {subreddit_map_path}")
    return
  logging.info("Creating SubReddit map")
  logging.info("Obtaining unique subreddits")
  subreddits = df[args.subreddit_key]
  counts = Counter(subreddits)
  most_common = counts.most_common()
  logging.info("Most common subreddits:")
  for sr, count in most_common[:10]:
    logging.info(f"  {sr} {count}")
  output_map = {}
  for i, sr in enumerate([x for x, _ in most_common[:config.num_action_types-1]]):
    output_map[sr] = i
  assert args.unk_subreddit not in output_map
  output_map[args.unk_subreddit] = len(output_map)
  assert len(output_map) == config.num_action_types
  logging.info(f"Kept {len(output_map)} subreddits")
  logging.info(f"Saving subreddit map to: {subreddit_map_path}")
  with open(subreddit_map_path, 'wb') as f:
    pickle.dump(output_map, f)


def fit_author_vocab(df):
  """ Keep track of author IDs """
  author_map_path = os.path.join(
    args.output_dir, 'authors.pickle')
  if os.path.exists(author_map_path):
    logging.info(f"Using existing author map: {author_map_path}")
    return
  author_map = {}
  for i, a in enumerate(set(df['author'])):
    author_map[a] = i
  logging.info(f"{len(author_map)} authors")
  with open(author_map_path, 'wb') as f:
    pickle.dump(author_map, f)


def print_examples(df, print_if_less_than=15):
  config = FeatureConfig.from_json(args.config)
  model_path = os.path.join(
    args.model_dir,
    f"{config.num_symbols}_{args.model_type}.model")
  sp = spm.SentencePieceProcessor()
  sp.Load(model_path)
  logging.info(f"Piece size: {sp.GetPieceSize()}")
  n_printed = 0
  for index, row in df.iterrows():
    raw_text = row[args.text_key]
    if len(raw_text.split()) < print_if_less_than:
      logging.info(raw_text)
      pieces = sp.EncodeAsPieces(raw_text)
      logging.info(" ".join(
        [f"{piece}:{sp.PieceToId(piece)}" for piece in pieces]))
      n_printed += 1
    if n_printed > args.n_to_print:
      break


def maybe_load_sample_file():
  if args.sample_file_path is None:
    return None
  samples = {}
  logging.info(f"Loading sample file: {args.sample_file_path}")
  with open(args.sample_file_path) as fh:
    for line in fh:
      sample = json.loads(line)
      samples[sample['author']] = sample
  assert samples
  return samples


def write_json(df):
  json_path = os.path.join(args.output_dir, args.json_filename)
  if os.path.exists(json_path):
    logging.info(f'{json_path} exists; delete to remake')
    return
  samples = maybe_load_sample_file()
  config = FeatureConfig.from_json(args.config)
  model_path = os.path.join(
    args.model_dir,
    f"{config.num_symbols}_{args.model_type}.model")
  sp = spm.SentencePieceProcessor()
  sp.Load(model_path)
  subreddit_map_path = os.path.join(
    args.model_dir,
    f'{config.num_action_types}_subreddits.pickle')
  with open(subreddit_map_path, 'rb') as fh:
    logging.info(f"Loading subreddit map: {subreddit_map_path}")
    subreddit_map = pickle.load(fh)
  author_map_path = os.path.join(
    args.output_dir, 'authors.pickle')
  with open(author_map_path, 'rb') as fh:
    logging.info(f"Loading author map: {author_map_path}")
    author_map = pickle.load(fh)
  logging.info(f"Writing preprocessed data to: {json_path}")
  N = len(open(args.ids).readlines())
  with open(json_path, 'w') as fout, \
       open(args.ids, 'r') as ids_file:
    for line in tqdm(ids_file, total=N):
      comment_ids = line.split()
      first_id = comment_ids[0]
      author = df.loc[first_id]['author']
      if samples:
        if author not in samples:
          continue
        sample = samples[author]
        assert len(comment_ids) == sample['num_actions_total']
        start_index = sample['start_index']
        length = sample['episode_length']
        comment_ids = comment_ids[start_index:start_index+length]
        assert len(comment_ids) == length
      history = {
        F.SYMBOLS.value: [],
        F.HOUR.value: [],
        F.ACTION_TYPE.value: [],
        F.AUTHOR_ID.value: author_map[author]
      }
      for id_ in comment_ids:
        comment = df.loc[id_]
        history[F.SYMBOLS.value].append(sp.EncodeAsIds(comment['body']))
        history[F.HOUR.value].append(
          get_hour_from_timestamp(comment['created_utc']))
        subreddit_index = subreddit_map[args.unk_subreddit]
        if comment['subreddit'] in subreddit_map:
          subreddit_index = subreddit_map[comment['subreddit']]
        history[F.ACTION_TYPE.value].append(subreddit_index)

      fout.write(json.dumps(history) + '\n')


def main(argv):
  logging.info(f"Output directory: {args.output_dir}")
  os.makedirs(args.output_dir, exist_ok=True)
  df = pd.read_pickle(args.df)
  fit_subword_vocabulary(df)
  print_examples(df)
  fit_subreddit_vocab(df)
  fit_author_vocab(df)
  write_json(df)


if __name__ == "__main__":
  app.run(main)
