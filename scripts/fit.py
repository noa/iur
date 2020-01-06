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

import os
import pathlib
from functools import partial

from tqdm import tqdm

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense

from aid.features import F
from aid.features import FeatureConfig
from aid.models import IurNet
from aid.losses import AngularMargin
from aid.evaluation import ranking

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'fit', ['fit', 'rank'],
                  'Use `fit` to train model or `rank` to evaluate it')
flags.DEFINE_enum('framework', 'fit', ['fit', 'custom'], 'Framework')
flags.DEFINE_string('expt_dir', None, 'Experiment directory')
flags.DEFINE_string('results_filename', 'results.txt', 'Written as expt_dir/results_filename')
flags.DEFINE_integer('fit_verbosity', 1, 'Use `1` for local jobs and `2` for grid jobs')
flags.DEFINE_integer('save_freq', 50000, 'Number of steps between checkpoints')
flags.DEFINE_string('train_tfrecord_path', None, 'Path to train TFRecords')
flags.DEFINE_string('valid_tfrecord_path', None, 'Path to validation TFRecords')
flags.DEFINE_string('monitor', 'MRR', 'Metric to monitor')
flags.DEFINE_integer('num_cpu', 4, 'Number of CPU processes')
flags.DEFINE_integer('num_classes', None, 'This is usually the number of authors')
flags.DEFINE_integer('episode_len', 16, 'Episode length')
flags.DEFINE_integer('samples_per_class', 4, 'Number of samples for each author history')
flags.DEFINE_integer('embedding_dim', 512, 'Embedding dimensionality')
flags.DEFINE_integer('num_epochs', 20, 'Number of training epochs')
flags.DEFINE_integer('steps_per_epoch', 10000, 'Steps per epoch')
flags.DEFINE_integer('valid_steps', 100, 'Number of validation steps')
flags.DEFINE_integer('num_queries', 25000, 'Number of ranking queries')
flags.DEFINE_string('optimizer', 'sgd', 'Optimizer')
flags.DEFINE_float('learning_rate', 0.05, 'Learning rates')
flags.DEFINE_integer('first_decay_steps', 10000, 'First decay steps for restarts')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay scale')
flags.DEFINE_string('schedule', 'piecewise', 'Learning rate schedule')
flags.DEFINE_float('momentum', 0.9, 'Momentum')
flags.DEFINE_boolean('nesterov', False, 'Use Nesterov momentum')
flags.DEFINE_float('grad_norm_clip', 1., 'Clip the norm of gradients to this value')
flags.DEFINE_integer('batch_size', 128, 'Mini-batch size for SGD')
flags.DEFINE_string('features', 'syms+action_type+hour', 'Features')
flags.DEFINE_enum('loss', 'am', ['am', 'sm'], 'Surrogate metric learning objective')
flags.DEFINE_float('scale', 64., 'Angular margin loss scale')
flags.DEFINE_float('margin', 0.5, 'Angular margin')
flags.DEFINE_string('final_activation', 'relu', 'Final activation')
flags.DEFINE_integer('num_filters', 256, 'Number of filters')
flags.DEFINE_integer('min_filter_width', 2, 'Smallest filter size')
flags.DEFINE_integer('max_filter_width', 5, 'Largest filter size')
flags.DEFINE_string('filter_activation', 'relu', 'Nonlinearity after feature')
flags.DEFINE_integer('num_layers', 2, 'Number of Transformer encoder layers')
flags.DEFINE_integer('d_model', 256, 'Transformer layer size')
flags.DEFINE_integer('num_heads', 4, 'Number of attention heads')
flags.DEFINE_integer('dff', 256, 'Size of feedforward layers in encoder')
flags.DEFINE_integer('log_steps', 1000, 'Steps at which to log progress')
flags.DEFINE_float('dropout_rate', 0.1, 'Rate of dropout')
flags.DEFINE_integer('subword_embed_dim', 512, 'Size of subword embedding')
flags.DEFINE_integer('action_embed_dim', 512, 'Size of action type embedding')
flags.DEFINE_string('expt_config_path', 'results', 'Model and log directory')
flags.DEFINE_integer('num_parallel_readers', 4, 'Number of files to read in parallel')
flags.DEFINE_integer('shuffle_seed', 42, 'Seed for data shuffle')
flags.DEFINE_integer('shuffle_buffer_size', 2**13, 'Size of shufle buffer')
flags.DEFINE_string('distance', 'cosine', 'How to compare embeddings')


def get_flagfile():
  return os.path.join(FLAGS.expt_dir, 'flags.cfg')


def get_ckpt_dir():
  return os.path.join(FLAGS.expt_dir, 'checkpoints')


def get_export_dir():
  return os.path.join(FLAGS.expt_dir, 'embedding')


def build_output_projection():
  if FLAGS.loss == 'sm':
    return Dense(FLAGS.num_classes, use_bias=False)
  elif FLAGS.loss == 'am':
    return AngularMargin(FLAGS.num_classes, scale=FLAGS.scale, margin=FLAGS.margin)
  else:
    raise ValueError(FLAGS.loss)


def build_episode_embedding(config):
  return IurNet(num_symbols=config.num_symbols,
                num_action_types=config.num_action_types,
                padded_length=config.padded_length,
                episode_len=FLAGS.episode_len,
                features=FLAGS.features,
                num_layers=FLAGS.num_layers,
                d_model=FLAGS.d_model,
                num_heads=FLAGS.num_heads,
                dff=FLAGS.dff,
                dropout_rate=FLAGS.dropout_rate,
                subword_embed_dim=FLAGS.subword_embed_dim,
                action_embed_dim=FLAGS.action_embed_dim,
                filter_activation=FLAGS.filter_activation,
                final_activation=FLAGS.final_activation,
                num_filters=FLAGS.num_filters,
                min_filter_width=FLAGS.min_filter_width,
                max_filter_width=FLAGS.max_filter_width)


class Model(tf.keras.Model):
  def __init__(self, config, **kwargs):
    super(Model, self).__init__(**kwargs)
    self.embedding = build_episode_embedding(config)
    self.projection = build_output_projection()

  @tf.function
  def call(self, inputs, training=False):
    v = self.embedding(inputs, training=training)
    if 'am' in FLAGS.loss:
      labels = tf.reshape(inputs[F.AUTHOR_ID.value], [-1])
      logits = self.projection([v, labels])
    else:
      logits = self.projection(v)
    return logits


def sample_random_episode(dataset, config,
                          min_episode_length=16,
                          max_episode_length=16,
                          repeat=1):
  assert repeat > 0
  
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
    new_features = {}
    for key in config.sequence_features:
      new_features[key.value] = features[key.value][start_index:end_index]
    for key in config.context_features:
      new_features[key.value] = features[key.value]
    new_features[F.NUM_POSTS.value] = length  # length of episode
    return new_features, label

  if repeat < 2:
    return dataset.map(
      sample_episode,
      num_parallel_calls=1)
  else:
    def repeat_sample_episode(features, label):
      xs = {}
      ys = []
      for _ in range(repeat):
        x, y = sample_episode(features, label)
        for k, v in x.items():
          if k in xs:
            xs[k].append(v)
          else:
            xs[k] = [v]
        ys.append(y)
      for k, v in xs.items():
        xs[k] = tf.stack(v, axis=0)
      return xs, tf.stack(ys, axis=0)
    ds = dataset.map(repeat_sample_episode, num_parallel_calls=1)
    return ds.unbatch()


def build_dataset(file_pattern, config, num_epochs=None, shuffle=True,
                  random_episode=True, take=None, samples_per_class=1):
  filenames = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
  if shuffle:
    ds = filenames.interleave(
      tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers)
    ds = ds.shuffle(FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)
  else:
    ds = tf.data.TFRecordDataset(filenames)
  if take and take > 0:
    ds = ds.take(take)
  ds = ds.repeat(num_epochs)
  ds = ds.map(config.parse_single_example_fn,
              num_parallel_calls=FLAGS.num_cpu)
  if random_episode:
    ds = sample_random_episode(
      ds, config,
      min_episode_length=FLAGS.episode_len,
      max_episode_length=FLAGS.episode_len,
      repeat=samples_per_class)
  ds = ds.batch(FLAGS.batch_size)
  ds = ds.prefetch(1)
  return ds


def embedding_and_labels(model, file_pattern, config, take=None,
                         random_episode=True):
  dataset = build_dataset(file_pattern, config, num_epochs=1,
                          shuffle=False, take=take)
  embeddings = []
  authors = []
  for batch in dataset:
    features, labels = batch
    if FLAGS.framework == 'custom' and FLAGS.mode == 'fit':
      e = model(features, training=False)
    else:
      e = model.predict(features, batch_size=FLAGS.batch_size)
    embeddings.append(e)
    authors.append(labels)
  return np.vstack(embeddings), np.concatenate(authors)


def rank(model, config, checkpoint=None):
  logging.info(f"Computing queries from {FLAGS.train_tfrecord_path}")
  query_vectors, query_labels = embedding_and_labels(
    model, FLAGS.train_tfrecord_path, config, take=FLAGS.num_queries)
  logging.info(f"{query_vectors.shape[0]} queries")
  logging.info(f"Computing targets from {FLAGS.valid_tfrecord_path}")
  target_vectors, target_labels = embedding_and_labels(
    model, FLAGS.valid_tfrecord_path, config)
  logging.info(f"{target_vectors.shape[0]} targets")
  logging.info(f"Performing ranking evaluation")
  metrics = ranking(
    query_vectors, query_labels, target_vectors, target_labels, metric=FLAGS.distance,
    n_jobs=FLAGS.num_cpu)
  return metrics


def get_lr_schedule():
  if FLAGS.schedule == 'constant':
    return PiecewiseConstantDecay([0], [FLAGS.learning_rate, FLAGS.learning_rate])
  elif FLAGS.schedule == 'piecewise':
    steps = [80000, 140000, 200000]
    lr = FLAGS.learning_rate
    return PiecewiseConstantDecay(
      steps,
      [lr, lr / 10., lr / 100., lr / 1000.])
    return learning_rate
  elif FLAGS.schedule == 'cosine_decay_restarts':
    return tf.keras.experimental.CosineDecayRestarts(
      FLAGS.learning_rate, FLAGS.first_decay_steps)
  else:
    raise ValueError(FLAGS.schedule)


def get_optimizer():
  if FLAGS.optimizer == 'sgd':
    optimizer = partial(SGD, momentum=FLAGS.momentum,
                        nesterov=FLAGS.nesterov)
  elif FLAGS.optimizer == 'adam':
    optimizer = partial(Adam)
  elif FLAGS.optimizer == 'adamw':
    import tensorflow_addons as tfa
    optimizer = partial(tfa.optimizers.AdamW)
  else:
    raise ValueError(FLAGS.optimizer)
  return optimizer


def build_optimizer(step=None):
  schedule = get_lr_schedule()
  optimizer = get_optimizer()

  if FLAGS.optimizer in ['adamw']:
    if FLAGS.schedule == 'constant':
      return optimizer(learning_rate=schedule,
                       weight_decay=FLAGS.weight_decay)
    else:
      raise NotImplementedError
  else:
    return optimizer(learning_rate=schedule)


def load_embedding(config, export_path):
  embedding = build_episode_embedding(config)

  # This initializes the variables associated with the embedding
  ds = build_dataset(FLAGS.train_tfrecord_path, config, shuffle=False)
  for x, _ in ds.take(1):
    e = embedding(x)
    logging.info(f"Output shape: {e.shape}")

  # Load model weights
  embedding.load_weights(export_path)

  return embedding


def sanity_check_saved_model(model, config, export_path):
  logging.info(f"Loading weights from {export_path}...")
  new_model = load_embedding(config, export_path)
  logging.info(f"Weights loaded")
  ds = build_dataset(FLAGS.valid_tfrecord_path, config, shuffle=False)
  for x, _ in ds.take(1):
    predictions = model(x, training=False)
    new_predictions = new_model.predict(x)
  np.testing.assert_allclose(predictions, new_predictions, rtol=1e-4, atol=1e-4)
  logging.info("Serialization worked!")


def custom_fit(model, config):
  total_num_steps = FLAGS.num_epochs * FLAGS.steps_per_epoch
  compute_loss = SparseCategoricalCrossentropy(from_logits=True)
  
  @tf.function
  def train_one_step(model, optimizer, inputs, targets):
    # By default `GradientTape` will automatically watch any trainable
    # variables that are accessed inside the context
    with tf.GradientTape() as tape:
      logits = model(inputs, training=True)
      loss_value = compute_loss(y_true=targets, y_pred=logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    if FLAGS.grad_norm_clip:
      grads, _ = tf.clip_by_global_norm(grads, FLAGS.grad_norm_clip)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

  def train(model):
    train_ds = build_dataset(FLAGS.train_tfrecord_path, config,
                             samples_per_class=FLAGS.samples_per_class,
                             shuffle=True, random_episode=True)
    best_score = -1
    step = 0
    epoch = 0
    optimizer = build_optimizer()
    loss = 0.0
    disable_progress = FLAGS.fit_verbosity == 2
    for x, y in tqdm(train_ds.take(total_num_steps),
                     total=total_num_steps, disable=disable_progress):
      step += 1
      loss = train_one_step(model, optimizer, x, y)
      
      if step % FLAGS.log_steps == 0:
        logging.info((f"[Step {step} of {total_num_steps}] loss {loss.numpy():.2f}, "
                      f"lr {optimizer.lr(step).numpy():.3f}, "
                      f"best {FLAGS.monitor} {best_score:.3f}"))

      if step > 0 and step % FLAGS.steps_per_epoch == 0:
        epoch += 1
        logging.info(f"[Epoch {epoch} of {FLAGS.num_epochs}]")
        metrics = rank(model.embedding, config)

        logging.info(f"Ranking metrics:")
        for name in sorted(metrics.keys()):
          score = metrics[name]
          logging.info(f"{name} {score:.3f}")

        if metrics[FLAGS.monitor] > best_score:
          best_score = metrics[FLAGS.monitor]
          export_path = get_export_dir()
          logging.info(f"Exporting best model to {export_path}")
          model.embedding.save_weights(export_path, save_format='tf')
          if epoch == 1:
            logging.info("Checking weight serialization...")
            sanity_check_saved_model(model.embedding, config, export_path)

  logging.info("Training!")
  train(model)


def fit(config):
  flags.mark_flags_as_required(['num_classes',
                                'train_tfrecord_path',
                                'valid_tfrecord_path',
                                'expt_dir'])
  total_num_steps = FLAGS.num_epochs * FLAGS.steps_per_epoch
  logging.info(f"Training for {total_num_steps} steps total")
  logdir = FLAGS.expt_dir
  checkpoint_file = "weights.{epoch:02d}.ckpt"
  checkpoint_path = get_ckpt_dir() + "/" + checkpoint_file
  checkpoint_dir = os.path.dirname(checkpoint_path)
  logging.info(f"Checkpoint directory: {checkpoint_dir}")
  model = Model(config)

  if FLAGS.framework == 'fit':
    optimizer = build_optimizer()
    assert optimizer
    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)
    callbacks = [
      tf.keras.callbacks.TensorBoard(logdir),
      tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                         save_weights_only=True,
                                         save_best_only=False,
                                         save_freq=FLAGS.save_freq,
                                         verbose=1),
    ]
    train_ds = build_dataset(FLAGS.train_tfrecord_path, config,
                             samples_per_class=FLAGS.samples_per_class)
    valid_ds = build_dataset(FLAGS.valid_tfrecord_path, config)
    logging.info(f"Training, logging results to {FLAGS.expt_dir}")
    model.fit(
      train_ds,
      epochs=FLAGS.num_epochs,
      steps_per_epoch=FLAGS.steps_per_epoch,
      callbacks=callbacks,
      verbose=FLAGS.fit_verbosity,
      validation_data=valid_ds,
      validation_steps=FLAGS.valid_steps)
    export_path = get_export_dir()
    logging.info(f"Exporting embedding weights to {export_path}")
    model.embedding.save_weights(export_path, save_format='tf')
    
  if FLAGS.framework == 'custom':
    custom_fit(model, config)


def handle_flags(argv):
    key_flags = FLAGS.get_key_flags_for_module(argv[0])
    s = '\n'.join(f.serialize() for f in key_flags)
    logging.info(f'fit.py flags:\n{s}')
    if FLAGS.mode == 'fit':
      flagfile = get_flagfile()
      with open(flagfile, 'w') as fh:
        logging.info(f"Writing flags to {flagfile}")
        fh.write(s)


def main(argv):
  # Set global environment flags
  logging.info(f"Limiting to {FLAGS.num_cpu} CPU")
  tf.config.threading.set_inter_op_parallelism_threads(FLAGS.num_cpu)
  tf.config.threading.set_intra_op_parallelism_threads(FLAGS.num_cpu)

  config = FeatureConfig.from_json(FLAGS.expt_config_path)
  logging.info(config)

  # Sanity check arguments
  if 'am' in FLAGS.loss and FLAGS.distance != 'cosine':
    raise ValueError("Use `cosine` distance for angular margin loss")

  if 'sm' in FLAGS.loss and FLAGS.distance != 'euclidean':
    raise ValueError("Use `euclidean` distance for softmax loss")

  if FLAGS.mode == 'fit':
    handle_flags(argv)
    fit(config)

  if FLAGS.mode == 'rank':
    flags.mark_flags_as_required(['train_tfrecord_path', 'expt_dir', 'valid_tfrecord_path'])
    export_dir = get_export_dir()
    logging.info(f"Loading embedding from {export_dir}")
    embedding = load_embedding(config, export_dir)
    metrics = rank(embedding, config)
    results_file = os.path.join(FLAGS.expt_dir, FLAGS.results_filename)
    logging.info(f"Results will be written to {results_file}")
    results = "\n".join([f"{k} {v}" for k, v in metrics.items()])
    logging.info(results)
    with open(results_file, 'w') as fh:
      fh.write(results)


if __name__ == '__main__':
  app.run(main)
