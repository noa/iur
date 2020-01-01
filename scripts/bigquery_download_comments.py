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

import os
import sys
import time
import json
import pathlib

from collections import namedtuple

from google.cloud import bigquery

from absl import app
from absl import logging
from absl import flags

import tensorflow as tf
import pandas as pd

flags.DEFINE_string('ids', None,
                    """Path to file containing a list of Reddit comment IDs,
                    where each line consists of one users' history.""")
flags.DEFINE_string('output', None, "Output path for comments")
flags.DEFINE_list('columns',
                  ['body', 'id', 'author', 'subreddit', 'created_utc'],
                  "Columns to keep from the original data")
flags.DEFINE_string('table', 'fh-bigquery.reddit_comments.2016_08',
                    """BigQuery table. You must have read access via your 
                    GCP account.""")
flags.DEFINE_string('location', 'US', 'Must match dataset location')
flags.DEFINE_float('wait', 10.0,
                   'Interval between checks on job status.')
flags.DEFINE_integer('max_job_size', 100000, 'Number of comments per job')
flags.DEFINE_integer('max_jobs', sys.maxsize, 'Maximum number of jobs')

flags.mark_flags_as_required(['ids', 'output'])

FLAGS = flags.FLAGS

Job = namedtuple('Job', 'info start_index end_index df')


def submit_job(client, ids, start_index, end_index, id_key='id'):
  logging.info(f"Submitting query job for {len(ids)} comment IDs")
  id_list = ",".join([f'"{id_}"' for id_ in ids])
  sql = (
    f"SELECT * FROM `{FLAGS.table}` "
    f"WHERE {id_key} IN ({id_list}) ")
  job_config = bigquery.QueryJobConfig()
  job_config.priority = bigquery.QueryPriority.BATCH
  query_job = client.query(sql, location=FLAGS.location,
                           job_config=job_config)
  return Job(query_job, start_index, end_index, None)


def big_query():
  client = bigquery.Client()
  jobs = []
  with tf.io.gfile.GFile(FLAGS.ids) as f:
    logging.info(f"Reading comment IDs from {FLAGS.ids}")
    job_ids = []
    start_index = 0
    total_num_comments = 0
    for index, line in enumerate(f):
      ids = line.split()
      total_num_comments += len(ids)
      job_ids += ids
      if len(job_ids) > FLAGS.max_job_size:
        jobs.append(submit_job(client, job_ids, start_index, index))
        start_index = index + 1
        job_ids.clear()
      if len(jobs) >= FLAGS.max_jobs:
        logging.info(f"Reached maximum number of jobs at index: {index}")
        break
    logging.info(f"Last index reached: {index}")
    num_class = index + 1
    if job_ids:
      jobs.append(submit_job(client, job_ids, start_index, num_class))
      job_ids.clear()

  logging.info(f"{len(jobs)} jobs downloading {total_num_comments} comments...")

  dataframes = [None] * len(jobs)
  n_jobs = len(jobs)
  while True:
    time.sleep(FLAGS.wait)
    logging.info("Querying job status...")
    n_done = 0
    for job_index in range(n_jobs):
      job = jobs[job_index]
      if dataframes[job_index] is None:
        query_job = client.get_job(
          job.info.job_id, location=FLAGS.location)
        if query_job.state == 'DONE':
          df = query_job.result().to_dataframe(
            progress_bar_type='tqdm')[FLAGS.columns]
          df = df.set_index('id')
          dataframes[job_index] = df
        else:
          pass  # wait
      else:
        n_done += 1
    logging.info(f"{n_done} of {n_jobs} jobs done")
    if n_done == n_jobs:
      break

  data = pd.concat(dataframes)
  logging.info(f"Downloaded {len(data)} comments")
  assert len(data) == total_num_comments
  return data


def main(argv):
  del argv
  if os.path.exists(FLAGS.output):
    logging.info(f'{FLAGS.output} exists; delete to download')
    return
  else:
    logging.info(f'{FLAGS.output} does not exist; downloading...')
  logging.info(f'Querying BigQuery table {FLAGS.table} for comments')
  pathlib.Path(FLAGS.output).parent.mkdir(parents=True, exist_ok=True)
  data = big_query()
  data.to_pickle(FLAGS.output)


if __name__ == '__main__':
  app.run(main)
