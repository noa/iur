#! /usr/bin/env bash

set -e
set -f
set -u

# Set OUTPUT_DIR, REPO_DIR, and DATA_DIR, where:
#
# OUTPUT_DIR: where to write the preprocessed data
# REPO_DIR: path to where the repository was downloaded
# DATA_DIR: path to where you unpacked https://cs.jhu.edu/~noa/data/reddit.tar.gz

DOWNLOAD="python ${SCRIPT_DIR}/bigquery_download_comments.py"
TRAIN_TABLE="fh-bigquery.reddit_comments.2016_08"
TEST_TABLE="fh-bigquery.reddit_comments.2016_09"
NUM_SUBWORDS=65536
N_SUBREDDIT=2048
MAX_LENGTH=32
SHARD_SIZE=5000
SCRIPT_DIR=${REPO_DIR}/scripts
CONFIG=${REPO_DIR}/data/reddit/config.json

# Train / validation
${DOWNLOAD} --ids ${DATA_DIR}/train_query.ids --output ${OUTPUT_DIR}/train_query.pickle --table ${TRAIN_TABLE}
${DOWNLOAD} --ids ${DATA_DIR}/train_target.ids --output ${OUTPUT_DIR}/train_target.pickle --table ${TRAIN_TABLE}

# Test
${DOWNLOAD} --ids ${DATA_DIR}/test_query.ids --output ${OUTPUT_DIR}/test_query.pickle --table ${TEST_TABLE}
${DOWNLOAD} --ids ${DATA_DIR}/test_target.ids --output ${OUTPUT_DIR}/test_target.pickle --table ${TEST_TABLE}

python ${SCRIPT_DIR}/reddit_preprocess.py \
       --df ${OUTPUT_DIR}/train_query.pickle \
       --json_filename queries.jsonl \
       --ids ${DATA_DIR}/train_query.ids \
       --config ${CONFIG} \
       --output_dir ${OUTPUT_DIR}/train \
       --model_dir ${OUTPUT_DIR}/train

python ${SCRIPT_DIR}/reddit_preprocess.py \
       --df ${OUTPUT_DIR}/train_target.pickle \
       --json_filename targets.jsonl \
       --ids ${DATA_DIR}/train_target.ids \
       --config ${CONFIG} \
       --output_dir ${OUTPUT_DIR}/train \
       --model_dir ${OUTPUT_DIR}/train

python ${SCRIPT_DIR}/reddit_preprocess.py \
       --df ${OUTPUT_DIR}/test_query.pickle \
       --json_filename queries.jsonl \
       --ids ${DATA_DIR}/test_query.ids \
       --sample_file_path ${DATA_DIR}/test_query_sample.jsonl \
       --config ${CONFIG} \
       --output_dir ${OUTPUT_DIR}/test \
       --model_dir ${OUTPUT_DIR}/train

python ${SCRIPT_DIR}/reddit_preprocess.py \
       --df ${OUTPUT_DIR}/test_target.pickle \
       --json_filename targets.jsonl \
       --ids ${DATA_DIR}/test_target.ids \
       --sample_file_path ${DATA_DIR}/test_target_sample.jsonl \
       --config ${CONFIG} \
       --output_dir ${OUTPUT_DIR}/test \
       --model_dir ${OUTPUT_DIR}/train

python ${SCRIPT_DIR}/json2tf.py \
       --json ${OUTPUT_DIR}/train/queries.jsonl \
       --tf ${OUTPUT_DIR}/train/queries \
       --config ${CONFIG} \
       --shard_size ${SHARD_SIZE} \
       --max_length ${MAX_LENGTH}

python ${SCRIPT_DIR}/json2tf.py \
       --json ${OUTPUT_DIR}/train/targets.jsonl \
       --tf ${OUTPUT_DIR}/train/targets \
       --config ${CONFIG} \
       --shard_size ${SHARD_SIZE} \
       --max_length ${MAX_LENGTH}

python ${SCRIPT_DIR}/json2tf.py \
       --json ${OUTPUT_DIR}/test/queries.jsonl \
       --tf ${OUTPUT_DIR}/test/queries \
       --config ${CONFIG} \
       --shard_size ${SHARD_SIZE} \
       --max_length ${MAX_LENGTH}

python ${SCRIPT_DIR}/json2tf.py \
       --json ${OUTPUT_DIR}/test/targets.jsonl \
       --tf ${OUTPUT_DIR}/test/targets \
       --config ${CONFIG} \
       --shard_size ${SHARD_SIZE} \
       --max_length ${MAX_LENGTH}

# eof
