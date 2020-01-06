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

import numpy as np
from sklearn.metrics import pairwise_distances
from six.moves import xrange


def ranking(query_vectors, query_authors, target_vectors,
            target_authors, metric='cosine', distances=None,
            n_jobs=None):
    """
    Arguments:
      query_vectors: Numpy matrix of size (N, V) for N authors and V features.
      query_authors: Numpy array of size (N) containing author IDs.
      target_vectors: Numpy matrix of size (M, V) for M authors and V features.
      target_authors: Numpy array of size (M) containing author IDs.
      metric: Metric used to compare different embeddings.
      pairwise_distances: If `metric` is "precomputed"
      n_jobs: (Optional) Number of threads to use to compute pairwise distances.

    Returns:
      A dictionary where the keys are the names of metrics
      and the values contain the corresponding computed 
      metrics.

    """
    if isinstance(query_authors, list):
        query_authors = np.array(query_authors)
    if isinstance(target_authors, list):
        target_authors = np.array(target_authors)
    
    # If Y is not None, then D_{i, j} is the distance between the ith
    # array from X and the jth array from Y.
    if metric == 'precomputed':
        D = distances
    else:
        D = pairwise_distances(query_vectors, Y=target_vectors, metric=metric, n_jobs=n_jobs)

    # Compute rank
    num_queries = query_authors.shape[0]
    ranks = np.zeros((num_queries), dtype=np.int32)
    reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)
    
    for query_index in xrange(num_queries):
        author = query_authors[query_index]
        distances = D[query_index]
        indices_in_sorted_order = np.argsort(distances)  # *increasing*
        labels_in_sorted_order = target_authors[indices_in_sorted_order]
        rank = np.where(labels_in_sorted_order == author)
        assert len(rank) == 1
        rank = rank[0] + 1.
        ranks[query_index] = rank
        reciprocal_rank = 1.0 / float(rank)
        reciprocal_ranks[query_index] = (reciprocal_rank)

    return {
        'MRR': np.mean(reciprocal_ranks),
        'MR': np.mean(ranks),
        'min_rank': np.min(ranks),
        'max_rank': np.max(ranks),
        'median_rank': np.median(ranks),
        'recall@1': np.sum(np.less_equal(ranks,1)) / np.float32(num_queries),
        'recall@2': np.sum(np.less_equal(ranks,2)) / np.float32(num_queries),
        'recall@4': np.sum(np.less_equal(ranks,4)) / np.float32(num_queries),
        'recall@8': np.sum(np.less_equal(ranks,8)) / np.float32(num_queries),
        'recall@16': np.sum(np.less_equal(ranks,16)) / np.float32(num_queries),
        'recall@32': np.sum(np.less_equal(ranks,32)) / np.float32(num_queries),
        'recall@64': np.sum(np.less_equal(ranks,64)) / np.float32(num_queries),
        'num_queries': num_queries,
        'num_targets': target_authors.shape[0]
    }
