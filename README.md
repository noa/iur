This is the official repository for the
[EMNLP 2019 paper](https://www.aclweb.org/anthology/D19-1178.pdf)
"Learning Invariant Representations of Social Media Users," by Nicholas Andrews
and Marcus Bishop.  If you use the code or datasets released here in
published work, the appropriate
[citation](https://www.aclweb.org/anthology/D19-1178.bib) is:

```
@inproceedings{andrews-bishop-2019-learning,
    title = "Learning Invariant Representations of Social Media Users",
    author = "Andrews, Nicholas and Bishop, Marcus",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1178",
    doi = "10.18653/v1/D19-1178",
    pages = "1684--1695"
}
```

The paper deals with learning vector representations of the activity 
of users on social media. In turn these representations can be used 
to identify authors from their posts. In fact, Author IDentification 
(`aid`) is the basis on which the representation is trained and is 
therefore the name of the Python module included in this repository.

# Package Setup

To train the representation , you will need TensorFlow `2.0` and 
`tqdm`, which you can install with the command `pip install 
tensorflow{-gpu} tqdm`. Note that while evaluating the representation 
can be performed on a CPU, the training stage is considerably faster 
on a GPU, which would require the package `tensorflow-gpu` rather 
than `tensorflow`. The Author IDdentification (`aid`) package 
contained in this repository may then be installed by executing the 
following command in the root directory of this repository.

```
python setup.py install
```

To perform ranking evaluations, `scikit-learn==0.21.3` is required. 
If you use `conda`, the following commands should be all you need to 
get a working environment:


```bash
conda create -n iur python=3.7
conda activate iur
pip install tensorflow-gpu
pip install tqdm
pip install scikit-learn
python setup.py install
```

# Quick Start

Use `scripts/fit.py` to train new models. You must supply paths to 
preprocessed data in `TFRecord` format, as described below. Once a 
model is trained, it may be evaluated using the same script with the 
`--mode rank` flag.

For example, to fit a new model using the provided Reddit dataset:

```bash
python scripts/fit.py --expt_config_path data/reddit/config.json
                      --expt_dir experiment
		      --train_tfrecord_path "$DATA/queries.*.tf"
		      --valid_tfrecord_path "$DATA/targets.*.tf"
		      --num_classes 120601
		      --framework custom
```

This assumes your preprocessed data is located in directory specified 
by the environment variable `$DATA` matching the given file pattern. 
Note the quotes around the data paths, which are intended to prevent 
the expansion of the glob by the shell.

**Note**: `fit.py` supports two different training frameworks, one 
based on the Keras `fit` method, and custom framework that provides 
more user feedback, including periodic ranking experiments to help 
gauge training progress. We recommend using the custom framework, as 
shown in the command above.

The optimization and model architecture may be customized by passing 
in various hyperparameters at the command line. We have found the 
model to be fairly robust across different datasets, but better 
performance can usually be obtained with some fine-tuning. To get 
started, we provide an example set of flags in 
`flagfiles/sample.cfg`. You can pass this to `fit.py` directly by 
introducing the flag `--flagfile flagfiles/sample.cfg` in the command 
above.

**Note**: When evaluating a trained model using `--mode rank`, the
hyperparameters must be the same as were used when training the
model. This can be accomplished, for example, by using the same flag
file.

If you simply wish to produce vector embeddings, use `--mode embed`
which will produce embeddings of data contained in the TFRecords
specified with the `--train_tfrecord_path` flag. The model is
specified using `--expt_dir` and the remaining flags should match
those used to train the model (which you can ensure by using the same
flag file). The resulting embeddings are written in `numpy` format to
`--output_embed_path`. Note that it is important that the data is
preprocessed using the same pipeline as was used to train the model.
For example, use the same subword vocabulary. Finally, the TFRecords
should contain exactly the episodes you wish to embed, and they should
be of the same length as was used to train the model
(`--episode_len`). We have not yet explored the variable-length
episode setting, but we suspect that sampling episodes of different
length during training would be sufficient to produce models capable
of handling variable-length episodes at test time.


# Reddit Data

We release the data in two formats: (1) in preprocessed binary format 
suitable for training new models with the code released in this 
repository; and (2) as raw comment IDs along with scripts to download 
and prepare the data from scratch.

* [Preprocessed data](https://cs.jhu.edu/~noa/data/emnlp2019.tar.gz). The
preprocessed data is in TFRecord format and divided into training and test
splits, each divided into queries and targets.

* [Raw comment IDs](https://cs.jhu.edu/~noa/data/reddit.tar.gz). We 
provide a script to download, preprocess, and store the same data the 
TFRecord format in `data/reddit/download_and_prepare.sh`.

**Note**: To run the script above, you will need some additional 
packages, namely `sentencepiece==0.1.82`, `pandas==0.25.2`, and a 
recent version of the Google Cloud Python API to download data from 
BigQuery. You will also need to export the environment variables 
listed at the top of the script. We secure Reddit data from BigQuery, 
which is considerably faster than using the Reddit API directly.

# Preparing New Data from Scratch

The simplest way to start is to adapt the process laid out in the 
script `data/reddit/download_and_prepare.sh` mentioned above to 
your data source. The remainder of this section describes what that 
would entail.

First you'll want to assemble all the data you plan to use to train 
and evaluate the model. In our experiments in the paper, this 
involves dozens of short documents composed by around 100,000 
different authors. However, we have also successfully trained 
models with fewer authors. Each document including text content and 
any associated meta-data you think will help distinguish authors. 
For example, in our Reddit experiment, the documents are posts and 
the meta-data includes the publication time of each post and the 
subreddit to which each post was published. You need to organize 
your data by author, and if using publication time as a feature, 
you should also sort the messages of each author by publication 
time. We will refer to the full sorted list of messages by a given 
author as that author's *history*.

Next you'll need to decide which data splits to construct, which 
will depend on the problem you're trying to solve. These include a 
*training* and an optional *vailidation* dataset, to be used to
train the model, and optionally a further training and validation
set to be used to evaluate the model, for a total of four possible 
datasets.

At a minimum you'll need a *training* dataset, which you specify to 
`scripts/fit.py` with the `--train_tfrecord_path` flag. Then you 
can train using the flag `--framework fit`, which will simply 
minimize classification error with respect to the the *closed set* 
of authors present in your dataset. You'll also need to specify the 
number of authors `N` using the flag `--num-classes N`. The 
training dataset should consist of the full histories of the `N` 
authors. Additionally, each author should be assigned an 
`author_id` in the range `0..N-1`.

Now in case you are interested in identifying authors who were not 
present in your training dataset, the method just described will be 
insufficient, because it calculates a probability distribution over 
the authors in the training set. Indeed, knowing the probability 
that an episode was composed by one of your training authors 
doesn't help you when you know the author to not be among them! 
Fortunately, although the model is trained to minimize 
classification error with respect to the fixed set of authors, it 
internally constructs an *embedding* of an episode, irrespective of 
its author, which can be used to assess authorship. Specifically, 
episodes by the same author will nearby in space, while episodes by 
different authors will be further apart.

Our primary method of assessing the ability to detect 
same-authorship is formalized by a *ranking experiment*. To conduct 
the experiment periodically during training to monotor progress, 
you use the flag `--framework custom` to `scripts/fit.py` and 
supply a validation dataset with the `--valid-tfrecored-path`. This 
dataset should consist of episodes by exactly the same `N` authors 
comprising the training set, with each episode assigned the same 
`author_id` as the episode by the corresponding author in the 
training set.

In effect, the ranking experiment consists of matching the training 
episodes with the corresponding validation episodes. The success of 
the experiment is measured by a number of metrics, such as *median 
rank*, which should be a small integer, and *mean recripricol 
rank*, which lies in the range `0..1` with larger values 
corresponding with better performance.

A simple way to constuct training and validation splits is to 
simply divide each author's cronologically sorted history in the 
middle, the first portion contributing to the training split, and 
the remainder to the validation split. In this way, the validation 
episodes will be *future* to the training episodes, so the success 
of the experiment will illustrate that the model generalizes well 
to future data.

If desired, you can test the model after training by calling 
`scripts/fit.py` with the `--mode rank` flag, now specifying a 
further training and validation set. The authors and the number of 
authors in these testing datasets may be different than those used 
for training, and there need not be any correspondence between the 
`author_id`s you assign to the training authors and those you 
assign to the testing authors.

# Data format

You need to store each of the desired splits in JSON format, with 
one author history per line. You can store each split in several 
files to avail of TensorFlow's highly optimized data reading 
mechanism, and also to make the dataset less unwieldy. Each line 
should look like the following, but without the newlines we added 
for readability.

```json
{ "author_id" : 0,
  "lens" : [1,2,,...],
  "syms" : [[3,4,...],[5,6,...],...],
  "action_type" : [7,8,...], 
  "hour" : [9,10,...]
}
```

You should use the keys specified in the Features enum defined in 
`aid/features.py`, as we have here. Each JSON object should have a 
unique `author_id` in the range `0...N-1`, where `N` is the number of 
authors in this split. The text content of each post should be 
encoded as a list of integers, with sentencepiece, for example. The 
length of each encoded post should appear in the `lens` field. The 
encoded posts themselves should be concatenated and stored in the 
`syms` field. If using a categorical feature of each post, such as 
subreddit, those can be stored in the action_type field. Finally, 
this example also uses the hour of the day of the publication time of 
each post as a feature. These are shown in the `hour` field.

For efficiency, the JSON file(s) should be converted to TFRecords 
using `scripts/json2tf.py`. You should run this program once of each 
of the four data splits. To run the program, you will need to specify 
a configuration file with the `--config` option. See the file 
`data/reddit/config.json` for an example of such a configuration 
file. You should specify the location of the JSON files you 
constructed with the `--json` option. If you opted to create multiple 
JSON files, you can use a file name glob, remembering to quote the 
whole glob. Finally, you specify the prefix of the names of the 
output files with the `--tf` option.

**Note**: For computational expediency, we truncate the length of each
individual comment to 32 subwords and consider only short episodes up
to length 16. However, with less truncation and longer episodes, we
expect the quality of the embeddings to improve, at the cost of
requiring more computational resources to train the model and produce
embeddings at inference time.