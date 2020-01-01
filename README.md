This is the official repository for the EMNLP 2019 paper, "Learning Invariant
Representations of Social Media Users," by Nicholas Andrews and Marcus Bishop.
If you use the code or datasets released here in published work, the appropriate
citation is:

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

# Package Setup

To train models, you will need TensorFlow `2.1` which you can install
using `pip install tf-nightly{-gpu}`. The Author IDdentification (`aid`)
package contained in this repository may then be installed using:

```
python setup.py install
```

To perform ranking evaluations, `scikit-learn==0.21.3` is required.

# Reddit Data

We release the data in two formats: (1) in preprocessed binary format suitable for training
new models with the code released in this repository; and (2) as raw comment IDs along with scripts 
to download and prepare the data.

**Download links**

* [Preprocessed data](https://cs.jhu.edu/~noa/data/emnlp2019.tar.gz). The
preprocessed data is in TFRecord format and divided into training and test
splits, each divided into queries and targets.

* [Raw comment IDs](https://cs.jhu.edu/~noa/data/reddit.tar.gz). We provide a
script to download and prepare the data into the TFRecord format in
`data/reddit/download_and_prepare.sh`.

# Preparing New Data from Scratch

The simplest way to start is to adapt the process laid out in
`data/reddit/download_and_prepare.sh` for Reddit to your data source. The
remainder of this section describes what that would entail.

**Note**: To run the script above, you will need some additional packages,
namely `sentencepiece==0.1.82`, `pandas==0.25.2`, and a recent version of the
Google Cloud Python API to download data from BigQuery. You will also need
to export the environment variables listed at the top of the script.

First you'll want to assemble all the data you plan to use to train and 
evaluate the model. This would typically include dozens of short 
documents composed by around 100,000 authors, each document including 
text content and any associated meta-data you think will help 
distinguish authors. For example, in our Reddit experiment, the 
documents are posts and the meta-data includes the publication time of 
each post and the subreddit to which each post was published. You need 
to organize your data by author, and if using publication time as a 
feature, you should also sort the messages by each author by publication 
time. We will refer to the full sorted list of messages by a given 
author as that author's history.

You'll want to create both a training and evaluation datasets. The 
evaluation data would typically be future to the training data and 
contain posts by a larger number of authors. Both splits should be 
further divided into query and target sub-splits, for a total of four 
datasets.

For the training splits, take each training author's history and split 
it into two portions, the first portion contributing to the query 
sub-split, and the second portion to the target sub-split. Since the 
targets are typically future to the queries, a simple way to divide the 
history is to take the first half of the posts to comprise the query, 
and the last half to comprise the target. You also need to assign author 
IDs to each author in the range `0..N-1`, where `N` is the number of 
training authors.

You should construct the evaluation sub-splits in the same way, although 
the number of authors can be different, and there need not be any 
correspondence between the IDs you assign to the training authors and 
those you assign to the evaluation authors.

Next you need to store each of the four splits in JSON format, with one 
author history per line. You can use several files for manageability. 
Each line should look like the following, but without the newlines we 
added for readability.

```json
{ "author_id" : 0,
  "lens" : [1,2,,...],
  "syms" : [3,4,...],
  "action_type" : [5,6,...], 
  "hour" : [7,8,...]
}
```

You should use the keys specified in the Features enum defined in 
`aid/io.py`, as we have here. Each JSON object should have a unique 
"author_id" in the range `0...N-1`, where `N` is the number of authors in 
this split. The text content of each post should be encoded as a list of 
integers, with sentencepiece, for example. The length of each encoded 
post should appear in the "lens" field. The encoded posts themselves 
should be concatenated and stored in the "syms" field. If using a 
categorical feature of each post, such as subreddit, those can be stored 
in the action_type field. Finally, this example also uses the hour of 
the day of the publication time of each post as a feature. These are 
shown in the "hour field".

For an example of how to create the JSON files, or if you're only 
interested in repeating the experiment from the paper, please see 
expts/reddit/cookUpRedditJSON.py. As suggested by the file name, this 
program cooks up the JSON file from Reddit data stored locally. We 
secured this data from BigQuery, which is considerably faster than using 
the Reddit API directly.

For efficiency, the JSON file(s) should be converted to TFRecords using
`scripts/json2tf.py`. You should run this program once of each of 
the four data splits. To run the program, you will need to specify a 
configuration file with the `--config` option. See the file 
`data/reddit/config.json` for an example of such a configuration file. You 
should specify the location of the JSON files you constructed with the 
`--json` option. If you opted to create multiple JSON files, you can use a 
file name glob, remembering to quote the whole glob. Finally, you 
specify the prefix of the names of the output files with the `--tf` 
option.

# Building Models

**Coming soon!**
