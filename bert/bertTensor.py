#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import scipy
import numpy as np
from scipy.stats import pearsonr as pr
from scipy.stats import spearmanr as sr
import pickle
import random

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_test", True, "Whether to run test on the test set.")    # ADDED

flags.DEFINE_bool("benchmark", False, "Whether to run a quick benchmark on prediction or not.")    # ADDED

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("test_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# flags.DEFINE_string('train_langs', 'cs-en_de-en_fi-en_ro-en_tr-en_ru-en', 'train language pairs')
flags.DEFINE_string('test_langs', 'cs-en_de-en_fi-en_lv-en_ru-en_tr-en_zh-en', 'test language pairs')
flags.DEFINE_integer('n_operation', 1, 'number of operation')
flags.DEFINE_string('exp_name', 'test', 'experiment name')
flags.DEFINE_string('exp_id', '1', 'experiment id for hyperparameter gridsearch')

flags.DEFINE_string('addSRC', 'False', 'whether to add SRC')
flags.DEFINE_string('onlyREF', 'True', 'whether to use only REF')
flags.DEFINE_string('onlySRC', 'False', 'whether to use only SRC')
flags.DEFINE_float('train_shrink', 1.0, 'ratio of train shrink')


# In[ ]:


def calc_pearson(pred, true):
    try:
        r, p_value = pr(np.asarray(pred), np.asarray(true))
    except ValueError:
        r = -1.0
    return r

def calc_spearman(pred, true):
    try:
        r, p_value = sr(np.asarray(pred), np.asarray(true))
    except ValueError:
        r = -1.0
    return r

def r_func(x):
    return x


# In[ ]:


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, raw_inputs):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.raw_inputs = raw_inputs


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
  
    # ADDED
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class DAProcessor(DataProcessor):
    """Processor for the WMT DA data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(data_dir), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(data_dir), "dev")

    # ADDED
    def get_test_examples(self, data_dir, lang):
        """See base class."""
        return self._create_examples(self._read_tsv(data_dir), "test-{}".format(lang))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[-3])
            text_b = tokenization.convert_to_unicode(line[-2])
            label = float(line[-1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    ###


# In[ ]:


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, shut_up=False, shrink_ratio=1.0):
    """Loads a data file into a list of `InputBatch`s."""

#     label_map = {}
#     for (i, label) in enumerate(label_list):
#         label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        raw_inputs = example.text_a+"\t"+example.text_b
        
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        label_id = example.label
        # label_id = label_map[example.label]
        if not shut_up:
            if ex_index < 1:
                tf.logging.info("*** Example ***")
                tf.logging.info("guid: %s" % (example.guid))
                tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                tf.logging.info("label: {} (id = {})".format(example.label, label_id))
        
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id, 
                raw_inputs=raw_inputs))
    random.shuffle(features)
    features = features[:int(len(features)*shrink_ratio)]
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, 
                 input_ids1, input_mask1, segment_ids1, labels1,
                 input_ids2, input_mask2, segment_ids2, labels2,
                 use_one_hot_embeddings, FLAGS):
    """Creates a classification model."""
    labels = labels1
    
    model1 = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids1,
                               input_mask=input_mask1,
                               token_type_ids=segment_ids1,
                               use_one_hot_embeddings=use_one_hot_embeddings)
    
    if FLAGS.addSRC == 'True':
        model2 = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids2,
                               input_mask=input_mask2,
                               token_type_ids=segment_ids2,
                               use_one_hot_embeddings=use_one_hot_embeddings)
    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model1.get_pooled_output()
    if FLAGS.addSRC == 'True':
        output_layer2 = model2.get_pooled_output()
        output_layer = tf.concat([output_layer, output_layer2], 1,name='concat')
    
    hidden_size = output_layer.shape[-1].value
    
    output_weights = tf.get_variable(
        "output_weights", [1, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [1], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.squeeze(logits, [-1])
        # log_probs = tf.nn.log_softmax(logits, axis=-1)

        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = tf.square(logits - labels)

        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, FLAGS, tokenizer):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        
        if not FLAGS.addSRC:
            input_ids1 = features["input_ids1"]
            input_mask1 = features["input_mask1"]
            segment_ids1 = features["segment_ids1"]
            label_ids1 = features["label_ids1"]
            raw_input1 = features["raw_input1"]
            
            input_ids2 = None
            input_mask2 = None
            segment_ids2 = None
            label_ids2 = None
        else:
            input_ids1 = features["input_ids1"]
            input_mask1 = features["input_mask1"]
            segment_ids1 = features["segment_ids1"]
            label_ids1 = features["label_ids1"]
            raw_input1 = features["raw_input1"]
            
            input_ids2 = features["input_ids2"]
            input_mask2 = features["input_mask2"]
            segment_ids2 = features["segment_ids2"]
            label_ids2 = features["label_ids2"]
            raw_input2 = features["raw_input2"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits) = create_model(
            bert_config, is_training, 
            input_ids1, input_mask1, segment_ids1, label_ids1,
            input_ids2, input_mask2, segment_ids2, label_ids2,
            use_one_hot_embeddings, FLAGS)
        label_ids = label_ids1
        
        
        tvars = tf.trainable_variables()

        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

#         tf.logging.info("**** Trainable Variables ****")
#         for var in tvars:
#             init_string = ""
#             if var.name in initialized_variable_names:
#                 init_string = ", *INIT_FROM_CKPT*"
#             tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          loss=total_loss,
                                                          train_op=train_op,
                                                          scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

#             def metric_fn(per_example_loss, label_ids, logits):
#                 predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
#                 accuracy = tf.metrics.accuracy(label_ids, predictions)
#                 loss = tf.metrics.mean(per_example_loss)
#                 return {
#                     "eval_accuracy": accuracy,
#                     "eval_loss": loss,
#                 }
            def metric_fn(per_example_loss, label_ids, logits, raw_input1):       
                # Display labels and predictions
                concat1 = tf.contrib.metrics.streaming_concat(logits)
                concat2 = tf.contrib.metrics.streaming_concat(label_ids)
                raws = tf.contrib.metrics.streaming_concat(raw_input1)

                # Compute Pearson correlation
                pearson = tf.contrib.metrics.streaming_pearson_correlation(logits, label_ids)

                # Compute MSE
                # mse = tf.metrics.mean(per_example_loss)    
                mse = tf.metrics.mean_squared_error(label_ids, logits)

                # Compute Spearman correlation
                size = tf.size(logits)
                indice_of_ranks_pred = tf.nn.top_k(logits, k=size)[1]
                indice_of_ranks_label = tf.nn.top_k(label_ids, k=size)[1]
                rank_pred = tf.nn.top_k(-indice_of_ranks_pred, k=size)[1]
                rank_label = tf.nn.top_k(-indice_of_ranks_label, k=size)[1]
                rank_pred = tf.to_float(rank_pred)
                rank_label = tf.to_float(rank_label)
                spearman = tf.contrib.metrics.streaming_pearson_correlation(rank_pred, rank_label)
                
                return {'pred': concat1, 'label_ids': concat2, 
                        'pearson': pearson, 'spearman': spearman, 'MSE': mse, 'raw':raws}

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits, raw_input1])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          loss=total_loss,
                                                          eval_metrics=eval_metrics,
                                                          scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(features1, features2, seq_length, is_training, drop_remainder, FLAGS):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids1 = []
    all_input_mask1 = []
    all_segment_ids1 = []
    all_label_ids1 = []
    all_raw_input1 = []
    all_input_ids2 = []
    all_input_mask2 = []
    all_segment_ids2 = []
    all_label_ids2 = []
    all_raw_input2 = []
    
    for feature in features1:
        all_input_ids1.append(feature.input_ids)
        all_input_mask1.append(feature.input_mask)
        all_segment_ids1.append(feature.segment_ids)
        all_label_ids1.append(feature.label_id)
        all_raw_input1.append(feature.raw_inputs)
    if features2 != None:
        for feature in features2:
            all_input_ids2.append(feature.input_ids)
            all_input_mask2.append(feature.input_mask)
            all_segment_ids2.append(feature.segment_ids)
            all_label_ids2.append(feature.label_id)
            all_raw_input2.append(feature.raw_inputs)
    
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features1)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        if not FLAGS.addSRC:
            d = tf.data.Dataset.from_tensor_slices({
                "input_ids1":
                    tf.constant(
                        all_input_ids1, shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask1":
                    tf.constant(
                        all_input_mask1,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "segment_ids1":
                    tf.constant(
                        all_segment_ids1,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "label_ids1":
                    # tf.constant(all_label_ids, shape=[num_examples, 0], dtype=tf.float32),
                    tf.constant(all_label_ids1, shape=[num_examples], dtype=tf.float32),
                "raw_input1":
                    tf.constant(all_raw_input1, shape=[num_examples], dtype=tf.string)
            })
        else:
            d = tf.data.Dataset.from_tensor_slices({
                "input_ids1":
                    tf.constant(
                        all_input_ids1, shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask1":
                    tf.constant(
                        all_input_mask1,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "segment_ids1":
                    tf.constant(
                        all_segment_ids1,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "label_ids1":
                    # tf.constant(all_label_ids, shape=[num_examples, 0], dtype=tf.float32),
                    tf.constant(all_label_ids1, shape=[num_examples], dtype=tf.float32),
                "raw_input1":
                    tf.constant(all_raw_input1, shape=[num_examples], dtype=tf.string),
                "input_ids2":
                    tf.constant(
                        all_input_ids2, shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask2":
                    tf.constant(
                        all_input_mask2,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "segment_ids2":
                    tf.constant(
                        all_segment_ids2,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "label_ids2":
                    # tf.constant(all_label_ids, shape=[num_examples, 0], dtype=tf.float32),
                    tf.constant(all_label_ids2, shape=[num_examples], dtype=tf.float32),
                "raw_input2":
                    tf.constant(all_raw_input2, shape=[num_examples], dtype=tf.string)
                
            })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# In[ ]:


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    
    dump_path = os.path.join(os.path.join(FLAGS.output_dir, FLAGS.exp_name), FLAGS.exp_id)
    tf.gfile.MakeDirs(dump_path)
    with tf.gfile.GFile(os.path.join(dump_path, 'params.txt'), 'w') as w:
        for key, value in FLAGS.__flags.items():
            w.write('{}:{}{}'.format(key, getattr(FLAGS, str(key)), os.linesep))
    
    processor = DAProcessor()

    # label_list = processor.get_labels()
    label_list = None

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=dump_path,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples1 = None
    train_examples2 = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        if FLAGS.onlySRC == 'True':
            train_examples1 = processor.get_train_examples(os.path.join(FLAGS.data_dir, 'train.SH.tsv'))
        elif FLAGS.addSRC == 'True':
            train_examples1 = processor.get_train_examples(os.path.join(FLAGS.data_dir, 'train.SH.tsv'))
            train_examples2 = processor.get_train_examples(os.path.join(FLAGS.data_dir, 'train.RH.tsv'))
        else:
            train_examples1 = processor.get_train_examples(os.path.join(FLAGS.data_dir, 'train.RH.tsv'))
        num_train_steps = int(len(train_examples1) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu, 
        FLAGS=FLAGS, 
        tokenizer=tokenizer)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    if FLAGS.do_train:
        import time
        train_t0 = time.time()
        train_features1 = convert_examples_to_features(
            train_examples1, label_list, FLAGS.max_seq_length, tokenizer, shut_up=True, shrink_ratio=FLAGS.train_shrink)
        train_features2 = None
        if FLAGS.addSRC  == 'True':
            train_features2 = convert_examples_to_features(
                train_examples2, label_list, FLAGS.max_seq_length, tokenizer, shut_up=True, shrink_ratio=FLAGS.train_shrink)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples1))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = input_fn_builder(
            features1=train_features1,
            features2=train_features2,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            FLAGS=FLAGS
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        train_t1 = time.time()
        tf.logging.info("training took {} secs".format(train_t1-train_t0))
    

    if FLAGS.do_eval:
        if FLAGS.onlySRC == 'True':
            eval_examples1 = processor.get_dev_examples(os.path.join(FLAGS.data_dir, 'valid.SH.tsv'))
        elif FLAGS.addSRC == 'True':
            eval_examples1 = processor.get_dev_examples(os.path.join(FLAGS.data_dir, 'valid.SH.tsv'))
            eval_examples2 = processor.get_dev_examples(os.path.join(FLAGS.data_dir, 'valid.RH.tsv'))
        else:
            eval_examples1 = processor.get_dev_examples(os.path.join(FLAGS.data_dir, 'valid.RH.tsv'))
        eval_features1 = convert_examples_to_features(
            eval_examples1, label_list, FLAGS.max_seq_length, tokenizer, shut_up=True)
        eval_features2 = None
        if FLAGS.addSRC == 'True':
            eval_features2 = convert_examples_to_features(
            eval_examples2, label_list, FLAGS.max_seq_length, tokenizer, shut_up=True)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples1))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            # Eval will be slightly WRONG on the TPU because it will truncate
            # the last batch.
            eval_steps = int(len(eval_examples1) / FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = input_fn_builder(
            features1=eval_features1,
            features2=eval_features2,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder, 
            FLAGS=FLAGS)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(dump_path, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                if key in ['pearson', 'spearman', 'MSE']:
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
    
    scores = {}
    results = {}
    if FLAGS.do_test:
        test_examples1 = {}
        test_examples2 = {}
        test_features1 = {}
        test_features2 = {}
        for test_lang in FLAGS.test_langs.split('_'):
            scores[test_lang] = {'pearson':-1.0, 'spearman':-1.0, 'loss':100}
            if FLAGS.onlySRC == 'True':
                test_examples1[test_lang] = processor.get_test_examples(os.path.join(FLAGS.data_dir, 'test.{}.SH.tsv'.format(test_lang)), test_lang)
                test_features1[test_lang] = convert_examples_to_features(test_examples1[test_lang],
                                                                         label_list, 
                                                                         FLAGS.max_seq_length,
                                                                         tokenizer, 
                                                                         shut_up=True)
                test_features2[test_lang] = None
            elif FLAGS.addSRC == 'True':
                test_examples1[test_lang] = processor.get_test_examples(os.path.join(FLAGS.data_dir, 'test.{}.SH.tsv'.format(test_lang)), test_lang)
                test_features1[test_lang] = convert_examples_to_features(test_examples1[test_lang],
                                                                         label_list, 
                                                                         FLAGS.max_seq_length,
                                                                         tokenizer, 
                                                                         shut_up=True)
                test_examples2[test_lang] = processor.get_test_examples(os.path.join(FLAGS.data_dir, 'test.{}.RH.tsv'.format(test_lang)), test_lang)
                test_features2[test_lang] = convert_examples_to_features(test_examples2[test_lang],
                                                                         label_list,
                                                                         FLAGS.max_seq_length,
                                                                         tokenizer,
                                                                         shut_up=True)
            else:
                test_examples1[test_lang] = processor.get_test_examples(os.path.join(FLAGS.data_dir, 'test.{}.RH.tsv'.format(test_lang)), test_lang)
                test_features1[test_lang] = convert_examples_to_features(test_examples1[test_lang],
                                                                         label_list,
                                                                         FLAGS.max_seq_length,
                                                                         tokenizer,
                                                                         shut_up=True)
                test_features2[test_lang] = None
                
            tf.logging.info("***** Running test {} *****".format(test_lang))
            tf.logging.info("  Num examples = %d", len(test_examples1[test_lang]))
            tf.logging.info("  Batch size = %d", FLAGS.test_batch_size)

            # This tells the estimator to run through the entire set.
            test_steps = None
            # However, if running eval on the TPU, you will need to specify the
            # number of steps.
            if FLAGS.use_tpu:
                # Eval will be slightly WRONG on the TPU because it will truncate
                # the last batch.
                test_steps = int(len(test_examples1[test_lang]) / FLAGS.test_batch_size)

            test_drop_remainder = True if FLAGS.use_tpu else False
            test_input_fn = input_fn_builder(
                features1=test_features1[test_lang],
                features2=test_features2[test_lang],
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=test_drop_remainder, 
                FLAGS=FLAGS)

            result = estimator.evaluate(input_fn=test_input_fn, steps=test_steps)
            
            output_test_file = os.path.join(dump_path, "test_results.{}.txt".format(test_lang))
            with tf.gfile.GFile(output_test_file, "w") as writer:
                tf.logging.info("***** Test results {} *****".format(test_lang))
                for key in sorted(result.keys()):
                    if key in ['pearson', 'spearman', 'MSE']:
                        tf.logging.info("  {} = {}".format(key, str(result[key])))
                        writer.write("{} = {}".format(key, str(result[key])))
                pearson = calc_pearson(result['pred'], result['label_ids'])
                spearman = calc_spearman(result['pred'], result['label_ids'])
                loss = result['MSE']
                scores[test_lang]['pearson'] = pearson
                scores[test_lang]['spearman'] = spearman
                scores[test_lang]['loss'] = loss
                if 'lang' not in results:
                    results['lang'] = []
                results['lang'].extend([test_lang for _ in range(len(result['pred']))])
                if 'pred' not in results:
                    results['pred'] = []
                results['pred'].extend(result['pred'])
                if 'true' not in results:
                    results['true'] = []
                results['true'].extend(result['label_ids'])
                if 'raw' not in results:
                    results['raw'] = []
                results['raw'].extend(result['raw'])
        
        with open(os.path.join(dump_path, 'scores{}.pkl'.format(FLAGS.n_operation)), mode='wb') as w:
            pickle.dump(scores, w)
        with open(os.path.join(dump_path, 'result{}.pkl'.format(FLAGS.n_operation)), mode='wb') as w:
            pickle.dump(results, w)
        
    # Benchmark how long it takes for prediction
    if FLAGS.benchmark:
        import time
        t0 = time.time()
        for i in range(100):
            predict_example = [InputExample(guid=2, text_a='He is a smart and experienced person', text_b='He\'s a truly wise man')]
            predict_features = convert_examples_to_features(predict_example,
                                                            label_list,
                                                            FLAGS.max_seq_length,
                                                            tokenizer,
                                                            shut_up=True)
            input_fn = input_fn_builder(features=predict_features, seq_length=FLAGS.max_seq_length, is_training=False, drop_remainder=False)
            result = estimator.predict(input_fn=input_fn)
        t1 = time.time()
        print("Time necessary for 1 prediction : {}".format((t1 - t0) / 100))
        print("Time necessary for training : {}".format(train_t1 - train_t0))

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()


# In[1]:





# In[ ]:




