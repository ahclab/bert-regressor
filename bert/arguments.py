#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
from collections import OrderedDict
import itertools
import copy
import matplotlib.pyplot as plt
import numpy as np
import argparse

class Arguments():
    def __init__(self):
        pass


# In[2]:


class GridSearch():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        
        self.grid_parmas = OrderedDict(
            [('MLPn_layers', [1])] +
            [('MLPn_units', [768])] + 
            [('MLPdropout', [0.1])] +
            [('learning_rate', [3e-5, 2e-5, 1e-5])] +
            [('train_batch_size', [16, 32])]
        )
        self.current_params = {}
        self.iter = self._iterator()
        self.iter_num = 0
        
    def _iterator(self):
        for MLPn_layer, MLPn_unit, MLPdropout, learning_rate, train_batch_size in itertools.product(self.grid_parmas['MLPn_layers'], 
                                                                                  self.grid_parmas['MLPn_units'], 
                                                                                  self.grid_parmas['MLPdropout'], 
                                                                                  self.grid_parmas['learning_rate'], 
                                                                                  self.grid_parmas['train_batch_size']):
            
            yield MLPn_layer, MLPn_unit, MLPdropout, learning_rate, train_batch_size
        
    def _printGridParams(self, Oneline=False):
        if Oneline:
            string = ''
            for key in self.current_params.keys():
                string = string + str(self.current_params[key])
                string = string + key.upper()
                string = string + '__'
            return string
        else:
            for key in self.current_params.keys():
                string = ''
                string = string + str(self.current_params[key])
                string = string + key.upper()
                print(string)
                
    def setParams(self):
        try:
            self.iter_num += self.iter_num
            
            MLPn_layer, MLPn_unit, MLPdropout, learning_rate, train_batch_size = next(self.iter)
            self.current_params['MLPn_layers'] = MLPn_layer
            self.current_params['MLPn_units'] = MLPn_unit
            self.current_params['MLPdropout'] = MLPdropout
            self.current_params['learning_rate'] = learning_rate
            self.current_params['train_batch_size'] = train_batch_size
            
        except StopIteration:
            print('stop gsearch')
            exit(-1)
            
        for key in self.grid_parmas.keys():
            setattr(FLAGS, str(key), self.current_params[key])


# In[3]:


#DATA_HOME = '/project/nakamura-lab08/Work/kosuke-t/data/SRHDA/bert'
# DATA_HOME = '/project/nakamura-lab08/Work/kosuke-t/data/SRHDA/15-18'
#DATA_HOME2 = '/project/nakamura-lab08/Work/kosuke-t/data/SRHDA/15-18/bert/de-en'

# SAVE_DIR = '/project/nakamura-lab08/Work/kosuke-t/SRHDA/bert/log'
BERT_BASE_DIR='/project/nakamura-lab08/Work/kosuke-t/bert-related/bert/model/uncased_L-12_H-768_A-12'
modes = ['train', 'valid', 'test']

# data_dir = {mode:{'SH':os.path.join(DATA_HOME, '{}SH.tsv'.format(mode)),
#                   'RH':os.path.join(DATA_HOME, '{}RH.tsv'.format(mode))} for mode in modes
#            }
# data_dir = {}

#args.testLangs = 'de-en'
# args.data_dir['train']={'SH':os.path.join(DATA_HOME, '15_17train.{}SH.tsv'.format(args.trainLang)),
#                         'RH':os.path.join(DATA_HOME, '15_17train.{}RH.tsv'.format(args.trainLang))
#                        }
# args.data_dir['valid']={'SH':os.path.join(DATA_HOME, '15_17valid.{}SH.tsv'.format(args.trainLang)),
#                         'RH':os.path.join(DATA_HOME, '15_17valid.{}RH.tsv'.format(args.trainLang))
#                        }

# args.data_dir['train']={'SH':os.path.join(DATA_HOME, '15_17trainSH.tsv'),
#                         'RH':os.path.join(DATA_HOME, '15_17trainRH.tsv')
#                        }
# args.data_dir['valid']={'SH':os.path.join(DATA_HOME, '15_17validSH.tsv'),
#                         'RH':os.path.join(DATA_HOME, '15_17validRH.tsv')
#                        }

# # args.data_dir['test']={lang:
# #                        {'SH':os.path.join(DATA_HOME, '15_17test.{}SH.tsv'.format(lang)),
# #                         'RH':os.path.join(DATA_HOME, '15_17test.{}RH.tsv'.format(lang))} for lang in args.testLangs.split('_')
# #                       }
# args.data_dir['test']={lang:
#                        {'SH':os.path.join(DATA_HOME, '17test.{}SH.tsv'.format(lang)),
#                         'RH':os.path.join(DATA_HOME, '17test.{}RH.tsv'.format(lang))} for lang in args.testLangs.split('_')
#                       }
# args.data_dir['test']['all']['RH'] = os.path.join(DATA_HOME, '15_17testRH.tsv')
# args.data_dir['DArange'] = os.path.join(DATA_HOME, 'DArange.pkl')
# args.data_dir['test']['all']['SH'] = os.path.join(DATA_HOME, '{}SH.tsv'.format('test'))
# args.data_dir['test']['all']['RH'] = os.path.join(DATA_HOME, '{}RH.tsv'.format('test'))

# args.MLPn_layers = 1
# args.MLPn_units = 512
# args.MLPdropout = 0.1

# args.LossPearson = False
# args.LossPearsonCoeff = 2

# args.CosineSimilarity = False
# args.ClassifyTree = False
# args.n_Classify = 5

#args.exp_name = 'RUSE_1517_tree'
# args.exp_name = 'bert-ruse'
# args.exp_id = '0'

# args.output_dir= os.path.join(os.path.join(SAVE_DIR, args.exp_name), str(args.exp_id))


# In[ ]:


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


# In[78]:


# flags = tf.flags
# _FLAGS = flags.FLAGS

# # gsearch = GridSearch(FLAGS)

# ## Required parameters
# flags.DEFINE_string(
#     "data_path", '/project/nakamura-lab08/Work/kosuke-t/data/SRHDA/bert',
#     "The input data dir. Should contain the .tsv files (or other data files) "
#     "for the task.")

# flags.DEFINE_string(
#     "bert_config_file", os.path.join(BERT_BASE_DIR , 'bert_config.json'),
#     "The config json file corresponding to the pre-trained BERT model. "
#     "This specifies the model architecture.")

# flags.DEFINE_string("task_name", 'DA', "The name of the task to train.")

# flags.DEFINE_string("vocab_file", os.path.join(BERT_BASE_DIR, 'vocab.txt'),
#                     "The vocabulary file that the BERT model was trained on.")

# flags.DEFINE_string(
#     "output_dir", None,
#     "The output directory where the model checkpoints will be written.")

# ## Other parameters

# flags.DEFINE_string(
#     "init_checkpoint", os.path.join(BERT_BASE_DIR, 'bert_model.ckpt'),
#     "Initial checkpoint (usually from a pre-trained BERT model).")

# flags.DEFINE_bool(
#     "do_lower_case", True,
#     "Whether to lower case the input text. Should be True for uncased "
#     "models and False for cased models.")

# flags.DEFINE_integer(
#     "max_seq_length", 128,
#     "The maximum total input sequence length after WordPiece tokenization. "
#     "Sequences longer than this will be truncated, and sequences shorter "
#     "than this will be padded.")

# flags.DEFINE_bool("do_train", True, "Whether to run training.")

# flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

# flags.DEFINE_bool(
#     "do_predict", True,
#     "Whether to run the model in inference mode on the test set.")

# flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

# flags.DEFINE_integer("eval_batch_size", 2, "Total batch size for eval.")

# flags.DEFINE_integer("predict_batch_size", 2, "Total batch size for predict.")

# flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")

# flags.DEFINE_integer("num_train_epochs", 10,
#                    "Total number of training epochs to perform.")

# flags.DEFINE_float(
#     "warmup_proportion", 0.05,
#     "Proportion of training to perform linear learning rate warmup for. "
#     "E.g., 0.1 = 10% of training.")

# flags.DEFINE_integer("save_checkpoints_steps", 1000,
#                      "How often to save the model checkpoint.")

# flags.DEFINE_integer("iterations_per_loop", 1000,
#                      "How many steps to make in each estimator call.")

# flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

# flags.DEFINE_string(
#     "tpu_name", None,
#     "The Cloud TPU to use for training. This should be either the name "
#     "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
#     "url.")

# flags.DEFINE_string(
#     "tpu_zone", None,
#     "[Optional] GCE zone where the Cloud TPU is located in. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")

# flags.DEFINE_string(
#     "gcp_project", None,
#     "[Optional] Project name for the Cloud TPU-enabled project. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")

# flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

# flags.DEFINE_integer(
#     "num_tpu_cores", 8,
#     "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# flags.DEFINE_string('exp_id', '1', 'number of experiment')
# flags.DEFINE_string('exp_name', 'test', 'experiment name')
# flags.DEFINE_bool('RUSE', True, 'whether using SRC or not. If yes, RUSE is False')
# flags.DEFINE_bool('write_example', False, 'whether write and make example file')

# # flags.DEFINE_integer('MLPn_layers', 1, 'number of MLP layers')
# # flags.DEFINE_integer('MLPn_units', 756, 'number of MLP units')
# # flags.DEFINE_float('MLPdropout', 0.1, 'MLP dropout rate')

# # flags.DEFINE_bool('LossPearson', False, 'add α(1-pearson) to loss')
# # flags.DEFINE_float('LossPearsonCoeff', 0.1, 'α')
# # flags.DEFINE_bool('CosineSimilarity', False, 'use cosine similarity')
# # flags.DEFINE_bool('ClassifyTree', False, 'output classify tree')
# # flags.DEFINE_integer('n_Classify', 5, 'number of categories to classify')

# flags.DEFINE_bool('output_test', False, 'output test or not')

# flags.DEFINE_string('trainLang', 'cs-en_de-en_fi-en_ro-en_tr-en_ru-en', 'experiment name')
# flags.DEFINE_string('testLangs', 'cs-en_de-en_fi-en_lv-en_ru-en_tr-en_zh-en_all', 'experiment name')
# flags.DEFINE_string('save_dir', '/project/nakamura-lab08/Work/kosuke-t/SRHDA/bert/log/bertTensor/1', 'experiment name')


# In[ ]:


# parse parameters
parser = argparse.ArgumentParser(description='bertTensor')

parser.add_argument("--data_path", type=str, default='/project/nakamura-lab08/Work/kosuke-t/data/SRHDA/bert',
                        help="The input data dir. Should contain the .tsv files (or other data files")
parser.add_argument("bert_config_file", type=str, default=os.path.join(BERT_BASE_DIR , 'bert_config.json'))
parser.add_argument('task_name', type=str, default='DA')
parser.add_argument('vocab_file', type=str, default=os.path.join(BERT_BASE_DIR, 'vocab.txt'))
parser.add_argument('output_dir', type=str, default='')

## Other parameters
parser.add_argument('init_checkpoint', type=str, default=os.path.join(BERT_BASE_DIR, 'bert_model.ckpt'))
parser.add_argument('do_lower_case', type=bool_flag, default=True)
parser.add_argument('max_seq_length', type=int, default=128, help='The maximum total input sequence length after WordPiece tokenization.')
parser.add_argument('do_train', type=bool_flag, default=True)
parser.add_argument('do_eval', type=bool_flag, default=True)
parser.add_argument('do_predict', type=bool_flag, default=True)
parser.add_argument('train_batch_size', type=int, default=32)
parser.add_argument('eval_batch_size', type=int, default=2)
parser.add_argument('predict_batch_size', type=int, default=2)
parser.add_argument('learning_rate', type=float, default=3e-5)
parser.add_argument('num_train_epochs', type=int, default=3)
parser.add_argument('warmup_proportion', type=float, default=0.05)
parser.add_argument('save_checkpoints_steps', type=int, default=1000)
parser.add_argument('iterations_per_loop', type=int, default=1000)
parser.add_argument('use_tpu', type=bool_flag, default=False)
parser.add_argument('tpu_name', type=str, default=None)
parser.add_argument('tpu_zone', type=str, default=None)
parser.add_argument('gcp_project', type=str, default=None)
parser.add_argument('master', type=str, default=None)
parser.add_argument('num_tpu_cores', type=int, default=8)
parser.add_argument('exp_id', type=str, default='1')
parser.add_argument('exp_name', type=str, default='test')
parser.add_argument('RUSE', type=bool_flag, default=True)
parser.add_argument('write_example', type=bool_flag, default=False)

parser.add_argument('output_test', type=bool_flag, default=False)
parser.add_argument('trainLang', type=str, default='cs-en_de-en_fi-en_ro-en_tr-en_ru-en')
parser.add_argument('testLangs', type=str, default='cs-en_de-en_fi-en_lv-en_ru-en_tr-en_zh-en_all')
parser.add_argument('save_dir', type=str, default='/project/nakamura-lab08/Work/kosuke-t/SRHDA/bert/log/bertTensor/1')


# In[ ]:


def set_args(FLAGS):
#     for flag in args.__dict__.keys():
#         setattr(FLAGS, str(flag), getattr(args, str(flag)))
    
#     gsearch.setParams()
#     args.exp_id = gsearch._printGridParams(Oneline=True)

#     FLAGS = Arguments()
#     for key in _FLAGS.__dict__.keys():
#         if hasattr(_FLAGS, str(key)):
#             setattr(FLAGS, str(key), getattr(_FLAGS, str(key)))
#     import pdb;pdb.set_trace()
    FLAGS = parser.parse_args()
    FLAGS.output_dir = os.path.join(os.path.join(FLAGS.save_dir, FLAGS.exp_name), str(FLAGS.exp_id))
    #     if not os.path.exists(args.output_dir):
    #         os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir, exist_ok=True)

    #     KEYS = ['exp_name', 'exp_id', 'output_dir']
    #     for flag in KEYS:
    #         setattr(FLAGS, str(flag), getattr(args, str(flag)))

    #     gsearch._printGridParams()

    FLAGS.data_dir = {}
    FLAGS.data_dir['train']={'SH':os.path.join(FLAGS.data_path, '15_17trainSH.tsv'),
                            'RH':os.path.join(FLAGS.data_path, '15_17trainRH.tsv')
                           }
    FLAGS.data_dir['valid']={'SH':os.path.join(FLAGS.data_path, '15_17validSH.tsv'),
                            'RH':os.path.join(FLAGS.data_path, '15_17validRH.tsv')
                           }

    # args.data_dir['test']={lang:
    #                        {'SH':os.path.join(DATA_HOME, '15_17test.{}SH.tsv'.format(lang)),
    #                         'RH':os.path.join(DATA_HOME, '15_17test.{}RH.tsv'.format(lang))} for lang in args.testLangs.split('_')
    #                       }
    FLAGS.data_dir['test']={lang:
                           {'SH':os.path.join(FLAGS.data_path, '17test.{}SH.tsv'.format(lang)),
                            'RH':os.path.join(FLAGS.data_path, '17test.{}RH.tsv'.format(lang))} for lang in FLAGS.testLangs.split('_')
                          }
    FLAGS.data_dir['test']['all']['RH'] = os.path.join(FLAGS.data_path, '15_17testRH.tsv')
    FLAGS.data_dir['DArange'] = os.path.join(FLAGS.data_path, 'DArange.pkl')

    with open('params.txt', mode='w', encoding='utf-8') as w:
        w.write('{}'.format(os.linesep).join("%s: %s" % item for item in FLAGS.items()))
        
    return FLAGS


# In[5]:


def get_best_result(PRINT=True, RETURN_RESULTS=True):
    result_home = os.path.join(SAVE_DIR, args.exp_name)
    results_dir = os.listdir(result_home)
    langs = args.testLangs.split('_')
    test_results = {lang:{params:[] for params in results_dir} for lang in langs}
    for lang in langs:
        for result_dir in results_dir:
            if result_dir.startswith('.'):
                continue
            params = result_dir
            rdir = os.path.join(result_home, result_dir)
            for i in range(args.num_train_epochs):
                with open(os.path.join(rdir, 'test_results.{}.{}.txt'.format(lang,i+1)), mode='r', encoding='utf-8') as r:
                    test_results[lang][params].append(r.read().split(os.linesep))
                    if test_results[lang][params][-1][-1] == '':
                        test_results[lang][params][-1].pop(-1)
     
        if PRINT:
            print('#### {} #####'.format(lang))
            print(test_results[lang])
            print('#############')
    if RETURN_RESULTS: 
        return test_results
    
    best_value = {lang:0.0 for lang in langs}
    best_param = {lang:'' for lang in langs}
    for lang in langs:
        for params in results_dir:
            test_results[lang][params] = [float(string[0].split()[-1]) for string in test_results[lang][params]]
            max_v = max(test_results[lang][params])
            if best_value[lang] < max_v:
                best_value[lang] = max_v
                best_param[lang] = params
    return best_value, best_param
    
def get_best_result_shimanaka(PRINT=True, RETURN_RESULTS=True):
    result_home = os.path.join(SAVE_DIR, args.exp_name)
    dirs = os.listdir(result_home)
    results_dir = []
    for d in dirs:
        if os.path.isdir(os.path.join(result_home,d)) and not d.startswith('.'):
            results_dir.append(d)
    langs = args.testLangs.split('_')
    test_results = {lang:{params:[] for params in results_dir} for lang in langs}
    for lang in langs:
        for result_dir in results_dir:
            if result_dir.startswith('.'):
                continue
            params = result_dir
            rdir = os.path.join(result_home, result_dir)
            for i in range(args.num_train_epochs):
                with open(os.path.join(rdir, 'test_results.{}.{}.txt'.format(lang,i+1)), mode='r', encoding='utf-8') as r:
                    test_results[lang][params].append(r.read().split(os.linesep))
                    if test_results[lang][params][-1][-1] == '':
                        test_results[lang][params][-1].pop(-1)
     
        if PRINT:
            print('#### {} #####'.format(lang))
            print(test_results[lang])
            print('#############')
    if RETURN_RESULTS: 
        return test_results
    
    best_value = {lang:0.0 for lang in langs}
    best_param = {lang:'' for lang in langs}
    for lang in langs:
        for params in test_results[lang].keys():
            test_results[lang][params] = [float(epoch_result[3].split()[-1]) for epoch_result in test_results[lang][params]]
            max_v = max(test_results[lang][params])
            if best_value[lang] < max_v:
                best_value[lang] = max_v
                best_param[lang] = params

    with open(os.path.join(result_home, 'best_results.txt'), mode='w', encoding='utf-8') as w:
        for lang in best_value.keys():
            w.write(str(lang)+' : '+str(best_value[lang])+'\n')
        for lang in best_param.keys():
            w.write(str(lang)+' : '+str(best_param[lang])+'\n')
    
    return best_value, best_param
    
def plot_curve():
    result_home = os.path.join(SAVE_DIR, args.exp_name)
    results_dir = os.listdir(result_home)
    langs = args.testLangs.split('_')
    eval_results = {}
    test_results = get_best_result(PRINT=False)
    for lang in langs:
        for params in results_dir:
            test_results[lang][params] = [float(string[0].split()[-1]) for string in test_results[lang][params]]
    for result_dir in results_dir:
        if result_dir.startswith('.'):
            continue
        params = result_dir
        rdir = os.path.join(result_home, result_dir)
        eval_results[params] = {'loss':[0.00]*(args.num_train_epochs), 
                                'pearson':[-1.0]*(args.num_train_epochs), 
                                'txt':['']*(args.num_train_epochs)}
        for i in range(args.num_train_epochs):
            with open(os.path.join(rdir, 'eval_results{}.txt'.format(i+1)), mode='r', encoding='utf-8') as r:
                eval_results[params]['txt'][i] = r.read().split(os.linesep)
                if eval_results[params]['txt'][i][-1] == '':
                    eval_results[params]['txt'][i].pop(-1)
            eval_results[params]['txt'][i].pop(0)
            eval_results[params]['txt'][i] = eval_results[params]['txt'][i]
            eval_results[params]['loss'][i] = float(eval_results[params]['txt'][i][0].split()[-1])
            eval_results[params]['pearson'][i] = float(eval_results[params]['txt'][i][1].split()[-1])
        
        r = max(test_results['all'][params])
        e = test_results['all'][params].index(r)+1
        epoch = args.num_train_epochs
        
        val_loss = eval_results[params]['loss']
        val_pearson = eval_results[params]['pearson']
        plt.figure(figsize=(7, 5), dpi=100)
        plt.plot(range(epoch), val_loss, color='blue', linestyle='--', label='val_loss')
        plt.xticks(np.arange(1, epoch + 1, 1))
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(params)
        plt.suptitle('Best Pearson\'s r: {:.4f} @ {} epoch'.format(r, e))
        plt.grid()
        plt.savefig(os.path.join(rdir, 'val_loss.png'))
        plt.close()

        plt.figure(figsize=(7, 5), dpi=100)
        plt.plot(range(epoch), val_pearson, color='blue', linestyle='--', label='val_pearson')
        plt.xticks(np.arange(1, epoch + 1, 1))
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('pearson')
        plt.title(params)
        plt.suptitle('Pearson\'s r: {:.4f} @ {} epoch'.format(r, e))
        plt.grid()
        plt.savefig(os.path.join(rdir, 'val_pearson.png'))
        plt.close()


# In[6]:


#get_best_result(PRINT=False, RETURN_RESULTS=False)


# In[36]:


#get_best_result_shimanaka(PRINT=False, RETURN_RESULTS=False)


# In[30]:


#plot_curve()


# In[7]:


#! ls /project/nakamura-lab08/Work/kosuke-t/data/SRHDA/15-18/bert


# In[16]:


# import tensorflow as tf
# t1 = tf.random_normal([32])
# t2 = tf.random_normal([32])
# t3 = tf.stack([t1, t2], axis=0)
# tf.stack([t1, t3], axis=1).shape.as_list()


# In[4]:


# ! pwd
# ! cp /home/is/kosuke-t/bert-related/bert/tokenization.py /home/is/kosuke-t/bert-related/utils/bert


# In[ ]:




