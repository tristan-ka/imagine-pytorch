import pickle
import numpy as np
import json
import random
import os
import sys
import torch
import logging
import argparse

sys.path.append('../..')
sys.path.append('../../..')

from src.architecture_le2.reward_function.model import RewardFunctionModel
from src.utils.reward_func_util import Batch
from src.utils.data_utils import pickle_dump, evaluate_metrics_pytorch, init_metric_dicts
from src.utils.data_utils import evaluate_metrics_pytorch, init_metric_dicts

# fix pickle import
from src.utils import nlp_tools
sys.modules['src.train.nlp_tools'] = nlp_tools

if __name__ == '__main__':

    dataset = 'rl_big_50k_no_hard_goals'
    # # Data reading
    if 'tristankarch' in os.environ['HOME']:
        DATA_DIR = '../../data/processed/{}/'.format(dataset)
    else:
        DATA_DIR = '/projets/flowers2/tristan/data/processed/{}/'.format(dataset)

    with open(DATA_DIR + 'descriptions_data.pk', 'rb') as f:
        descriptions_data = pickle.load(f)
    id2one_hot = descriptions_data['id2one_hot']
    id2description = descriptions_data['id2description']
    vocab = descriptions_data['vocab']
    max_seq_length = descriptions_data['max_seq_length']
    with open(DATA_DIR + 'train_set.pk', 'rb') as f:
        train_set = pickle.load(f)
    state_idx_buffer = train_set['state_idx_buffer']
    states_train = train_set['states']
    with open(DATA_DIR + 'test_set.pk', 'rb') as f:
        test_set = pickle.load(f)
    state_idx_reward_buffer = test_set['state_idx_reward_buffer']
    states_test = test_set['states']
    descriptions_id_states = state_idx_reward_buffer.keys()
    descriptions_states = [id2description[id] for id in descriptions_id_states]
    # Read testing set with unseen descriptions
    with open(DATA_DIR + 'test_set_language_generalization.pk', 'rb') as f:
        test_set_language = pickle.load(f)
    state_idx_reward_buffer_language = test_set_language['state_idx_reward_buffer']
    states_test_language = test_set_language['states']
    descriptions_id_language = state_idx_reward_buffer_language.keys()
    descriptions_language = [id2description[id] for id in descriptions_id_language]

    vocab_size = vocab.size
    state_size = len(states_train[0])
    n_obj = 3
    body_size = 3
    obj_size = (state_size // 2 - body_size) // n_obj
    batch_size = 512
    n_batch = 200
    body_size = 3
    learning_rate = 0.0001
    num_hidden_lstm = 100
    n_epochs = 0

    or_params = dict()
    for n_obj in range(3, 4):
        or_params[n_obj] = '../../or_module/or_params_pytorch/or_params_{}objs.pk'.format(n_obj)
    reward_func = RewardFunctionModel(or_params, body_size, obj_size,
                                      n_obj, state_size, vocab_size, max_seq_length, batch_size, num_hidden_lstm)


    reward_func_params = dict(vocab_size=vocab_size, state_size=state_size, obj_size=obj_size, body_size=3,
                              learning_rate=learning_rate, n_epochs=n_epochs, n_batch=n_batch, batch_size=batch_size)

    train_metrics_log, metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language = init_metric_dicts(
        descriptions_id_states, descriptions_id_language)

    with open('../../data/expe/PlaygroundNavigation-v1/999/reward_checkpoints/reward_func_latest_checkpoint','rb') as f:
        reward_func.load_state_dict(torch.load(f))


    f1_dict_states, metrics_dict_states = evaluate_metrics_pytorch(reward_func, descriptions_id_states,
                                                                   state_idx_reward_buffer, states_test, id2one_hot,
                                                                   id2description, f1_dict_states,
                                                                   metrics_dict_states,
                                                                   logging)
    f1_dict_language, metrics_dict_language = evaluate_metrics_pytorch(reward_func, descriptions_id_language,
                                                                       state_idx_reward_buffer_language,
                                                                       states_test,
                                                                       id2one_hot, id2description, f1_dict_language,
                                                                       metrics_dict_language, logging)
