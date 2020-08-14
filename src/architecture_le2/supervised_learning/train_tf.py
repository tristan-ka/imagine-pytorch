import pickle
import numpy as np
import json
import random
import os
import sys
import tensorflow as tf
import logging
import argparse

sys.path.append('../..')
sys.path.append('../../..')

from src.architecture_le2.reward_function.model import RewardFunctionModel
from src.utils.reward_func_util import Batch
from src.utils.data_utils import pickle_dump, evaluate_metrics
from src.architecture_le2.reward_function.model_reward_function_lstm import RewardFunctionCastAttentionShareOr, \
    RewardFunctionAttention, RewardFunctionConcat,RewardFunctionCastAttentionShareMax


# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# add = parser.add_argument
# add('--architecture', type=str, help='Type of architecture to train')
# add('--n_epochs', type=int, help='Number of training epochs', default=300)
# add('--positive_ratio', type=float, help='Ratio of positive rewards per descriptions')
# add('--dataset', type=str, help='name of dataset folder in the processed directory of data')
# add('--trial_id', type=int, default='333', help='Trial identifier, name of the saving folder')
# add('--git_commit', type=str, help='Hash of git commit', default='no git commit')
# add('--evaluate', type=str, help='whether to evaluate or not', default='yes')
# add('--freq_eval', type=int, help='Frequency of evaluation during training', default=25)


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def find_save_path(dir, trial_id):
    i = 0
    while True:
        save_dir = dir + str(trial_id + i * 100) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            break
        i += 1
    return save_dir


if __name__ == '__main__':
    # os.environ["MKL_NUM_THREADS"] = '1'
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # os.environ['OMP_NUM_THREADS'] = '1'

    # Params parsing
    # params = vars(parser.parse_args())
    # architecture = params['architecture']
    # n_epochs = params['n_epochs']
    # positive_ratio = params['positive_ratio']
    # trial_id = params['trial_id']
    # git_commit = params['git_commit']
    # evaluate = params['evaluate']
    # freq_eval = params['freq_eval']
    # dataset = params['dataset']

    dataset = 'debug_set'
    trial_id = 1
    # Data reading
    DATA_DIR = '../../data/processed/{}/'.format(dataset)

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

    # Model init
    seed = int(random.choice(range(1, int(1e6))))
    # params['seed'] = seed
    # set_global_seeds(seed)

    OUTPUT_DIR = '../../data/output/'
    TRIAL_DIR = find_save_path(OUTPUT_DIR, trial_id)

    vocab_size = vocab.size
    state_size = len(states_train[0])
    n_obj = 3
    body_size = 3
    obj_size = (state_size // 2 - body_size) // n_obj
    batch_size = 512
    n_batch = 200
    body_size = 3
    n_epochs = 50
    positive_ratio = 0.1
    ff_size = 100
    learning_rate = 0.001
    num_hidden_lstm = 100
    path_to_or_params = dict()
    path_to_or_params[n_obj] = '../../or_module/or_params/or_params_3objs.pk'
    reward_func = RewardFunctionCastAttentionShareMax(path_to_or_params, body_size, obj_size, n_obj, state_size,
                                                     vocab_size, max_seq_length, batch_size, learning_rate, ff_size,
                                                     num_hidden_lstm)

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10000)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    with tf.Session(config=session_conf) as sess:
        sess.run(init)
        sess.run(init_local)
        for i in range(n_epochs):
            batch = Batch(states_train, state_idx_buffer, reward_func.batch_size, id2one_hot, positive_ratio)
            for bb in range(n_batch):
                batch_s, batch_i, batch_r = batch.next_batch()
                batch_r = batch_r.reshape([len(batch_r), 1])
                sess.run(reward_func.get_optimizer(),
                         feed_dict={reward_func.S: batch_s, reward_func.I: np.array(batch_i), reward_func.Y: batch_r})
            cost = sess.run(reward_func.get_cost(),feed_dict={reward_func.S: batch_s, reward_func.I: batch_i,
                               reward_func.Y: batch_r})
            print(cost)