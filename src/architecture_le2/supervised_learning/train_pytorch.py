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

# fix pickle import
from src.utils import nlp_tools
sys.modules['src.train.nlp_tools'] = nlp_tools

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


def perform_evaluation(metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language, reward_func,
                       descriptions_id_states, descriptions_id_language, state_idx_reward_buffer,
                       state_idx_reward_buffer_language, states_test, id2one_hot, id2description, reward_func_dir,
                       trial_dir, save=True):
    f1_dict_states, metrics_dict_states = evaluate_metrics_pytorch(reward_func, descriptions_id_states,
                                                                   state_idx_reward_buffer, states_test, id2one_hot,
                                                                   id2description, f1_dict_states,
                                                                   metrics_dict_states,
                                                                   logging)
    pickle_dump(metrics_dict_states, trial_dir + '/metrics_states.pk')
    pickle_dump(f1_dict_states, trial_dir + '/f1_states.pk')

    f1_dict_language, metrics_dict_language = evaluate_metrics_pytorch(reward_func, descriptions_id_language,
                                                                       state_idx_reward_buffer_language,
                                                                       states_test,
                                                                       id2one_hot, id2description, f1_dict_language,
                                                                       metrics_dict_language, logging)
    pickle_dump(metrics_dict_language, trial_dir + '/metrics_language.pk')
    pickle_dump(f1_dict_language, trial_dir + '/f1_language.pk')

    if save == True:
        with open(reward_func_dir + '/model_{}.pk'.format(0), 'wb') as f:
            torch.save(reward_func.state_dict(), f)

    return metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--architecture', type=str, help='Type of architecture to train', default=None)
    add('--n_epochs', type=int, help='Number of training epochs', default=10)
    add('--positive_ratio', type=float, help='Ratio of positive rewards per descriptions', default=0.1)
    add('--dataset', type=str, help='name of dataset folder in the processed directory of data', default='rl_big_small')
    add('--trial_id', type=int, default='333', help='Trial identifier, name of the saving folder')
    add('--git_commit', type=str, help='Hash of git commit', default='no git commit')
    add('--evaluate', type=str, help='whether to evaluate or not', default='yes')
    add('--freq_eval', type=int, help='Frequency of evaluation during training', default=50)

    # os.environ["MKL_NUM_THREADS"] = '1'
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # os.environ['OMP_NUM_THREADS'] = '1'

    # Params parsing
    params = vars(parser.parse_args())
    n_epochs = params['n_epochs']
    positive_ratio = params['positive_ratio']
    trial_id = params['trial_id']
    git_commit = params['git_commit']
    evaluate = params['evaluate']
    freq_eval = params['freq_eval']
    dataset = params['dataset']

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

    # Model init
    seed = int(random.choice(range(1, int(1e6))))
    # params['seed'] = seed
    # set_global_seeds(seed)

    OUTPUT_DIR = '../../data/output/'
    TRIAL_DIR = find_save_path(OUTPUT_DIR, trial_id)
    REWARD_FUNC_DIR = TRIAL_DIR + '/reward_func_ckpt'
    os.makedirs(REWARD_FUNC_DIR)
    LOG_FILENAME = TRIAL_DIR + '/log.log'
    pickle_dump(id2description, TRIAL_DIR + 'id2description.pk')
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

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

    or_params = dict()
    for n_obj in range(3, 4):
        or_params[n_obj] = '../../or_module/or_params_pytorch/or_params_{}objs.pk'.format(n_obj)

    reward_func = RewardFunctionModel(or_params, body_size, obj_size,
                                      n_obj, state_size, vocab_size, max_seq_length, batch_size, num_hidden_lstm)

    reward_func_params = dict(vocab_size=vocab_size, state_size=state_size, obj_size=obj_size, body_size=3,
                              learning_rate=learning_rate, n_epochs=n_epochs, n_batch=n_batch, batch_size=batch_size)
    params['reward_func_params'] = reward_func_params
    params['test_descriptions_language'] = descriptions_language
    params['test_descriptions_states'] = descriptions_states
    with open(TRIAL_DIR + '/params.json', 'w') as fp:
        json.dump(params, fp)

    optimizer = torch.optim.Adam(params=reward_func.get_params(), lr=learning_rate)

    train_metrics_log, metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language = init_metric_dicts(
        descriptions_id_states, descriptions_id_language)

    # if evaluate == 'yes':
    #     metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language = perform_evaluation(
    #         metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language, reward_func,
    #         descriptions_id_states, descriptions_id_language, state_idx_reward_buffer,
    #         state_idx_reward_buffer_language, states_test, id2one_hot, id2description, REWARD_FUNC_DIR,
    #         TRIAL_DIR)

    loss_log = []
    for i in range(n_epochs):
        print('epoch: ', i)
        logging.info('Epoch: ' + str(i))
        batch = Batch(states_train, state_idx_buffer, reward_func.batch_size, id2one_hot, positive_ratio)

        losses_over_epoch = []
        for bb in range(n_batch):
            batch_s, batch_descr, batch_r = batch.next_batch()

            # include this in next_batch() also add reshape of bathc_descr
            batch_s = torch.tensor(batch_s, dtype=torch.float32)
            batch_descr = torch.tensor(batch_descr, dtype=torch.float32)
            batch_r = torch.tensor(batch_r, dtype=torch.float32)
            batch_pred = reward_func(batch_s, batch_descr).squeeze()

            loss = torch.mean(-torch.log(batch_pred) * batch_r + (1 - batch_r) * (-torch.log(1 - batch_pred)))
            losses_over_epoch.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = np.mean(losses_over_epoch)
        print(mean_loss)
        loss_log.append(mean_loss)

        if i % freq_eval == 0:
            if evaluate == 'yes':
                metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language = perform_evaluation(
                    metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language, reward_func,
                    descriptions_id_states, descriptions_id_language, state_idx_reward_buffer,
                    state_idx_reward_buffer_language, states_test, id2one_hot, id2description, REWARD_FUNC_DIR,
                    TRIAL_DIR)

    if evaluate == 'yes':
        if evaluate == 'yes':
            metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language = perform_evaluation(
                metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language, reward_func,
                descriptions_id_states, descriptions_id_language, state_idx_reward_buffer,
                state_idx_reward_buffer_language, states_test, id2one_hot, id2description, REWARD_FUNC_DIR,
                TRIAL_DIR)

    pickle_dump(loss_log, TRIAL_DIR + '/loss_log.pk')
