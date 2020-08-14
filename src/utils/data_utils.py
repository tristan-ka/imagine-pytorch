import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch


def pickle_dump(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def create_train_set(obs_train, descriptions_ids_train, id2description, size_state):
    state_train_idx = 0
    state_train_list = []
    state_idx_buffer = dict(
        zip(list(id2description.keys()), [{'pos_reward': [], 'neg_reward': []} for _ in range(len(id2description))]))
    for obs_episode, descriptions_ids_episode in zip(obs_train, descriptions_ids_train):
        o = obs_episode[-1]
        d_list = descriptions_ids_episode[-1]
        state_train_list.append(o[:size_state])
        indices_pos = set(d_list)
        indices_neg = list(set(id2description.keys()) - set(indices_pos))
        if len(indices_pos) + len(indices_neg) == len(id2description.keys()):
            for id_pos in indices_pos:
                state_idx_buffer[id_pos]['pos_reward'].append(state_train_idx)
            for id_neg in indices_neg:
                state_idx_buffer[id_neg]['neg_reward'].append(state_train_idx)
        else:
            raise ('Error in data')
        state_train_idx += 1

    return dict(states=state_train_list, state_idx_buffer=state_idx_buffer, id2description=id2description)


def create_test_set(obs_test, descriptions_ids_test, id2description, size_state):
    state_test_idx = 0
    state_test_list = []
    state_idx_reward_buffer = dict(
        zip(id2description.keys(), [[] for _ in range(len(id2description.keys()))]))  # key - goal idx, value [(s,r)]
    count = 1
    for obs_episode, descriptions_ids_episode in zip(obs_test, descriptions_ids_test):
        if count % 100 == 0:
            print(count, '/', len(obs_test))
        for o, d_list in zip(obs_episode, descriptions_ids_episode):
            state_test_list.append(o[:size_state])
            indices_pos = set(d_list)
            indices_neg = list(set(id2description.keys()) - set(indices_pos))
            if len(indices_pos) + len(indices_neg) == len(id2description.keys()):
                for id_pos in indices_pos:
                    state_idx_reward_buffer[id_pos].append((state_test_idx, 1))
                for id_neg in indices_neg:
                    state_idx_reward_buffer[id_neg].append((state_test_idx, 0))
                state_test_idx += 1
            else:
                raise ('Error in data')
        count += 1
    return dict(states=state_test_list, state_idx_reward_buffer=state_idx_reward_buffer, id2description=id2description)


def create_test_set_only_last_transition(obs_test, descriptions_ids_test, id2description, size_state):
    state_test_idx = 0
    state_test_list = []
    state_idx_reward_buffer = dict(
        zip(id2description.keys(), [[] for _ in range(len(id2description.keys()))]))  # key - goal idx, value [(s,r)]
    count = 1
    for obs_episode, descriptions_ids_episode in zip(obs_test, descriptions_ids_test):
        if count % 100 == 0:
            print(count, '/', len(obs_test))
        o = obs_episode[-1]
        d_list = descriptions_ids_episode[-1]
        state_test_list.append(o[:size_state])
        indices_pos = set(d_list)
        indices_neg = list(set(id2description.keys()) - set(indices_pos))
        if len(indices_pos) + len(indices_neg) == len(id2description.keys()):
            for id_pos in indices_pos:
                state_idx_reward_buffer[id_pos].append((state_test_idx, 1))
            for id_neg in indices_neg:
                state_idx_reward_buffer[id_neg].append((state_test_idx, 0))
            state_test_idx += 1
        else:
            raise ('Error in data')
        count += 1
    return dict(states=state_test_list, state_idx_reward_buffer=state_idx_reward_buffer, id2description=id2description)


def generate_full_train_set(state_idx_buffer, train_states, instructions_id, id2encoded):
    states_train_out = []
    seqs_train_out = []
    rewards_train_out = []
    for id in instructions_id:
        for state_id in state_idx_buffer[id]['pos_reward']:
            states_train_out.append(train_states[state_id])
            rewards_train_out.append(1)
            seqs_train_out.append(id2encoded[id])
        for state_id in state_idx_buffer[id]['neg_reward']:
            states_train_out.append(train_states[state_id])
            rewards_train_out.append(0)
            seqs_train_out.append(id2encoded[id])

    return states_train_out, seqs_train_out, rewards_train_out


def generate_test_set_from_buffer(state_idx_reward_buffer, test_states, instructions_id, id2encoded, short=False,
                                  threshold=1500):
    states_test_out = []
    seqs_out = []
    rewards_test = []
    count_reward_dict = dict(zip(instructions_id, [0 for _ in range(len(instructions_id))]))
    for id in instructions_id:
        for state_id, reward in state_idx_reward_buffer[id]:
            if reward == 1:
                count_reward_dict[id] += 1
            if short:
                if count_reward_dict[id] < threshold:
                    states_test_out.append(test_states[state_id])
                    seqs_out.append(id2encoded[id])
                    rewards_test.append(reward)
            else:
                states_test_out.append(test_states[state_id])
                seqs_out.append(id2encoded[id])
                rewards_test.append(reward)

    return states_test_out, seqs_out, rewards_test


def generate_test_set_for_instruction(state_idx_reward_buffer, test_states, instruction_id, id2encoded, short=False,
                                      treshold=1500):
    states_test_out = []
    one_hot_seqs_test = []
    rewards_test = []
    count_reward = 0
    for state_id, reward in state_idx_reward_buffer[instruction_id]:
        if reward == 1:
            count_reward += 1
        if short:
            if count_reward < treshold:
                states_test_out.append(test_states[state_id])
                one_hot_seqs_test.append(id2encoded[instruction_id])
                rewards_test.append(reward)
        else:
            states_test_out.append(test_states[state_id])
            one_hot_seqs_test.append(id2encoded[instruction_id])
            rewards_test.append(reward)

    return states_test_out, one_hot_seqs_test, rewards_test


def evaluate_metrics(session, reward_func, descriptions_id, state_idx_reward_buffer, states_test, id2one_hot,
                     id2description,
                     f1_dict, metrics_dict, logging):
    metrics = [reward_func.get_accuracy(), reward_func.get_precision(), reward_func.get_recall()]
    logging.info('Evaluating')
    for id in list(descriptions_id):
        s_test, one_hot_test, r_test = generate_test_set_for_instruction(state_idx_reward_buffer,
                                                                         states_test, id,
                                                                         id2one_hot,
                                                                         short=True,
                                                                         treshold=50)
        s_test = np.array(s_test)
        r_test = np.array(r_test)
        r_test = r_test.reshape([len(r_test), 1])
        acc_goal, prec_goal, recal_goal = session.run(metrics, feed_dict={reward_func.S: s_test,
                                                                          reward_func.I: one_hot_test,
                                                                          reward_func.Y: r_test})
        f1_goal = 2 * prec_goal * recal_goal / (prec_goal + recal_goal)

        f1_dict[id].append(f1_goal)
        metrics_dict[id].append((acc_goal, prec_goal, recal_goal))
        logging.info(id2description[id] + ':   f1_ score is: ' + str(f1_goal))
    return f1_dict, metrics_dict


def init_metric_dicts(descriptions_id_states, descriptions_id_language):
    train_metrics_log = {'accuracy': [], 'cost': [], 'precision': []}
    metrics_dict_states = dict(zip(descriptions_id_states, [[] for _ in range(len(descriptions_id_states))]))
    metrics_dict_language = dict(zip(descriptions_id_language, [[] for _ in range(len(descriptions_id_language))]))
    f1_dict_states = dict(zip(descriptions_id_states, [[] for _ in range(len(descriptions_id_states))]))
    f1_dict_language = dict(zip(descriptions_id_language, [[] for _ in range(len(descriptions_id_language))]))
    return train_metrics_log, metrics_dict_states, metrics_dict_language, f1_dict_states, f1_dict_language


def evaluate_metrics_pytorch(model, descriptions_id, state_idx_reward_buffer, states_test, id2one_hot,
                             id2description,
                             f1_dict, metrics_dict, logging):
    logging.info('Evaluating')
    model.eval()
    with torch.no_grad():
        for id in list(descriptions_id):
            s_test, one_hot_test, r_test = generate_test_set_for_instruction(state_idx_reward_buffer,
                                                                             states_test, id,
                                                                             id2one_hot,
                                                                             short=True,
                                                                             treshold=50)
            batch_s = torch.tensor(s_test, dtype=torch.float32)
            batch_descr = torch.tensor(one_hot_test, dtype=torch.float32)
            batch_r = torch.tensor(r_test, dtype=torch.float32)
            batch_pred_proba = model(batch_s, batch_descr)

            y_pred = (batch_pred_proba > 0.5).to(torch.float32).squeeze()
            y_true = batch_r
            tp = torch.mul(y_true, y_pred).sum().to(torch.float32)
            tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
            fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
            fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
            accuracy = ((tp + tn) / (tp + tn + fp + fn)).float()
            precision = (tp / (tp + fp)).float()
            recall = (tp / (tp + fn)).float()
            f1 = 2 * precision * recall / (precision + recall)
            f1_dict[id].append(f1)
            metrics_dict[id].append((accuracy, precision, recall))
            logging.info(id2description[id] + ':   f1_ score is: ' + str(f1))
            logging.info(str(accuracy) + ', ' + str(precision) + ', ' + str(recall))
            print(id2description[id] + ':   f1_ score is: ' + str(f1))
            print(str(accuracy) + ', ' + str(precision) + ', ' + str(recall))
    return f1_dict, metrics_dict
