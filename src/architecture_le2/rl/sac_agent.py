import torch
from collections import deque
import time
import numpy as np
import pickle
from src.architecture_le2.rl.mpi_utils.mpi_utils import sync_networks, sync_grads
from src.architecture_le2.rl.replay_buffer import ReplayBuffer
from src.architecture_le2.rl.sac_deepset_models import DeepSetSAC
from src.architecture_le2.rl.mpi_utils.normalizer import Normalizer
import torch.nn.functional as F

"""
ddpg with HER (MPI-version)
​
"""


class SACAgent:
    def __init__(self, params, sample_transitions):
        params = params.copy()
        params['make_env'] = None

        self.dims = params['dims']
        self.cuda = params['cuda']
        self.logdir = params['logdir']
        self.clip_range = params['clip_range']
        # self.positive_ratio = params['positive_ratio'] defined in her sample transition
        self.lr_actor = params['lr_actor']
        self.lr_critic = params['lr_critic']
        self.buffer_size = params['buffer_size']
        # self.env_name = params['env_name']
        self.random_eps = params['random_eps']
        self.noise_eps = params['noise_eps']
        self.polyak = params['polyak']
        self.clip_obs = params['clip_obs']
        self.action_l2 = params['action_l2']
        self.normalize = params['normalize_obs']
        # self.batch_size = params['batch_size'] defined in her
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.model_path = params['logdir'] + '/policies/'
        self.T = params['T']
        self.reward_function = params['reward_function']
        self.bias_buffer = params['bias_buffer']
        self.sample_transitions = sample_transitions
        self.batch_size = params['batch_size']

        hidden = params['hidden']
        layers = params['layers']

        # create the replay buffer
        buffer_shapes = dict(acts=(self.T, self.dims['acts']),
                             obs=(self.T + 1, self.dims['obs']),
                             g_id=(self.dims['g_id'],),
                             g_encoding=(1, self.dims['g_encoding']),
                             )
        self.buffer = ReplayBuffer(buffer_shapes,
                                   self.buffer_size,
                                   self.T,
                                   self.sample_transitions,
                                   params['goal_sampler'],
                                   self.reward_function,
                                   self.bias_buffer)

        self.replay_proba = None
        self.memory_replay_ratio_positive_rewards = deque(maxlen=50)
        self.memory_replay_ratio_positive_per_goal = deque(maxlen=50)

        self.model = DeepSetSAC(params['dims'], layers, hidden)
        # sync the networks across the CPUs
        sync_networks(self.model.rho_actor)
        sync_networks(self.model.rho_critic)
        sync_networks(self.model.single_phi_actor)
        sync_networks(self.model.single_phi_critic)
        sync_networks(self.model.attention_actor)
        sync_networks(self.model.attention_critic_1)
        sync_networks(self.model.attention_critic_2)

        self.model.attention_target_critic_1.load_state_dict(self.model.attention_critic_1.state_dict())
        self.model.attention_target_critic_2.load_state_dict(self.model.attention_critic_2.state_dict())
        self.model.single_phi_target_critic.load_state_dict(self.model.single_phi_critic.state_dict())
        self.model.rho_target_critic.load_state_dict(self.model.rho_critic.state_dict())
        sync_networks(self.model.attention_target_critic_1)
        sync_networks(self.model.attention_target_critic_2)
        sync_networks(self.model.single_phi_target_critic)
        sync_networks(self.model.rho_target_critic)

        self.policy_optim = torch.optim.Adam(list(self.model.single_phi_actor.parameters()) +
                                             list(self.model.rho_actor.parameters()) +
                                             list(self.model.attention_actor.parameters()),
                                             lr=self.lr_actor)

        self.critic_optim = torch.optim.Adam(list(self.model.single_phi_critic.parameters()) +
                                             list(self.model.rho_critic.parameters()) +
                                             list(self.model.attention_critic_1.parameters()) +
                                             list(self.model.attention_critic_2.parameters()),
                                             lr=self.lr_critic)

        # create the normalizer
        self.o_norm = Normalizer(size=params['dims']['obs'], default_clip_range=self.clip_range)
        self.g_norm = Normalizer(size=params['dims']['g_encoding'], default_clip_range=self.clip_range)

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'o_norm', 'g_norm', 'env', 'sample_transitions', 'params',
                             'goal_sampler',
                             'stage_shapes', 'create_actor_critic', 'reward_function', 'rollout']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size

        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        if self.normalize:
            obs_norm = self.o_norm.normalize(obs)
            g_norm = self.g_norm.normalize(g)
            # concatenate the stuffs
            inputs = np.concatenate([obs_norm, g_norm])
        else:
            inputs = np.concatenate([obs, g])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.cuda:
            inputs = inputs.cuda()
        return inputs

    def _preproc_inputs_no_concat(self, obs, g):
        if self.normalize:
            obs_norm = self.o_norm.normalize(obs)
            g_norm = self.g_norm.normalize(g)
            # concatenate the stuffs
        else:
            obs_norm = obs
            g_norm = g
        obs_norm = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        g_norm = torch.tensor(g_norm, dtype=torch.float32).unsqueeze(0)
        if self.cuda:
            obs_norm = obs_norm.cuda()
            g_norm = g_norm.cuda()
        return obs_norm, g_norm

    def _random_action(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, self.dims['acts']))

    def get_actions(self, obs, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        with torch.no_grad():
            obs, g = self._preproc_inputs_no_concat(obs, g)
            if random_eps > 0:
                eval = False
            else:
                eval = True
            action = self.model.policy_forward_pass(obs, g, eval)
            if compute_Q:
                q_values = self.model.compute_q_values(obs, g, action)
                q_value = torch.min(q_values).numpy()
            u = action.cpu().numpy()
            action = u
            if compute_Q:
                return action, q_value
            else:
                return action

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

    def soft_update(self):
        self._soft_update_target_network(self.model.attention_target_critic_1, self.model.attention_critic_1)
        self._soft_update_target_network(self.model.attention_target_critic_2, self.model.attention_critic_2)
        self._soft_update_target_network(self.model.single_phi_target_critic,
                                         self.model.single_phi_critic)
        self._soft_update_target_network(self.model.rho_target_critic, self.model.rho_critic)

    def update(self, epoch):
        times_training = self._update_network(epoch)
        return times_training

    def save_model(self, epoch):
        if epoch % self.params['save_every'] == 0:
            torch.save(
                [self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                 self.model.attention_actor.state_dict(), self.model.single_phi_actor.state_dict(),
                 self.model.rho_actor.state_dict()], \
                self.model_path + '/model_{}.pt'.format(epoch))

    def store_episode(self, episodes, goals_reached_ids):
        self.buffer.store_episode(episodes, goals_reached_ids)

    def sample_batch(self, epoch):
        out = self.buffer.sample(self.batch_size, epoch)
        transitions, replay_ratio_positive_rewards, self.replay_proba, replay_ratio_positive_per_goal, time_dict = out
        self.memory_replay_ratio_positive_rewards.append(replay_ratio_positive_rewards)
        self.memory_replay_ratio_positive_per_goal.append(replay_ratio_positive_per_goal)

        # pre-process the observation and goal
        o, o_2, g = transitions['obs'], transitions['obs_2'], transitions['g_encoding']
        transitions['obs'], transitions['g_encoding'] = self._preproc_og(o, g)
        transitions['obs_2'], _ = self._preproc_og(o_2, g)
        return transitions, time_dict

    # update the network
    def _update_network(self, epoch):

        timee = time.time()
        # sample the episodes
        transitions, time_dict = self.sample_batch(epoch)
        # start to do the update

        if self.normalize:
            obs_norm = torch.tensor(self.o_norm.normalize(transitions['obs']), dtype=torch.float32)
            g_norm = torch.tensor(self.g_norm.normalize(transitions['g_encoding']), dtype=torch.float32)
            obs_next_norm = torch.tensor(self.o_norm.normalize(transitions['obs_2']), dtype=torch.float32)
            g_next_norm = torch.tensor(self.g_norm.normalize(transitions['g_encoding']), dtype=torch.float32)
        else:
            obs_norm = torch.tensor(transitions['obs'], dtype=torch.float32)
            g_norm = torch.tensor(transitions['g_encoding'], dtype=torch.float32)
            obs_next_norm = torch.tensor(transitions['obs_2'], dtype=torch.float32)
            g_next_norm = torch.tensor(transitions['g_encoding'], dtype=torch.float32)

        actions_tensor = torch.tensor(transitions['acts'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        if self.cuda:
            obs_norm = obs_norm.cuda()
            g_norm = g_norm.cuda()
            obs_next_norm = obs_next_norm.cuda()
            g_next_norm = g_next_norm.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function

        times_training = dict(time_batch=time.time() - timee)
        timee = time.time()
        with torch.no_grad():
            self.model.forward_pass(obs_next_norm, g_norm)
            actions_next, log_pi_next = self.model.pi_tensor, self.model.log_prob
            qf1_next_target, qf2_next_target = self.model.target_q1_pi_tensor, self.model.target_q2_pi_tensor
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_pi_next
            next_q_value = (r_tensor.squeeze() + self.gamma * min_qf_next_target.squeeze()).reshape(
                [self.batch_size, 1])

        qf1, qf2 = self.model.forward_pass(obs_norm, g_norm, actions=actions_tensor)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        pi, log_pi = self.model.pi_tensor, self.model.log_prob
        qf1_pi, qf2_pi = self.model.q1_pi_tensor, self.model.q2_pi_tensor
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        sync_grads(self.model.attention_actor)
        sync_grads(self.model.single_phi_actor)
        sync_grads(self.model.rho_actor)
        self.policy_optim.step()

        self.critic_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        sync_grads(self.model.attention_critic_1)
        sync_grads(self.model.single_phi_critic)
        sync_grads(self.model.rho_critic)
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        sync_grads(self.model.attention_critic_2)
        sync_grads(self.model.single_phi_critic)
        sync_grads(self.model.rho_critic)
        self.critic_optim.step()

        times_training.update(time_update=time.time() - timee, **time_dict)
        return times_training

    def get_replay_ratio_positive_reward_stat(self):
        return np.mean(self.memory_replay_ratio_positive_rewards)

    def get_replay_ratio_positive_per_goal_stat(self):
        return np.nanmean(self.memory_replay_ratio_positive_per_goal, axis=0)

    def logs(self, prefix=''):
        logs = []

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs
