import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""
# Parameters for sampling in StochasticActor
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# TODO put this in utils because it is used everywhere (Reward func, DPPG, SAC)
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, layers_sizes):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.layers_sizes = layers_sizes
        self.fc_layers = nn.ModuleList()
        self.activations = []

        size_tmp = self.input_size
        for i, size in enumerate(self.layers_sizes):
            self.activations.append(nn.ReLU() if i < len(self.layers_sizes) - 1 else None)
            fc = torch.nn.Linear(size_tmp, size)
            nn.init.kaiming_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
            self.fc_layers.append(fc)
            size_tmp = size

    def forward(self, input):
        for fc, activation in zip(self.fc_layers, self.activations):
            input = fc(input)
            if activation:
                input = activation(input)
        return input


class Actor(nn.Module):
    def __init__(self, dims, layers, hidden):
        super(Actor, self).__init__()
        self.layers = layers
        self.dimo = dims['obs']
        self.dimg = dims['g_encoding']
        self.dimu = dims['acts']
        self.inds_objs = dims['inds_objs']
        self.hidden = hidden

        self.half_o = self.dimo // 2
        self.n_objs = len(self.inds_objs)
        self.dim_obj = 2 * len(self.inds_objs[0])
        self.dim_body = self.inds_objs[0][0] * 2

        self.fc_cast = Feedforward(self.dimg, [self.dim_body + self.dim_obj])
        self.fc_actor = Feedforward(self.dim_body + self.dim_obj,
                                    [self.hidden] + [self.n_objs * (self.dim_obj + self.dim_body)])
        self.fc_pi = Feedforward(self.n_objs * (self.dim_obj + self.dim_body),
                                 [self.hidden] * self.layers + [self.dimu])

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, o, g):
        attention = F.sigmoid(self.fc_cast(g))
        obs_body = torch.cat(tensors=[o[:, :self.inds_objs[0][0]],
                                      o[:, self.half_o: self.half_o + self.inds_objs[0][0]]], dim=1)
        input_pi = torch.zeros([len(o), self.n_objs * (self.dim_obj + self.dim_body)])
        for i in range(self.n_objs):
            obs_obj = torch.cat(tensors=[o[:, self.inds_objs[i][0]: self.inds_objs[i][-1] + 1],
                                         o[:,
                                         self.inds_objs[i][0] + self.half_o: self.inds_objs[i][-1] + 1 + self.half_o]],
                                dim=1)
            body_obj_input = torch.cat(dim=1, tensors=[obs_body, obs_obj])
            deepset_input = torch.mul(body_obj_input, attention)

            input_obj = F.relu(self.fc_actor(deepset_input))
            input_pi += input_obj

        return self.tanh(self.fc_pi(input_pi))

    def get_attention(self, g):
        return self.sigmoid(self.fc_cast(g))

    def load_from_tf_params(self, params_dict, name='main'):
        self.fc_cast.fc_layers[0].weight = torch.nn.Parameter(
            torch.tensor(np.transpose(params_dict['ddpg/{}/pi/attention_0/kernel:0'.format(name)]),
                         dtype=torch.float32))
        self.fc_cast.fc_layers[0].bias = torch.nn.Parameter(
            torch.tensor(params_dict['ddpg/{}/pi/attention_0/bias:0'.format(name)], dtype=torch.float32))
        self.fc_actor.fc_layers[0].weight = torch.nn.Parameter(
            torch.tensor(np.transpose(params_dict['ddpg/{}/pi/obj_0/kernel:0'.format(name)]), dtype=torch.float32))
        self.fc_actor.fc_layers[0].bias = torch.nn.Parameter(
            torch.tensor(params_dict['ddpg/{}/pi/obj_0/bias:0'.format(name)], dtype=torch.float32))
        self.fc_actor.fc_layers[1].weight = torch.nn.Parameter(
            torch.tensor(np.transpose(params_dict['ddpg/{}/pi/obj_1/kernel:0'.format(name)]), dtype=torch.float32))
        self.fc_actor.fc_layers[1].bias = torch.nn.Parameter(
            torch.tensor(params_dict['ddpg/{}/pi/obj_1/bias:0'.format(name)], dtype=torch.float32))
        self.fc_pi.fc_layers[0].weight = torch.nn.Parameter(
            torch.tensor(np.transpose(params_dict['ddpg/{}/pi/pi_0/kernel:0'.format(name)]), dtype=torch.float32))
        self.fc_pi.fc_layers[0].bias = torch.nn.Parameter(
            torch.tensor(params_dict['ddpg/{}/pi/pi_0/bias:0'.format(name)], dtype=torch.float32))
        self.fc_pi.fc_layers[1].weight = torch.nn.Parameter(
            torch.tensor(np.transpose(params_dict['ddpg/{}/pi/pi_1/kernel:0'.format(name)]), dtype=torch.float32))
        self.fc_pi.fc_layers[1].bias = torch.nn.Parameter(
            torch.tensor(params_dict['ddpg/{}/pi/pi_1/bias:0'.format(name)], dtype=torch.float32))


class StochasticActor(nn.Module):
    def __init__(self, dims, layers, hidden):
        super(StochasticActor, self).__init__()
        self.layers = layers
        self.dimo = dims['obs']
        self.dimg = dims['g_encoding']
        self.dimu = dims['acts']
        self.inds_objs = dims['inds_objs']
        self.hidden = hidden

        self.half_o = self.dimo // 2
        self.n_objs = len(self.inds_objs)
        self.dim_obj = 2 * len(self.inds_objs[0])
        self.dim_body = self.inds_objs[0][0] * 2

        self.fc_cast = Feedforward(self.dimg, [self.dim_body + self.dim_obj])
        self.fc_actor = Feedforward(self.dim_body + self.dim_obj,
                                    [self.hidden] + [self.n_objs * (self.dim_obj + self.dim_body)])
        self.fc_linear = Feedforward(self.n_objs * (self.dim_obj + self.dim_body), [self.hidden])
        self.fc_mean = Feedforward(self.hidden, [self.dimu])
        self.fc_log_std = Feedforward(self.hidden, [self.dimu])
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, o, g):
        attention = F.sigmoid(self.fc_cast(g))
        obs_body = torch.cat(tensors=[o[:, :self.inds_objs[0][0]],
                                      o[:, self.half_o: self.half_o + self.inds_objs[0][0]]], dim=1)
        input_pi = torch.zeros([len(o), self.n_objs * (self.dim_obj + self.dim_body)])
        for i in range(self.n_objs):
            obs_obj = torch.cat(tensors=[o[:, self.inds_objs[i][0]: self.inds_objs[i][-1] + 1],
                                         o[:,
                                         self.inds_objs[i][0] + self.half_o: self.inds_objs[i][-1] + 1 + self.half_o]],
                                dim=1)
            body_obj_input = torch.cat(dim=1, tensors=[obs_body, obs_obj])
            deepset_input = torch.mul(body_obj_input, attention)

            input_obj = F.relu(self.fc_actor(deepset_input))
            input_pi += input_obj

        latent = self.fc_linear(input_pi)
        mean = self.fc_mean(latent)
        log_std = self.fc_log_std(latent)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, o, g, eval=False):
        mean, log_std = self.forward(o, g)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        if eval:
            return torch.tanh(mean), log_prob
        else:
            return action, log_prob

    def get_attention(self, g):
        return self.sigmoid(self.fc_cast(g))


class Critic(nn.Module):
    def __init__(self, dims, layers, hidden):
        super(Critic, self).__init__()
        self.layers = layers
        self.dimo = dims['obs']
        self.dimg = dims['g_encoding']
        self.dimu = dims['acts']
        self.inds_objs = dims['inds_objs']
        self.hidden = hidden

        self.half_o = self.dimo // 2
        self.n_objs = len(self.inds_objs)
        self.dim_obj = 2 * len(self.inds_objs[0])
        self.dim_body = self.inds_objs[0][0] * 2

        self.fc_cast = Feedforward(self.dimg, [self.dim_body + self.dim_obj + self.dimu])
        self.fc_critic = Feedforward(self.dim_body + self.dim_obj + self.dimu,
                                     [self.hidden] + [self.n_objs * (self.dim_obj + self.dim_body + self.dimu)])
        self.fc_Q = Feedforward(self.n_objs * (self.dim_obj + self.dim_body + self.dimu),
                                [self.hidden] * self.layers + [1])

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, o, g, actions):
        attention = F.sigmoid(self.fc_cast(g))
        obs_body = torch.cat(tensors=[o[:, :self.inds_objs[0][0]],
                                      o[:, self.half_o: self.half_o + self.inds_objs[0][0]]], dim=1)
        input_Q = torch.zeros([len(o), self.n_objs * (self.dim_obj + self.dim_body + self.dimu)])
        for i in range(self.n_objs):
            obs_obj = torch.cat(tensors=[o[:, self.inds_objs[i][0]: self.inds_objs[i][-1] + 1],
                                         o[:,
                                         self.inds_objs[i][0] + self.half_o: self.inds_objs[i][-1] + 1 + self.half_o]],
                                dim=1)
            body_obj_act_input = torch.cat(dim=1, tensors=[obs_body, obs_obj, actions])
            deepset_input = torch.mul(body_obj_act_input, attention)

            input_obj = F.relu(self.fc_critic(deepset_input))
            input_Q += input_obj

        return self.fc_Q(input_Q)

    def get_attention(self, g):
        return self.sigmoid(self.fc_cast(g))

    def load_from_tf_params(self, params_dict, name='main'):
        self.fc_cast.fc_layers[0].weight = torch.nn.Parameter(
            torch.tensor(np.transpose(params_dict['ddpg/{}/Q/attention_0/kernel:0'.format(name)]),
                         dtype=torch.float32))
        self.fc_cast.fc_layers[0].bias = torch.nn.Parameter(
            torch.tensor(params_dict['ddpg/{}/Q/attention_0/bias:0'.format(name)], dtype=torch.float32))
        self.fc_critic.fc_layers[0].weight = torch.nn.Parameter(
            torch.tensor(np.transpose(params_dict['ddpg/{}/Q/obj_0/kernel:0'.format(name)]), dtype=torch.float32))
        self.fc_critic.fc_layers[0].bias = torch.nn.Parameter(
            torch.tensor(params_dict['ddpg/{}/Q/obj_0/bias:0'.format(name)], dtype=torch.float32))
        self.fc_critic.fc_layers[1].weight = torch.nn.Parameter(
            torch.tensor(np.transpose(params_dict['ddpg/{}/Q/obj_1/kernel:0'.format(name)]), dtype=torch.float32))
        self.fc_critic.fc_layers[1].bias = torch.nn.Parameter(
            torch.tensor(params_dict['ddpg/{}/Q/obj_1/bias:0'.format(name)], dtype=torch.float32))
        self.fc_Q.fc_layers[0].weight = torch.nn.Parameter(
            torch.tensor(np.transpose(params_dict['ddpg/{}/Q/critic_0/kernel:0'.format(name)]), dtype=torch.float32))
        self.fc_Q.fc_layers[0].bias = torch.nn.Parameter(
            torch.tensor(params_dict['ddpg/{}/Q/critic_0/bias:0'.format(name)], dtype=torch.float32))
        self.fc_Q.fc_layers[1].weight = torch.nn.Parameter(
            torch.tensor(np.transpose(params_dict['ddpg/{}/Q/critic_1/kernel:0'.format(name)]), dtype=torch.float32))
        self.fc_Q.fc_layers[1].bias = torch.nn.Parameter(
            torch.tensor(params_dict['ddpg/{}/Q/critic_1/bias:0'.format(name)], dtype=torch.float32))
