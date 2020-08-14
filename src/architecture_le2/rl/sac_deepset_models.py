import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
LATENT = 3


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class AttentionNetwork(nn.Module):
    def __init__(self, inp, hid, out):
        super(AttentionNetwork, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        # self.linear2 = nn.Linear(hid, hid)
        self.linear2 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        # x = self.linear2(x)
        x = nn.Sigmoid()(self.linear2(x))

        return x


class SinglePhiActor(nn.Module):
    def __init__(self, inp, hid, out):
        super(SinglePhiActor, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)
        # self.linear3 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.relu(self.linear2(x))
        # x = torch.tanh(self.linear3(x))

        return x


class RhoActor(nn.Module):
    def __init__(self, inp, out, action_space=None):
        super(RhoActor, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        # self.linear2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, out)
        self.log_std_linear = nn.Linear(256, out)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, x):
        # x = torch.tanh(inp)
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, eval=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class SinglePhiCritic(nn.Module):
    def __init__(self, inp, hid, out):
        super(SinglePhiCritic, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)
        # self.linear3 = nn.Linear(hid, out)

        self.linear4 = nn.Linear(inp, hid)
        self.linear5 = nn.Linear(hid, out)
        # self.linear6 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp1,inp2):
        x1 = F.relu(self.linear1(inp1))
        x1 = F.relu(self.linear2(x1))
        # x1 = F.relu(self.linear3(x1))

        x2 = F.relu(self.linear4(inp2))
        x2 = F.relu(self.linear5(x2))
        # x2 = F.relu(self.linear6(x2))

        return x1, x2


class RhoCritic(nn.Module):
    def __init__(self, inp, out, action_space=None):
        super(RhoCritic, self).__init__()
        self.linear1 = nn.Linear(inp, 256)  # Added one layer (To Check)
        # self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, out)

        self.linear4 = nn.Linear(inp, 256)
        # self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, inp1, inp2):
        # x1 = torch.tanh(self.linear1(inp1))
        x1 = F.relu(self.linear1(inp1))
        # x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # x2 = torch.tanh(self.linear2(inp2))
        x2 = F.relu(self.linear4(inp2))
        # x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class DeepSetSAC:
    def __init__(self, dims, layers, hidden):

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

        self.observation = None
        self.ag = None
        self.g = None

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # Define dimensions according to parameters use_attention
        # if attention not used, then concatenate [g, ag] in input ==> dimension = 2 * dim_goal

        dim_phi_actor_input = self.dim_body + self.dim_obj
        dim_phi_actor_output = self.n_objs * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dimu

        dim_phi_critic_input = self.dim_body + self.dim_obj + self.dimu
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.attention_actor = AttentionNetwork(self.dimg, hidden,
                                                self.dim_body + self.dim_obj)
        self.attention_critic_1 = AttentionNetwork(self.dimg, hidden,
                                                   self.dim_body + self.dim_obj + self.dimu)
        self.attention_critic_2 = AttentionNetwork(self.dimg, hidden,
                                                   self.dim_body + self.dim_obj + self.dimu)

        self.single_phi_actor = SinglePhiActor(dim_phi_actor_input, hidden, dim_phi_actor_output)

        self.rho_actor = RhoActor(dim_rho_actor_input, dim_rho_actor_output)

        self.single_phi_critic = SinglePhiCritic(dim_phi_critic_input, hidden, dim_phi_critic_output)
        self.rho_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

        self.attention_target_critic_1 = AttentionNetwork(self.dimg, hidden,
                                                          self.dim_body + self.dim_obj + self.dimu)
        self.attention_target_critic_2 = AttentionNetwork(self.dimg, hidden,
                                                          self.dim_body + self.dim_obj + self.dimu)
        self.single_phi_target_critic = SinglePhiCritic(dim_phi_critic_input, hidden, dim_phi_critic_output)
        self.rho_target_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

    def policy_forward_pass(self, o, g, eval=False):
        self.observation = o
        self.g = g
        obs_body = self.observation[:, :self.dim_body]
        obs_objects = [torch.cat(tensors=[o[:, self.inds_objs[i][0]: self.inds_objs[i][-1] + 1],
                                          o[:,
                                          self.inds_objs[i][0] + self.half_o: self.inds_objs[i][-1] + 1 + self.half_o]],
                                 dim=1) for i in range(self.n_objs)]

        output_attention_actor = self.attention_actor(self.g)

        # Parallelization by stacking input tensors
        input_actor_obs_attention = torch.stack(
            [torch.cat([obs_body, x], dim=1) * output_attention_actor for x in obs_objects])
        output_phi_actor = self.single_phi_actor(input_actor_obs_attention).sum(dim=0)
        if not eval:
            self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.rho_actor.sample(output_phi_actor)

        return self.pi_tensor.squeeze()

    def compute_q_values(self, o, g, action):
        output_attention_critic_1 = self.attention_critic_1(g)
        output_attention_critic_2 = self.attention_critic_2(g)
        obs_body = self.observation[:, :self.dim_body]
        obs_objects = [torch.cat(tensors=[o[:, self.inds_objs[i][0]: self.inds_objs[i][-1] + 1],
                                          o[:,
                                          self.inds_objs[i][0] + self.half_o: self.inds_objs[i][-1] + 1 + self.half_o]],
                                 dim=1) for i in range(self.n_objs)]

        # Parallelization by stacking input tensors
        action = action.expand([1,len(action)])
        input_critic_1 = torch.stack(
            [torch.cat([obs_body, x, action], dim=1) * output_attention_critic_1 for x in obs_objects])
        input_critic_2 = torch.stack(
            [torch.cat([obs_body, x, action], dim=1) * output_attention_critic_2 for x in obs_objects])

        with torch.no_grad():
            output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(input_critic_1,
                                                                              input_critic_2)
            output_phi_critic_1_aggregated = output_phi_critic_1.sum(dim=0)
            output_phi_critic_2_aggregated = output_phi_critic_2.sum(dim=0)
            q_value_1, q_value_2 = self.rho_critic(output_phi_critic_1_aggregated, output_phi_critic_2_aggregated)

        return torch.cat([q_value_1, q_value_2])

    def forward_pass(self, o, g, eval=False, actions=None):
        batch_size = o.shape[0]
        self.observation = o
        self.g = g
        obs_body = self.observation[:, :self.dim_body]

        obs_objects = [torch.cat(tensors=[o[:, self.inds_objs[i][0]: self.inds_objs[i][-1] + 1],
                                          o[:,
                                          self.inds_objs[i][0] + self.half_o: self.inds_objs[i][-1] + 1 + self.half_o]],
                                 dim=1) for i in range(self.n_objs)]

        # Pass through the attention network
        output_attention_actor = self.attention_actor(self.g)
        output_attention_critic_1 = self.attention_critic_1(self.g)
        output_attention_critic_2 = self.attention_critic_2(self.g)

        # Parallelization by stacking input tensors
        input_actor_obs_attention = torch.stack(
            [torch.cat([obs_body, x], dim=1) * output_attention_actor for x in obs_objects])

        output_phi_actor = self.single_phi_actor(input_actor_obs_attention).sum(dim=0)
        if not eval:
            self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.rho_actor.sample(output_phi_actor)


        # The critic part
        input_critic_1 = torch.stack(
            [torch.cat([obs_body, x, self.pi_tensor], dim=1) * output_attention_critic_1 for x in obs_objects])
        input_critic_2 = torch.stack(
            [torch.cat([obs_body, x, self.pi_tensor], dim=1) * output_attention_critic_2 for x in obs_objects])


        # Compute target Q values
        with torch.no_grad():
            # Todo add target attention network
            output_target_attention_critic_1 = self.attention_target_critic_1(g)
            output_target_attention_critic_2 = self.attention_target_critic_2(g)
            input_target_critic_1 = torch.stack(
                [torch.cat([obs_body, x, self.pi_tensor], dim=1) * output_target_attention_critic_1 for x in obs_objects])
            input_target_critic_2 = torch.stack(
                [torch.cat([obs_body, x, self.pi_tensor], dim=1) * output_target_attention_critic_2 for x in obs_objects])
            output_phi_target_critic_1, output_phi_target_critic_2 = self.single_phi_target_critic(
                input_target_critic_1, input_target_critic_2)
            output_phi_target_critic_1 = output_phi_target_critic_1.sum(dim=0)
            output_phi_target_critic_2 = output_phi_target_critic_2.sum(dim=0)
            self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.rho_target_critic(output_phi_target_critic_1,
                                                                                        output_phi_target_critic_2)
        # Compute main Q values
        output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(input_critic_1, input_critic_2)
        output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        self.q1_pi_tensor, self.q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        return self.q1_pi_tensor, self.q2_pi_tensor
