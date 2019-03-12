import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from deepx import nn, T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = (
            nn.Reshape([32, 32, 6])
            >> nn.Convolution([5, 5, 16], strides=(2, 2)) >> nn.Relu()
            >> nn.Convolution([5, 5, 16], strides=(2, 2)) >> nn.Relu()
            >> nn.Reshape([8 * 8 * 16]) >> nn.Relu(128)
            >> nn.Tanh(action_dim)
        )
        self.l1(torch.ones([1, 32 * 32 * 6]))

        self.max_action = max_action

    def parameters(self):
        return self.l1.get_parameters()

    def forward(self, x):
        return self.max_action * self.l1(x)


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.conv_l1 = (
            nn.Reshape([32, 32, 6])
            >> nn.Convolution([5, 5, 16], strides=(2, 2)) >> nn.Relu()
            >> nn.Convolution([5, 5, 16], strides=(2, 2)) >> nn.Relu()
            >> nn.Reshape([8 * 8 * 16])
        )
        self.fc_l1 = nn.Relu(128) >> nn.Linear(1)
        self.conv_l1(torch.ones([1, 32 * 32 * 6]))
        self.fc_l1(torch.ones([1, 8 * 8 * 16 + action_dim]))

        # Q2 architecture
        self.conv_l2 = (
            nn.Reshape([32, 32, 6])
            >> nn.Convolution([5, 5, 16], strides=(2, 2)) >> nn.Relu()
            >> nn.Convolution([5, 5, 16], strides=(2, 2)) >> nn.Relu()
            >> nn.Reshape([8 * 8 * 16])
        )
        self.fc_l2 = nn.Relu(128) >> nn.Linear(1)
        self.conv_l2(torch.ones([1, 32 * 32 * 6]))
        self.fc_l2(torch.ones([1, 8 * 8 * 16 + action_dim]))

    def forward(self, x, u):
        x_ = self.conv_l1(x)
        xu = T.concat([x_, u], 1)
        x1 = self.fc_l1(xu)

        x_ = self.conv_l2(x)
        xu = T.concat([x_, u], 1)
        x2 = self.fc_l2(xu)
        return x1, x2


    def Q1(self, x, u):
        x_ = self.conv_l1(x)
        xu = T.concat([x_, u], 1)
        x1 = self.fc_l1(xu)
        return x1

    def parameters(self):
        return (
            self.conv_l1.get_parameters() + self.fc_l1.get_parameters()
            + self.conv_l2.get_parameters() + self.fc_l2.get_parameters()
        )


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
