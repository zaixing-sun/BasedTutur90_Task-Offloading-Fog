import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions import Categorical  # (optional for epsilon random selection)

from policies.model.TaskFormer import TaskFormer

import numpy as np

from core.env import Env
from core.task import Task

class TaskFormerPolicy:
    def __init__(self, env, config):
        """
        A simple deep Q-learning policy.

        Args:
            env: The simulation environment.
            config (dict): A configuration dictionary containing:
                - training: with keys 'lr', 'gamma', 'epsilon'
                - model: with key 'd_model' (used as the hidden size)
        """
        self.env = env
        
        self.n_observations = len(self._make_observation(env, None)[0])
        
        self.d_obs = len(self._make_observation(env, None)[0][0])
        
        
        
        self.num_actions = len(env.scenario.node_id2name)

        # Retrieve configuration parameters.
        self.gamma = config["training"]["gamma"]
        self.epsilon = config["training"]["epsilon"]
        self.lr = config["training"]["lr"]
        self.beta = config["training"]["beta"]
        
        
        
        d_model = config["model"]["d_model"]
        n_layers = config["model"]["n_layers"]
        n_heads = config["model"]["n_heads"]
        mlp_ratio = config["model"]["mlp_ratio"]
        dropout = config["model"]["dropout"]
        mode = config["model"]["mode"]
        
        
        self.model = TaskFormer(d_in=self.d_obs, d_pos=self.n_observations, d_task=4, d_model=d_model, d_ff=d_model*mlp_ratio, n_heads=n_heads, n_layers=n_layers, dropout=dropout, mode=mode)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Replay buffer for transitions.
        self.replay_buffer = []

    def _make_observation(self, env, task):
        """
        Returns a flat observation vector.
        For instance, we return the free CPU frequency for each node.
        """
        cpu_obs = {node_name: env.scenario.get_node(node_name).free_cpu_freq 
               for node_name in env.scenario.get_nodes()}
        
        buffer_obs = {node_name: env.scenario.get_node(node_name).buffer_free_size()
               for node_name in env.scenario.get_nodes()}
        # print(env.scenario.get_links())
        bw_obs = {link_name:env.scenario.get_link(link_name[0], link_name[1]).free_bandwidth for link_name in env.scenario.get_links()}
        
        obs = np.zeros((len(env.scenario.get_nodes()), 4))
        
        for i, node_name in enumerate(env.scenario.get_nodes()):
            obs[env.scenario.node_name2id[node_name], 0] = cpu_obs[node_name]
            obs[env.scenario.node_name2id[node_name], 1] = buffer_obs[node_name]

        for i, link_name in enumerate(env.scenario.get_links()):
            
            if link_name[0] == 'e0':
                obs[env.scenario.node_name2id[link_name[1]], 2] = bw_obs[link_name]
            else:
                obs[env.scenario.node_name2id[link_name[0]], 3] = bw_obs[link_name]
        
        if task is None:
            task_obs = [0, 0, 0, 0]
        else:
            task_obs = [
                task.task_size,
                task.cycles_per_bit,
                task.trans_bit_rate,
                task.ddl,
            ]
        
        return obs, task_obs

    def act(self, env, task, train=True):
        """
        Chooses an action using an ε-greedy strategy and records the current state.
        """
        state = self._make_observation(env, task)
        obs, task_obs = state
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        task_tensor = torch.tensor(task_obs, dtype=torch.float32).unsqueeze(0)

        rand = random.random()
        
        if rand < self.epsilon and train:
            action = random.randrange(self.num_actions)
        elif rand - self.epsilon < self.beta * (1-self.epsilon) and train:
            with torch.no_grad():
                q_values = self.model(obs_tensor, task_tensor, False)
                action = torch.argmax(q_values, dim=1).item()
        else:
            with torch.no_grad():
                q_values = self.model(obs_tensor, task_tensor, True)
                action = torch.argmax(q_values, dim=1).item()


        # Return both the chosen action and the current state.
        return action, state

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def update(self):
        """
        Performs an update over all stored transitions and clears the buffer.
        """
        if not self.replay_buffer:
            return 0.0
        
        loss_total = 0.0
        self.optimizer.zero_grad()
        for state, action, reward, next_state, done in self.replay_buffer:
            obs, task_obs = state
            next_obs, next_task_obs = next_state
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            task_tensor = torch.tensor(task_obs, dtype=torch.float32).unsqueeze(0)
            next_task_tensor = torch.tensor(next_task_obs, dtype=torch.float32).unsqueeze(0)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            done_tensor = torch.tensor(done, dtype=torch.float32)
            
            q_values = self.model(obs_tensor, task_tensor).squeeze()
            predicted_q = q_values[action]
            with torch.no_grad():
                next_q_values = self.model(next_obs_tensor, next_task_tensor)
                max_next_q = torch.max(next_q_values)
                if self.gamma == 0:
                    target_q = reward_tensor
                else:
                    target_q = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q
            loss = self.criterion(predicted_q, target_q)
            loss_total += loss
        loss_total.backward()
        self.optimizer.step()
        self.replay_buffer.clear()
        return loss_total.item()
