import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from agents.models.linear_net import LinearNet
from agents.memory import ReplayMemory


class DQNAgent:
    def __init__(self, config):
        self.learning_rate = config["agent"]["learning_rate"]
        self.gamma = config["agent"]["gamma"]
        self.epsilon = config["agent"]["epsilon_start"]
        self.epsilon_min = config["agent"]["epsilon_min"]
        self.epsilon_decay = config["agent"]["epsilon_decay"]
        self.batch_size = config["agent"]["batch_size"]

        self.model = LinearNet()
        self.memory = ReplayMemory(capacity=config["agent"]["memory_size"])

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        expl_prob = random.random()
        if expl_prob < self.epsilon:
            return random.randint(0, 3)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
            return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.int).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)

        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)

        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, file_path, is_training=False):
        self.model.load_state_dict(torch.load(file_path))

        if not is_training:
            self.model.eval()
            self.epsilon = 0.0
        else:
            self.model.train()
            self.epsilon = max(self.epsilon_min, 0.05)
