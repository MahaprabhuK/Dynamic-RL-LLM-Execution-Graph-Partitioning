import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """
    Neural network used by the DQN agent.
    """

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network agent for dynamic LLM execution scheduling.
    """

    def __init__(
        self,
        state_size=5,
        action_size=3,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        memory_size=5000
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=memory_size)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = DQNNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_model = DQNNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """Store an experience in replay memory."""

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy exploration.
        """

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)

        return int(torch.argmax(q_values).item())

    def train_step(self, batch_size=32):
        """
        Train the DQN using a random batch
        from replay memory.
        """

        if len(self.memory) < batch_size:
            return None

        batch = random.sample(
            self.memory,
            batch_size
        )

        states = torch.tensor(
            np.array([x[0] for x in batch]),
            dtype=torch.float32
        ).to(self.device)

        actions = torch.tensor(
            [x[1] for x in batch],
            dtype=torch.long
        ).to(self.device)

        rewards = torch.tensor(
            [x[2] for x in batch],
            dtype=torch.float32
        ).to(self.device)

        next_states = torch.tensor(
            np.array([x[3] for x in batch]),
            dtype=torch.float32
        ).to(self.device)

        dones = torch.tensor(
            [x[4] for x in batch],
            dtype=torch.float32
        ).to(self.device)

        current_q = self.model(states).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():

            next_q = self.target_model(
                next_states
            ).max(1)[0]

            target_q = rewards + (
                1 - dones
            ) * self.gamma * next_q

        loss = self.loss_function(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        """Copy learned weights to the target network."""

        self.target_model.load_state_dict(
            self.model.state_dict()
        )
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"DQN model saved to: {path}")
        
    def load(self, path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.target_model.load_state_dict(
            self.model.state_dict()
        )
        print(f"DQN model loaded from: {path}")

if __name__ == "__main__":

    agent = DQNAgent()

    test_state = np.array(
        [36, 358, 36.746, 62.4, 97.4],
        dtype=np.float32
    )

    action = agent.choose_action(test_state)

    print("DQN agent initialized successfully.")
    print("State size:", agent.state_size)
    print("Action size:", agent.action_size)
    print("Selected action:", action)
    print("Exploration rate:", agent.epsilon)